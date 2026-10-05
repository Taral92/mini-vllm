"""Benchmark harness: time each generation stage and append rows to results.csv.

Usage (from the repo root):

    mini-vllm-benchmark --stages naive,kv --output-lens 32,64,128,256,512
    mini-vllm-benchmark --stages kv,static,continuous --batch-size 8
    mini-vllm-benchmark --stages static,continuous --batch-size 8 --workload mixed
    mini-vllm-benchmark --device cuda --iters 3          # Kaggle T4, README numbers

Stages:
- naive, kv            one request at a time (batch_size column = 1)
- static, continuous   --batch-size requests per forward pass. "uniform"
  workload: batch_size requests, all output_len long. "mixed": 4 x batch_size
  requests with lengths output_len x (1/8, 1, 1/4, 1/2, ...), all arriving
  at t=0; latency and TTFT then include time spent queueing. Stage is
  recorded as e.g. "continuous-mixed".

Rules this file enforces (see CLAUDE.md, "Benchmark discipline"):
- device sync before every timer stop
- warmup runs are discarded (3 by default)
- fixed seed, greedy decoding, EOS ignored so every request emits exactly
  ``output_len`` tokens and runs are comparable
"""

import argparse
import csv
import gc
import json
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch import device as TorchDevice

from engine.device import synchronize
from engine.engine import generate_cached, generate_naive
from engine.loader import MODEL_NAME, load_model
from engine import EngineConfig, LLMEngine, SeqStatus, generate_static

GenerateFn = Callable[..., list[int]]

# Stage name -> generate function with the signature
# fn(model, ids, max_new, eos_id, **sampling) -> list[int].
STAGES: dict[str, GenerateFn] = {
    "naive": generate_naive,
    "kv": generate_cached,
}
# Batched stages, run by run_batched_benchmark.
BATCHED_STAGES = ("static", "continuous")
WORKLOADS = ("uniform", "mixed")
# Output length pattern for the mixed workload, as fractions of output_len.
MIXED_FRACTIONS = (1 / 8, 1.0, 1 / 4, 1 / 2)

# Token ids are never negative, so passing this as eos_id means "never stop
# early": every request produces exactly output_len tokens.
IGNORE_EOS = -1

CSV_FIELDS = (
    "stage",
    "device",
    "batch_size",
    "output_len",
    "tokens_per_sec",
    "ttft_ms",
    "tpot_ms",
    "p50_ms",
    "p99_ms",
    "peak_mem_gb",
    "git_sha",
    "timestamp",
)

# A speedup this large between stages at the same length usually means a
# timer stopped before the device finished.
SUSPICIOUS_SPEEDUP = 20.0


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate latency and throughput for one stage at one output length."""

    stage: str
    device: str
    batch_size: int
    output_len: int
    request_count: int
    elapsed_seconds: float
    tokens_per_second: float
    ttft_ms: float
    tpot_ms: float
    p50_ms: float
    p99_ms: float
    peak_mem_gb: float

    def to_row(self, git_sha: str, timestamp: str) -> dict[str, object]:
        """Return the CSV row in ``CSV_FIELDS`` order."""
        return {
            "stage": self.stage,
            "device": self.device,
            "batch_size": self.batch_size,
            "output_len": self.output_len,
            "tokens_per_sec": self.tokens_per_second,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "p50_ms": self.p50_ms,
            "p99_ms": self.p99_ms,
            "peak_mem_gb": self.peak_mem_gb,
            "git_sha": git_sha,
            "timestamp": timestamp,
        }


class TimedModel:
    """Wrap a model so every forward pass records a synced finish time.

    The generate functions call ``model(...)`` once per emitted token, so the
    first finish time gives time-to-first-token and the rest give the decode
    rate, without touching the engine code.
    """

    def __init__(self, model: Any, device: TorchDevice) -> None:
        self._model = model
        self._device = device
        self.forward_end: list[float] = []
        self.peak_bytes = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._model(*args, **kwargs)
        synchronize(self._device)
        self.forward_end.append(time.perf_counter())
        if self._device.type == "mps":
            # MPS has no peak counter, so sample after every forward.
            self.peak_bytes = max(self.peak_bytes, torch.mps.current_allocated_memory())
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def reset(self) -> None:
        """Forget recorded forward times (peak memory is kept)."""
        self.forward_end.clear()


def load_prompts(path: Path) -> tuple[str, ...]:
    """Load benchmark prompts from a JSON list of strings.

    Raises:
        ValueError: The file is not a non-empty list of strings.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of strings")
    if not all(isinstance(item, str) and item for item in data):
        raise ValueError(f"{path}: every prompt must be a non-empty string")
    return tuple(data)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same method as numpy's default)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _reset_peak(device: TorchDevice) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _release_memory(device: TorchDevice) -> None:
    """Drop cached allocator blocks so one run's leftovers don't slow the next."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _peak_gb(device: TorchDevice, timed: TimedModel) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    if device.type == "mps":
        return timed.peak_bytes / 1e9
    # CPU: whole-process peak RSS (Python + libraries + model). Not comparable
    # to GPU rows. ru_maxrss is KiB on Linux and bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def run_benchmark(
    model: Any,
    prompt_ids: Sequence[Tensor],
    stage: str,
    output_len: int,
    device: TorchDevice,
    warmup: int = 3,
    warmup_len: int = 64,
    iters: int = 1,
    seed: int = 2026,
) -> BenchmarkResult:
    """Benchmark one stage at one output length.

    Each measured request is one prompt run to exactly ``output_len`` tokens
    with greedy decoding. ``iters`` repeats the whole prompt set.

    Raises:
        KeyError: ``stage`` is not in ``STAGES``.
        RuntimeError: A stage returned the wrong number of tokens.
    """
    if output_len < 1:
        raise ValueError("output_len must be at least 1")
    if not prompt_ids:
        raise ValueError("need at least one prompt")
    generate = STAGES[stage]
    timed = TimedModel(model, device)
    _release_memory(device)

    for index in range(warmup):
        ids = prompt_ids[index % len(prompt_ids)]
        generate(timed, ids, min(output_len, warmup_len), IGNORE_EOS, temperature=0)
    synchronize(device)
    _reset_peak(device)
    timed.peak_bytes = 0

    latencies: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    for _ in range(iters):
        for ids in prompt_ids:
            torch.manual_seed(seed)
            timed.reset()
            synchronize(device)
            start = time.perf_counter()
            tokens = generate(timed, ids, output_len, IGNORE_EOS, temperature=0)
            synchronize(device)
            end = time.perf_counter()

            if len(tokens) != output_len:
                raise RuntimeError(
                    f"stage {stage!r} returned {len(tokens)} tokens, "
                    f"expected {output_len}"
                )
            first = timed.forward_end[0]
            latencies.append(end - start)
            ttfts.append(first - start)
            if output_len > 1:
                tpots.append((end - first) / (output_len - 1))

    elapsed = sum(latencies)
    request_count = len(latencies)
    return BenchmarkResult(
        stage=stage,
        device=device.type,
        batch_size=1,
        output_len=output_len,
        request_count=request_count,
        elapsed_seconds=elapsed,
        tokens_per_second=request_count * output_len / elapsed,
        ttft_ms=statistics.fmean(ttfts) * 1000,
        tpot_ms=statistics.fmean(tpots) * 1000 if tpots else 0.0,
        p50_ms=percentile(latencies, 50) * 1000,
        p99_ms=percentile(latencies, 99) * 1000,
        peak_mem_gb=_peak_gb(device, timed),
    )


def workload_lengths(output_len: int, num_requests: int, workload: str) -> list[int]:
    """Per-request output lengths for a workload."""
    if workload == "uniform":
        return [output_len] * num_requests
    if workload == "mixed":
        return [
            max(1, round(output_len * MIXED_FRACTIONS[i % len(MIXED_FRACTIONS)]))
            for i in range(num_requests)
        ]
    raise ValueError(f"unknown workload {workload!r}; use {WORKLOADS}")


def _summarize(
    label: str,
    device: TorchDevice,
    batch_size: int,
    output_len: int,
    lengths: Sequence[int],
    elapsed: float,
    firsts: Sequence[float],
    finishes: Sequence[float],
    peak_gb: float,
) -> BenchmarkResult:
    """Build a result from per-request first-token and finish times (seconds from arrival)."""
    tpots = [
        (finish - first) / (n - 1)
        for first, finish, n in zip(firsts, finishes, lengths, strict=True)
        if n > 1
    ]
    return BenchmarkResult(
        stage=label,
        device=device.type,
        batch_size=batch_size,
        output_len=output_len,
        request_count=len(lengths),
        elapsed_seconds=elapsed,
        tokens_per_second=sum(lengths) / elapsed,
        ttft_ms=statistics.fmean(firsts) * 1000,
        tpot_ms=statistics.fmean(tpots) * 1000 if tpots else 0.0,
        p50_ms=percentile(finishes, 50) * 1000,
        p99_ms=percentile(finishes, 99) * 1000,
        peak_mem_gb=peak_gb,
    )


def _serve_static(
    model: TimedModel,
    prompts: Sequence[list[int]],
    lengths: Sequence[int],
    batch_size: int,
    device: TorchDevice,
    pad_id: int,
) -> tuple[float, list[float], list[float]]:
    """Serve requests in fixed groups; each group runs until its longest request is done."""
    firsts: list[float] = []
    finishes: list[float] = []
    synchronize(device)
    start = time.perf_counter()
    for offset in range(0, len(prompts), batch_size):
        group = list(prompts[offset : offset + batch_size])
        group_lens = lengths[offset : offset + batch_size]
        model.reset()
        rows = generate_static(
            model, group, max(group_lens), IGNORE_EOS, pad_id, temperature=0
        )
        synchronize(device)
        done = time.perf_counter() - start
        if any(len(row) != max(group_lens) for row in rows):
            raise RuntimeError("static batch returned the wrong number of tokens")
        first = model.forward_end[0] - start
        firsts.extend([first] * len(group))
        finishes.extend([done] * len(group))  # the group returns together
    return time.perf_counter() - start, firsts, finishes


def _serve_continuous(
    engine: LLMEngine,
    prompts: Sequence[list[int]],
    lengths: Sequence[int],
    device: TorchDevice,
    peak: list[int],
) -> tuple[float, list[float], list[float]]:
    """Queue every request at t=0 and step the engine until all are done."""
    ids = [f"bench-{i}" for i in range(len(prompts))]
    for request_id, prompt, length in zip(ids, prompts, lengths, strict=True):
        engine.add_request(request_id, prompt, max_new_tokens=length, ignore_eos=True)
    first: dict[str, float] = {}
    finish: dict[str, float] = {}
    synchronize(device)
    start = time.perf_counter()
    while engine.has_unfinished():
        emitted = engine.step()
        synchronize(device)
        now = time.perf_counter() - start
        if device.type == "mps":
            peak[0] = max(peak[0], torch.mps.current_allocated_memory())
        for request_id in emitted:
            first.setdefault(request_id, now)
            if engine.sequences[request_id].status is SeqStatus.FINISHED:
                finish[request_id] = now
    elapsed = time.perf_counter() - start
    for request_id, length in zip(ids, lengths, strict=True):
        seq = engine.sequences.pop(request_id)
        if len(seq.output_ids) != length:
            raise RuntimeError(
                f"{request_id} produced {len(seq.output_ids)} tokens, expected {length}"
            )
    return elapsed, [first[r] for r in ids], [finish[r] for r in ids]


def run_batched_benchmark(
    model: Any,
    prompt_ids: Sequence[list[int]],
    stage: str,
    output_len: int,
    device: TorchDevice,
    batch_size: int = 4,
    workload: str = "uniform",
    warmup: int = 3,
    warmup_len: int = 64,
    iters: int = 1,
    num_blocks: int = 1024,
    block_size: int = 16,
    pad_id: int = 0,
) -> BenchmarkResult:
    """Benchmark a batched stage: many requests, up to ``batch_size`` per forward.

    Requests cycle through ``prompt_ids``. All arrive at t=0, so TTFT and the
    p50/p99 latencies count from arrival (queueing included).

    Raises:
        ValueError: Unknown stage/workload, or bad sizes.
    """
    if stage not in BATCHED_STAGES:
        raise ValueError(f"unknown batched stage {stage!r}; use {BATCHED_STAGES}")
    if output_len < 1 or batch_size < 1:
        raise ValueError("output_len and batch_size must be at least 1")
    if not prompt_ids:
        raise ValueError("need at least one prompt")
    num_requests = batch_size if workload == "uniform" else 4 * batch_size
    lengths = workload_lengths(output_len, num_requests, workload)
    prompts = [list(prompt_ids[i % len(prompt_ids)]) for i in range(num_requests)]
    warm_prompts = prompts[:batch_size]
    warm_lens = [min(output_len, warmup_len)] * len(warm_prompts)
    label = stage if workload == "uniform" else f"{stage}-{workload}"
    _release_memory(device)

    timed = TimedModel(model, device)
    peak = [0]
    engine: LLMEngine | None = None
    if stage == "continuous":
        longest = max(len(p) for p in prompts) + output_len
        engine = LLMEngine(
            model,
            config=EngineConfig(
                max_batch_size=batch_size,
                max_tokens_per_batch=max(2048, longest),
                block_size=block_size,
                num_blocks=num_blocks,
            ),
        )

    def serve(
        ps: Sequence[list[int]], ls: Sequence[int]
    ) -> tuple[float, list[float], list[float]]:
        if engine is not None:
            return _serve_continuous(engine, ps, ls, device, peak)
        return _serve_static(timed, ps, ls, batch_size, device, pad_id)

    for _ in range(warmup):
        serve(warm_prompts, warm_lens)
    synchronize(device)
    _reset_peak(device)
    timed.peak_bytes = 0
    peak[0] = 0

    elapsed = 0.0
    firsts: list[float] = []
    finishes: list[float] = []
    for _ in range(iters):
        run_elapsed, run_firsts, run_finishes = serve(prompts, lengths)
        elapsed += run_elapsed
        firsts.extend(run_firsts)
        finishes.extend(run_finishes)

    if device.type == "mps":
        peak_gb = max(peak[0], timed.peak_bytes) / 1e9
    else:
        peak_gb = _peak_gb(device, timed)
    return _summarize(
        label,
        device,
        batch_size,
        output_len,
        lengths * iters,
        elapsed,
        firsts,
        finishes,
        peak_gb,
    )


def git_sha() -> str:
    """Short HEAD sha, with ``-dirty`` when tracked files have changes."""
    base = ["git", "--no-optional-locks"]
    try:
        sha = subprocess.run(
            [*base, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [*base, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def append_results(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Append rows to the CSV, writing the header only for a new file.

    Raises:
        ValueError: An existing file has a different header.
    """
    path = Path(path)
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if tuple(header) != CSV_FIELDS:
            raise ValueError(f"{path} has header {header}, expected {list(CSV_FIELDS)}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def check_speedups(results: Sequence[BenchmarkResult]) -> list[str]:
    """Return warnings for stage-vs-stage speedups above ``SUSPICIOUS_SPEEDUP``."""
    warnings: list[str] = []
    # Compare only rows at the same length AND batch size: batching is
    # supposed to beat batch-1 stages by a lot.
    groups: dict[tuple[int, int], list[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault((result.output_len, result.batch_size), []).append(result)
    for (length, _), group in groups.items():
        slowest = min(group, key=lambda r: r.tokens_per_second)
        for result in group:
            ratio = result.tokens_per_second / slowest.tokens_per_second
            if ratio > SUSPICIOUS_SPEEDUP:
                warnings.append(
                    f"{result.stage} is {ratio:.0f}x {slowest.stage} at "
                    f"{length} tokens: check for a missing device sync"
                )
    return warnings


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-vllm-benchmark",
        description="Benchmark generation stages and append rows to results.csv.",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--revision",
        default=None,
        help="hub commit; default is the pin in engine/loader.py",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--stages",
        default="naive,kv",
        help=f"comma list from {sorted(STAGES) + list(BATCHED_STAGES)}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="requests per forward (static, continuous)",
    )
    parser.add_argument("--workload", default="uniform", choices=list(WORKLOADS))
    parser.add_argument(
        "--num-blocks", type=int, default=1024, help="paged KV blocks (continuous)"
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--output-lens", default="64", help="comma list, e.g. 32,64,128,256,512"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="discarded runs per stage and length"
    )
    parser.add_argument(
        "--warmup-len",
        type=int,
        default=64,
        help="tokens per warmup run (capped at output len)",
    )
    parser.add_argument(
        "--iters", type=int, default=1, help="passes over the prompt set"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompts", type=Path, default=Path("prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("results.csv"))
    parser.add_argument(
        "--no-write", action="store_true", help="print only, do not touch the CSV"
    )
    return parser


def _print_table(results: Sequence[BenchmarkResult]) -> None:
    header = f"{'stage':<18}{'batch':>6}{'len':>6}{'tok/s':>10}{'TTFT ms':>10}{'TPOT ms':>10}{'p50 ms':>11}{'p99 ms':>11}{'peak GB':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.stage:<18}{r.batch_size:>6}{r.output_len:>6}{r.tokens_per_second:>10.2f}{r.ttft_ms:>10.1f}"
            f"{r.tpot_ms:>10.1f}{r.p50_ms:>11.0f}{r.p99_ms:>11.0f}{r.peak_mem_gb:>9.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = _build_parser().parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    known = sorted(STAGES) + list(BATCHED_STAGES)
    unknown = [s for s in stages if s not in known]
    if unknown:
        print(f"unknown stage(s) {unknown}; choose from {known}", file=sys.stderr)
        return 2
    output_lens = _parse_ints(args.output_lens)
    prompts = load_prompts(args.prompts)

    model, tokenizer, device = load_model(
        args.model, args.device, args.revision, args.dtype
    )
    prompt_ids = [
        tokenizer(p, return_tensors="pt").input_ids.to(device) for p in prompts
    ]
    sha = git_sha()
    print(
        f"model={args.model} device={device.type} dtype={next(model.parameters()).dtype} "
        f"seed={args.seed} prompts={len(prompts)} iters={args.iters} warmup={args.warmup} git={sha}"
    )
    if device.type == "mps":
        print("NOTE: MPS rows are directional only. README numbers come from the T4.")

    prompt_lists = [ids[0].tolist() for ids in prompt_ids]
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    results: list[BenchmarkResult] = []
    for length in output_lens:
        for stage in stages:
            if stage in STAGES:
                result = run_benchmark(
                    model,
                    prompt_ids,
                    stage,
                    length,
                    device,
                    warmup=args.warmup,
                    warmup_len=args.warmup_len,
                    iters=args.iters,
                    seed=args.seed,
                )
            else:
                result = run_batched_benchmark(
                    model,
                    prompt_lists,
                    stage,
                    length,
                    device,
                    batch_size=args.batch_size,
                    workload=args.workload,
                    warmup=args.warmup,
                    warmup_len=args.warmup_len,
                    iters=args.iters,
                    num_blocks=args.num_blocks,
                    block_size=args.block_size,
                    pad_id=pad_id,
                )
            results.append(result)
            print(
                f"  done {stage} @ {length}: {result.tokens_per_second:.2f} tok/s",
                flush=True,
            )

    print()
    _print_table(results)
    for warning in check_speedups(results):
        print(f"WARNING: {warning}", file=sys.stderr)

    if not args.no_write:
        stamp = datetime.now(UTC).isoformat()
        append_results(args.out, [r.to_row(sha, stamp) for r in results])
        print(f"\nappended {len(results)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCHED_STAGES",
    "CSV_FIELDS",
    "STAGES",
    "BenchmarkResult",
    "TimedModel",
    "append_results",
    "check_speedups",
    "load_prompts",
    "main",
    "percentile",
    "run_batched_benchmark",
    "run_benchmark",
    "workload_lengths",
]
