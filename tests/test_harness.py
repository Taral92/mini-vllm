"""Offline tests for the benchmark harness, CLI and device helpers.

These use a tiny randomly initialised Qwen2 model (same architecture family:
GQA, RoPE, KV cache) so they run anywhere in seconds without downloading
weights. The real-weights correctness gate stays in test_correctness.py.
"""

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import run
from benchmarks import harness
from benchmarks.harness import (
    CSV_FIELDS,
    IGNORE_EOS,
    BenchmarkResult,
    TimedModel,
    append_results,
    check_speedups,
    load_prompts,
    percentile,
    run_benchmark,
)
from engine.device import resolve_device
from engine.engine import generate_cached, generate_naive
from engine.loader import resolve_dtype
from engine import EngineConfig, LLMEngine

CPU = torch.device("cpu")
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_model() -> Any:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 2 query heads share each KV head
        max_position_embeddings=256,
    )
    model_cls: Any = Qwen2ForCausalLM  # transformers classes are untyped
    model = model_cls(config)
    model.eval()
    return model


@pytest.fixture()
def prompt_ids() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(0, 128, (1, 7), generator=generator),
        torch.randint(0, 128, (1, 11), generator=generator),
    ]


class FakeTokenizer:
    """Just enough tokenizer for the CLI paths: fixed ids in, digits out."""

    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> SimpleNamespace:
        ids = [1 + (ord(ch) % 100) for ch in text[:12]]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in tokens)


def _row(stage: str, length: int, tps: float) -> BenchmarkResult:
    return BenchmarkResult(
        stage, "cpu", 1, length, 3, 1.0, tps, 1.0, 1.0, 1.0, 1.0, 0.1
    )


# --- generation paths on the tiny model -------------------------------------


def test_tiny_model_naive_matches_cached(
    tiny_model: Any, prompt_ids: list[torch.Tensor]
) -> None:
    """Architecture-level gate that runs offline: KV reuse must not change greedy output."""
    for ids in prompt_ids:
        naive = generate_naive(tiny_model, ids, 24, IGNORE_EOS, temperature=0)
        cached = generate_cached(tiny_model, ids, 24, IGNORE_EOS, temperature=0)
        assert naive == cached
        assert len(naive) == 24


@pytest.mark.parametrize("generate", [generate_naive, generate_cached])
def test_one_forward_per_token(
    tiny_model: Any, prompt_ids: list[torch.Tensor], generate: Any
) -> None:
    """TTFT/TPOT math assumes token k is ready after forward k."""
    timed = TimedModel(tiny_model, CPU)
    tokens = generate(timed, prompt_ids[0], 6, IGNORE_EOS, temperature=0)
    assert len(tokens) == 6
    assert len(timed.forward_end) == 6
    assert timed.forward_end == sorted(timed.forward_end)


def test_ignore_eos_forces_full_length(
    tiny_model: Any, prompt_ids: list[torch.Tensor]
) -> None:
    ids = prompt_ids[0]
    first = generate_cached(tiny_model, ids, 1, IGNORE_EOS, temperature=0)[0]
    # With the first greedy token as EOS, a normal run stops after 1 token...
    assert len(generate_cached(tiny_model, ids, 8, first, temperature=0)) == 1
    # ...and the harness still gets all 8 because it passes IGNORE_EOS.
    assert len(generate_cached(tiny_model, ids, 8, IGNORE_EOS, temperature=0)) == 8


# --- run_benchmark -----------------------------------------------------------


@pytest.mark.parametrize("stage", ["naive", "kv"])
def test_run_benchmark_result(
    tiny_model: Any, prompt_ids: list[torch.Tensor], stage: str
) -> None:
    result = run_benchmark(tiny_model, prompt_ids, stage, 8, CPU, warmup=1, iters=2)
    assert result.stage == stage
    assert result.device == "cpu"
    assert result.batch_size == 1
    assert result.output_len == 8
    assert result.request_count == 4  # 2 prompts x 2 iters
    assert result.tokens_per_second == pytest.approx(4 * 8 / result.elapsed_seconds)
    assert result.ttft_ms > 0
    assert result.tpot_ms > 0
    assert result.p99_ms >= result.p50_ms > 0
    assert result.peak_mem_gb > 0


def test_run_benchmark_single_token(
    tiny_model: Any, prompt_ids: list[torch.Tensor]
) -> None:
    result = run_benchmark(tiny_model, prompt_ids, "kv", 1, CPU, warmup=0)
    assert result.tpot_ms == 0.0
    assert result.ttft_ms > 0


def test_run_benchmark_rejects_bad_input(
    tiny_model: Any, prompt_ids: list[torch.Tensor]
) -> None:
    with pytest.raises(KeyError):
        run_benchmark(tiny_model, prompt_ids, "continuous", 4, CPU, warmup=0)
    with pytest.raises(ValueError):
        run_benchmark(tiny_model, prompt_ids, "kv", 0, CPU, warmup=0)
    with pytest.raises(ValueError):
        run_benchmark(tiny_model, [], "kv", 4, CPU, warmup=0)


def test_run_benchmark_catches_short_output(
    prompt_ids: list[torch.Tensor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that stops early must fail loudly, not report inflated tok/s."""

    def stops_early(
        model: Any, ids: torch.Tensor, max_new: int, eos_id: int, **_: Any
    ) -> list[int]:
        model(input_ids=ids)
        return [1]

    monkeypatch.setitem(harness.STAGES, "broken", stops_early)
    with pytest.raises(RuntimeError, match="returned 1 tokens, expected 4"):
        run_benchmark(lambda **_: None, prompt_ids, "broken", 4, CPU, warmup=0)


# --- helpers -----------------------------------------------------------------


def test_percentile_matches_numpy_linear() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 99) == pytest.approx(3.97)
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_load_prompts(tmp_path: Path) -> None:
    assert len(load_prompts(REPO_ROOT / "prompts.json")) >= 1
    bad = tmp_path / "bad.json"
    for content in ('{"a": 1}', "[]", '["ok", ""]', '["ok", 3]'):
        bad.write_text(content)
        with pytest.raises(ValueError):
            load_prompts(bad)


def test_append_results_writes_header_once(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    append_results(out, [_row("naive", 32, 10.0).to_row("abc1234", "t1")])
    append_results(out, [_row("kv", 32, 20.0).to_row("abc1234", "t2")])
    with out.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == CSV_FIELDS
    assert [r[0] for r in rows[1:]] == ["naive", "kv"]


def test_append_results_matches_existing_csv_schema() -> None:
    """New rows must line up with the columns already in results.csv."""
    existing = REPO_ROOT / "results.csv"
    if not existing.exists():
        pytest.skip("no results.csv in this checkout (it is gitignored)")
    with existing.open(newline="") as handle:
        assert tuple(next(csv.reader(handle))) == CSV_FIELDS


def test_append_results_rejects_other_header(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    out.write_text("stage,tok_s\nkv,1\n")
    with pytest.raises(ValueError, match="header"):
        append_results(out, [_row("kv", 32, 20.0).to_row("x", "t")])


def test_check_speedups_flags_missing_sync() -> None:
    assert check_speedups([_row("naive", 64, 6.0), _row("kv", 64, 18.0)]) == []
    warnings = check_speedups([_row("naive", 64, 6.0), _row("kv", 64, 300.0)])
    assert len(warnings) == 1
    assert "kv is 50x naive at 64 tokens" in warnings[0]


def test_resolve_device() -> None:
    assert resolve_device("cpu") == CPU
    assert resolve_device("auto").type in {"mps", "cuda", "cpu"}
    with pytest.raises(ValueError):
        resolve_device("tpu")  # type: ignore[arg-type]
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="not available"):
            resolve_device("cuda")


def test_resolve_dtype() -> None:
    assert resolve_dtype(CPU) == torch.float32
    assert resolve_dtype(torch.device("mps")) == torch.bfloat16
    assert resolve_dtype(CPU, "float16") == torch.float16


# --- CLIs end to end (model loading patched to the tiny model) ---------------


def test_benchmark_main_appends_rows(
    tiny_model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "load_model", lambda *a: (tiny_model, FakeTokenizer(), CPU)
    )
    out = tmp_path / "results.csv"
    code = harness.main(
        [
            "--device",
            "cpu",
            "--stages",
            "naive,kv",
            "--output-lens",
            "4,8",
            "--warmup",
            "1",
            "--prompts",
            str(REPO_ROOT / "prompts.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["stage"], r["output_len"]) for r in rows] == [
        ("naive", "4"),
        ("kv", "4"),
        ("naive", "8"),
        ("kv", "8"),
    ]
    assert all(r["device"] == "cpu" and r["batch_size"] == "1" for r in rows)
    assert all(r["git_sha"] for r in rows)


def test_benchmark_main_rejects_unknown_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        harness, "load_model", lambda *a: pytest.fail("should not load")
    )
    assert harness.main(["--stages", "kv,paged"]) == 2


def test_benchmark_main_no_write(
    tiny_model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "load_model", lambda *a: (tiny_model, FakeTokenizer(), CPU)
    )
    out = tmp_path / "results.csv"
    harness.main(
        [
            "--output-lens",
            "4",
            "--warmup",
            "0",
            "--prompts",
            str(REPO_ROOT / "prompts.json"),
            "--out",
            str(out),
            "--no-write",
        ]
    )
    assert not out.exists()


@pytest.mark.parametrize("mode", ["kv", "naive"])
def test_run_cli_generates(
    tiny_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    import engine.loader

    monkeypatch.setattr(
        engine.loader, "load_model", lambda *a: (tiny_model, FakeTokenizer(), CPU)
    )
    code = run.main(
        ["--prompt", "hello world", "--max-new-tokens", "5", "--mode", mode]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
    assert f"mode={mode}" in captured.err


def test_run_cli_rejects_zero_tokens() -> None:
    assert run.main(["--prompt", "x", "--max-new-tokens", "0"]) == 2


# --- batched stages ----------------------------------------------------------


def test_workload_lengths() -> None:
    assert harness.workload_lengths(64, 3, "uniform") == [64, 64, 64]
    assert harness.workload_lengths(64, 5, "mixed") == [8, 64, 16, 32, 8]
    assert harness.workload_lengths(4, 1, "mixed") == [1]  # never below 1
    with pytest.raises(ValueError):
        harness.workload_lengths(64, 2, "bursty")


@pytest.mark.parametrize("stage", ["static", "continuous"])
@pytest.mark.parametrize("workload", ["uniform", "mixed"])
def test_run_batched_benchmark(
    tiny_model: Any, prompt_ids: list[torch.Tensor], stage: str, workload: str
) -> None:
    prompts = [ids[0].tolist() for ids in prompt_ids]
    result = harness.run_batched_benchmark(
        tiny_model,
        prompts,
        stage,
        8,
        CPU,
        batch_size=2,
        workload=workload,
        warmup=1,
        iters=2,
        num_blocks=64,
    )
    requests = (2 if workload == "uniform" else 8) * 2
    useful = sum(harness.workload_lengths(8, requests // 2, workload)) * 2
    assert result.stage == (stage if workload == "uniform" else f"{stage}-mixed")
    assert result.batch_size == 2
    assert result.request_count == requests
    assert result.tokens_per_second == pytest.approx(useful / result.elapsed_seconds)
    assert 0 < result.ttft_ms <= result.p50_ms <= result.p99_ms


def test_continuous_finishes_short_requests_early(
    tiny_model: Any, prompt_ids: list[torch.Tensor]
) -> None:
    """Mixed workload: continuous batching returns short requests sooner than static."""
    prompts = [ids[0].tolist() for ids in prompt_ids]
    lengths = [2, 40, 2, 40]
    static = harness._serve_static(
        TimedModel(tiny_model, CPU), prompts * 2, lengths, 2, CPU, pad_id=0
    )
    engine = LLMEngine(tiny_model, config=EngineConfig(max_batch_size=2, num_blocks=64))
    continuous = harness._serve_continuous(engine, prompts * 2, lengths, CPU, [0])
    # Request 2 (2 tokens) waits for request 1's 40 tokens under static batching,
    # but takes request 0's seat after 2 steps under continuous batching.
    assert continuous[2][2] < static[2][2]


def test_run_batched_benchmark_rejects_bad_input(tiny_model: Any) -> None:
    with pytest.raises(ValueError, match="batched stage"):
        harness.run_batched_benchmark(tiny_model, [[1, 2]], "kv", 4, CPU)
    with pytest.raises(ValueError):
        harness.run_batched_benchmark(
            tiny_model, [[1, 2]], "static", 4, CPU, batch_size=0
        )
    with pytest.raises(ValueError):
        harness.run_batched_benchmark(tiny_model, [], "static", 4, CPU)


def test_check_speedups_ignores_batching_gains() -> None:
    batched = BenchmarkResult(
        "continuous", "cpu", 8, 64, 8, 1.0, 600.0, 1.0, 1.0, 1.0, 1.0, 0.1
    )
    assert check_speedups([_row("naive", 64, 6.0), batched]) == []


def test_benchmark_main_batched_stages(
    tiny_model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "load_model", lambda *a: (tiny_model, FakeTokenizer(), CPU)
    )
    out = tmp_path / "results.csv"
    code = harness.main(
        [
            "--stages",
            "kv,static,continuous",
            "--batch-size",
            "3",
            "--output-lens",
            "6",
            "--warmup",
            "1",
            "--num-blocks",
            "64",
            "--prompts",
            str(REPO_ROOT / "prompts.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["stage"], r["batch_size"]) for r in rows] == [
        ("kv", "1"),
        ("static", "3"),
        ("continuous", "3"),
    ]
