# mini-vLLM

A small LLM inference engine built from scratch around one idea: decode is
memory-bandwidth bound. Each generated token reads all ~1 GB of
Qwen2.5-0.5B's weights to produce ~150 KB of logits, so every optimization
here is about moving less memory per token.

Model: `Qwen/Qwen2.5-0.5B`, pinned to hub commit `060db64` (see
`engine/loader.py`). One model, one device.

## Status

| stage | what it does | status |
|-------|--------------|--------|
| naive | re-runs the full sequence for every token (baseline) | done |
| kv | prefill once, then decode one token at a time from the KV cache | done, matches naive token for token |
| static batching | several requests per forward pass (left-padded) | reference done + tested, porting to `engine/` |
| paged KV cache | KV stored in 16-token blocks from a shared pool | reference done + tested, porting to `engine/` |
| continuous batching | finished requests leave, waiting ones join, every step; preemption when blocks run out | reference done + tested, porting to `engine/` |

`reference/` holds the working implementation; `PORTING.md` is the plan for
moving it into the hand-written engine files.

Benchmark numbers for this README will come from a Kaggle T4. The M1 (MPS)
numbers in `results.csv` are directional only.

## Layout

- `engine/engine.py`: `generate_naive`, `generate_cached`, `LLMEngine`
- `engine/sampler.py`: greedy, temperature, top-k, top-p
- `engine/cache.py`, `engine/scheduler.py`, `engine/model.py`: block KV cache,
  scheduler and model runner interfaces
- `engine/loader.py`: pinned model id, dtype choice, model and tokenizer loading
- `engine/device.py`: device selection (`auto` = MPS, then CUDA, then CPU) and sync
- `benchmarks/harness.py`: benchmark CLI that appends rows to `results.csv`
- `run.py`: generation CLI
- `reference/`: tested static batching, paged KV cache, paged Qwen2 runner,
  continuous-batching scheduler and `LLMEngine` (see `PORTING.md`)
- `server/api.py`: FastAPI interface (not implemented yet)
- `tests/`: real-model correctness gate plus offline harness tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Generate

```bash
mini-vllm --prompt "Explain paged attention in simple terms."
mini-vllm --prompt "Explain paged attention." --mode naive --max-new-tokens 32
```

## Benchmark

```bash
# M1 length sweep (directional only)
mini-vllm-benchmark --stages naive,kv --output-lens 32,64,128,256,512

# batching: 8 requests per forward pass
mini-vllm-benchmark --stages kv,static,continuous --batch-size 8 --output-lens 128

# mixed output lengths, where continuous batching beats static
mini-vllm-benchmark --stages static,continuous --workload mixed --batch-size 8 --output-lens 256

# Kaggle T4 (README numbers): fp16 is picked automatically, T4 has no bf16.
# benchmarks/kaggle_t4.ipynb runs the gates and every sweep.
mini-vllm-benchmark --device cuda --stages naive,kv --output-lens 32,64,128,256,512 --iters 3
```

For `static` and `continuous`, all requests arrive at once, so TTFT and
p50/p99 latency count from arrival and include queueing. The `mixed` workload
sends 4 x batch-size requests with lengths of 1/8, 1, 1/4 and 1/2 of
`--output-lens`.

Each run syncs the device before every timer stop, discards 3 warmup runs,
uses greedy decoding with seed 2026, ignores EOS so every request emits exactly
`output_len` tokens, and appends one row per stage and length to `results.csv`:
`stage, device, batch_size, output_len, tokens_per_sec, ttft_ms, tpot_ms,
p50_ms, p99_ms, peak_mem_gb, git_sha, timestamp`. A `-dirty` suffix on
`git_sha` means the run had uncommitted changes. Use `--no-write` to print only.

## Tests

```bash
pytest
```

- `tests/test_correctness.py`: on the real model in float32, naive, cached,
  static-batched and paged continuous-batched decoding must all produce
  identical greedy tokens (32 tokens). Skipped, with the reason printed, only
  when the weights can't be loaded.
- `tests/test_batching.py`: block allocator, slot mapping, paged logits vs HF,
  static and continuous batching vs single requests, preemption, EOS, seeding.
- `tests/test_harness.py`: runs offline on a tiny random Qwen2 model:
  naive vs cached equivalence, one forward per token, timing and CSV output,
  and both CLIs end to end.
