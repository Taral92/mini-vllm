# mini-vllm — project handoff

## Me
Senior Python engineer, Surat India. Targeting Senior/Staff ML Engineer roles
(Google, Meta, Anthropic). Background: Python, PyTorch, RAG, Flask, Docker, cloud.

Preferences: point-wise bullets not prose. Reasoning before implementation.
No padding. Single recommendation, not option menus. Plain-language analogy
before technical detail. Tell me proactively when I'm overengineering.
Every bullet needs a concrete worked example.

## Hardware
MacBook Air M1, 8GB, MPS backend. No paid compute.
Kaggle free T4 (30 GPU-hrs/week) for real benchmarks.
Build locally, benchmark on Kaggle — M1 unified memory (~68 GB/s) doesn't show
the HBM-bandwidth effects this project is about, so MPS numbers are directional
only, never for the README.

## Project
A from-scratch LLM inference engine — a small vLLM. Not training a model, not an
agent, not an API wrapper. Model: Qwen/Qwen2.5-0.5B (24 layers, 14 query heads,
2 KV heads, head_dim 64, hidden 896, vocab 151936).

Purpose: my portfolio is agent/RAG-heavy. This proves ML systems skill.
Target roles: ML Systems Engineer, Inference Engineer, ML Infrastructure.

Core insight the project hangs on: decode is memory-bandwidth bound, not compute
bound. Producing one token requires reading all ~1GB of weights from HBM to emit
~150KB of logits. Every optimization targets memory movement, not FLOPs. That's
why batching is the biggest lever — weights are read once and reused across the batch.

The deliverable is the benchmark table, not the code.

## Repo
~/Downloads/mini-vllm (git, branch main)
Worktrees: ../mv-bench (feat/benchmarks), ../mv-server (feat/server),
../mv-docs (feat/docs), ../mv-attn (feat/attention), ../mv-quant (feat/quant)
All feature branches currently sit at 70a81a4 (same as main) — no work committed on them yet.
All build prompts live in prompts/ALL.md
Note: run.py and prompts.json are at repo ROOT, not under benchmarks/
results.csv is at repo root and is gitignored (`*.csv` in .gitignore).

## Done
- engine/sampler.py — temperature, top_k, top_p, greedy at temp=0
- engine/engine.py — generate_naive (full re-forward baseline) and
  generate_cached (HF past_key_values)
- Correctness gate PASSED on real Qwen2.5-0.5B, MPS, seed 2026, temp 0,
  32 tokens: naive and cached produce identical token lists
- 4 tests in tests/test_correctness.py on main (1 naive-vs-cached gate,
  3 sampler tests). The stash adds 3 more (custom-model layer check,
  custom decode logits check, block-cache unit test) = the original "7 tests".
- First benchmark (MPS, 64 tokens, seed 2026, 3 discarded warmups,
  mps.synchronize verified before every timer stop):

  | stage | tok/s | TTFT ms | TPOT ms | p99 ms | peak GB |
  |-------|-------|---------|---------|--------|---------|
  | naive | 6.58  | 84.2    | 152.0   | 9914   | 0.988   |
  | kv    | 18.53 | 64.6    | 54.0    | 3499   | 0.988   |

  2.8x. Low because 64 tokens is short — naive is O(n^2), cached is O(n),
  so the gap widens with length.
- Partial MPS length sweep in results.csv (directional only):

  | output_len | naive tok/s | kv tok/s | speedup |
  |------------|-------------|----------|---------|
  | 32         | 10.76       | 20.57    | 1.9x    |
  | 64         | 6.58        | 18.53    | 2.8x    |
  | 128        | 7.65        | 21.13    | 2.8x    |
  | 256        | 5.13        | 18.78    | 3.7x    |

  512 still missing. Gap widens with length as predicted.

## Parked
- Custom block-aware attention + paged KV cache: ~900 lines (engine/model.py,
  engine/cache.py, engine/engine.py, engine/__init__.py, tests) were written on
  main, then `git stash`ed because that work belongs on feat/attention.
  Recoverable: stash@{0} "WIP on main: 70a81a4 prompts".
- Deferred to phase 2 deliberately: you cannot debug a custom cache without a
  known-good baseline to diff against.

## Not done
- engine/quant.py — does NOT exist on any branch or in the stash. Only the
  BONUS-1 prompt in prompts/ALL.md. May be uncommitted in ../mv-quant; unverified.
- benchmarks/harness.py is still a STUB — benchmarks so far came from a
  throwaway script that is not in the repo. Real harness is the blocker.
- run.py, server/api.py, engine/device.py (resolve_device), and the
  LLMEngine/ModelRunner/KVCache/Scheduler classes on main are stubs.
- README.md is stale (still says everything is a stub).
- Static batching
- Continuous batching  <- highest interview value, completely untouched
- Paged KV cache (parked in stash)
- Speculative decoding
- FastAPI server + Docker
- Kaggle T4 benchmark run (all README numbers must come from here)
- Length sweep: 512 on MPS, and the whole sweep on T4

## Rules
- AI may write: harness, server, Docker, load test, tests, docs, README
- I write by hand: engine/engine.py, engine/cache.py, engine/scheduler.py,
  engine/sampler.py. These four are the interview. If I can't explain the KV
  cache indexing and the scheduler step loop, the project is worthless.
- Not production scale. Production-quality code at research scope: one model,
  one GPU, clean, tested, benchmarked.
- No CUDA kernels. PyTorch/Triton is enough.

## Benchmark discipline
- Device sync before EVERY timer stop (cuda.synchronize / mps.synchronize / no-op cpu)
- 3 discarded warmup iterations minimum
- Fixed seed, recorded in output
- Append to results.csv: stage, device, batch_size, output_len, tokens_per_sec,
  ttft_ms, tpot_ms, p50_ms, p99_ms, peak_mem_gb, git_sha, timestamp
- Label MPS runs clearly — directional only, never README numbers
- If a speedup looks >20x, suspect a missing sync before believing it

## Correctness gate
- Any new inference path (cache, batching, quantization, paged attention) must
  produce IDENTICAL greedy output to the HF baseline: temperature=0, fixed seed,
  32 tokens, real Qwen2.5-0.5B, not a mock.
- On failure, report the first divergent token index.
- Never mark an optimization done without this test passing.

## Bug traps already identified
- RoPE position during decode must be current seq length, not 0.
  Symptom: garbage after token 1.
- GQA: 2 KV heads must be repeated to serve 14 query heads
- Causal mask applied BEFORE softmax, not after
- Missing device sync before timer stop produces fake 50x speedups
- Correctness gate: greedy output from any new path must match the HF baseline
  token for token

## KV cache memory formula
bytes = 2 × L × S × H_kv × d_head × dtype_bytes
Qwen-0.5B at 2048 tokens: 2×24×2048×2×64×2 ≈ 25MB/sequence.
At batch 64 → 1.6GB, larger than the 1GB of weights. That's why paging matters.

## Notes for Claude (Cowork)
- The Cowork shell is a Linux VM, not the Mac. The repo's .venv is a macOS venv
  and torch/MPS are not available there — pytest and benchmarks must be run by
  me in my own terminal. Git, reading, and editing work from Cowork.
- From the Cowork VM, the worktrees show as "prunable" (their /Users/... paths
  aren't mounted). NEVER run `git worktree prune` from Cowork.
- Use `git --no-optional-locks` for read-only git commands from Cowork: plain
  `git status` can leave a stale .git/index.lock (the VM can't unlink it by
  default), which then blocks git on the Mac.
- Update the Done / Not done / Next up sections of this file at the end of any
  session where a stage completes or a benchmark runs. Never claim a stage is
  done without a benchmark row and a passing correctness test.

## Next up
1. Real benchmark harness (unblocks everything)
2. Length sweep: add 512 (MPS), then the full sweep on T4
3. Continuous batching
4. Kaggle T4 run for README numbers
