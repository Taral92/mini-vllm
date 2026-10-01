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
GitHub: https://github.com/Taral92/mini-vllm (public). Name stays mini-vllm.
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
- MPS length sweep in results.csv (directional only):

  | output_len | naive tok/s | kv tok/s | speedup |
  |------------|-------------|----------|---------|
  | 32         | 10.76       | 20.57    | 1.9x    |
  | 64         | 6.58        | 18.53    | 2.8x    |
  | 128        | 7.65        | 21.13    | 2.8x    |
  | 256        | 5.13        | 18.78    | 3.7x    |
  | 512        | 2.00        | 21.84    | 10.9x   |

  Sweep complete on MPS. Gap widens with length as predicted (naive O(n^2)).
- Real benchmark harness (benchmarks/harness.py, 2026-09-29, AI-written):
  `mini-vllm-benchmark --stages naive,kv --output-lens 32,64,128,256,512`.
  Syncs before every timer stop, 3 discarded warmups (capped at 64 tokens,
  `--warmup-len`), greedy, seed 2026, ignores EOS so every request emits
  exactly output_len tokens, appends the 12-column rows to results.csv,
  `-dirty` suffix on git_sha for uncommitted runs, warns on >20x speedups.
  TTFT/TPOT come from a TimedModel wrapper that records a synced timestamp
  after every forward — no changes to engine.py needed. New stages plug
  into `STAGES` in harness.py.
  Note: MPS peak_mem now sampled after every forward (includes KV +
  activations), so it will read slightly above the old 0.988 GB.
- engine/device.py: resolve_device (auto = MPS > CUDA > CPU) + synchronize()
- engine/loader.py: MODEL_NAME, MODEL_REVISION pin
  (060db6499f32faf8b98477b0a26969ef7d8b9987), dtype choice (bf16 on MPS,
  fp16 on T4 since sm_75 has no bf16, fp32 on CPU), load_model()
- run.py CLI: `mini-vllm --prompt "..." [--mode kv|naive]`
- transformers pinned to 5.17.0 (torch >=2.4, tested 2.14.0)
- tests/test_harness.py: 23 offline tests on a tiny random Qwen2 (GQA 4q/2kv):
  naive==cached, one forward per token, IGNORE_EOS, run_benchmark metrics,
  CSV header/schema, percentile, >20x warning, both CLIs end to end.
  Full suite: 27 tests (26 pass in the cloud; the real-model gate skips there
  with the reason printed because it has no weights).
- README.md rewritten (status table, how to run); benchmark table still
  waits for T4 numbers.
- 2026-09-29 (session 2): I chose "reference + tests": Claude wrote a
  complete, tested implementation in reference/ (NOT in my four files) for
  me to port by hand. PORTING.md = order, tests per step, what to explain.
  - reference/block_cache.py: BlockAllocator + PagedKVCache (flat slot pools
    [layers, blocks*16, 2, 64], slot = table[p//16]*16 + p%16)
  - reference/paged_model.py: Qwen2 forward over HF weights, paged attention
    (write new K/V to slots, gather context, GQA repeat, SDPA causal+len mask),
    mixed prefill+decode in one step, lm_head only on last tokens
  - reference/scheduler.py: mutable Sequence, 2-phase schedule (running first,
    preempt newest + recompute when out of blocks; then admit waiting)
  - reference/llm_engine.py: LLMEngine add_request/step/generate, one seeded
    generator per request (fixes the Sampler.sample bug), batched greedy argmax
  - reference/static_batch.py: left-padded static batching over HF cache
  - tests/test_batching.py: 15 offline tests (all batched paths == single
    cached decoding token for token, preemption, EOS, seeding, limits)
  - tests/test_correctness.py: real-model fixture now float32 (bf16 flips
    near-tied greedy tokens: naive vs kv text diverged at ~token 33 on MPS
    bf16); added real-model gates for static batching and paged engine
  - harness: stages static, continuous; --batch-size, --workload
    uniform|mixed (mixed = 4x batch requests, lengths x 1/8,1,1/4,1/2,
    stage label "<stage>-mixed"), --num-blocks, --block-size; TTFT/latency
    for batched rows count from arrival (queueing included); gc +
    empty_cache between runs; >20x warning only within same batch size
  - CPU smoke (random 8-layer Qwen2, batch 8, len 64 — shape only, not
    numbers): kv 149 tok/s, static 588, continuous 618; mixed workload:
    static 283 vs continuous 405 tok/s, p99 3.4 s vs 2.3 s
  - benchmarks/kaggle_t4.ipynb: gates + length sweep + batch-size sweep
    (1,4,8,16,32) + mixed workload, prints the README table
  - Full suite in cloud: 50 passed, 3 skipped (real-model gates need weights)
- 2026-09-29 MPS batching run #1 (bf16, batch 8, len 128): kv 27.3, static
  157.7, continuous 47.8 tok/s (TPOT 166 ms vs static 50). Mixed len 256:
  static 70.8 vs continuous 36.7. Cause: paged_model used boolean-mask
  indexing (k[q_valid]) = 2 GPU->CPU syncs/layer x 24 = 48 stalls per step;
  invisible on CPU. Fixed with CPU-built index + index_select; new test
  test_paged_forward_has_no_device_syncs (TorchDispatchMode) fails if any
  sync op / bool-mask index appears in forward. Re-run pending.
  Interview story: "no data-dependent shapes on the GPU in the hot loop".

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
- server/api.py and the LLMEngine/ModelRunner/KVCache/Scheduler classes on
  main are stubs. Server waits for LLMEngine.step().
- Port reference/ into my four files + engine/model.py (PORTING.md order:
  sampler fix, static batching, cache, model runner, scheduler, step loop)
- Speculative decoding
- FastAPI server + Docker
- Kaggle T4 benchmark run + full length sweep (all README numbers come from here)

## Open issues in the hand-written files (found 2026-09-29, not fixed — mine to fix)
- sampler.py: `Sampler.sample` builds and seeds a new torch.Generator on
  every call, so every decode step reuses the same random draw. Seed once per
  request and keep the generator on the request. (Module-level `sample()`
  used by generate_* is unaffected.) Fixed in reference/llm_engine.py.
- scheduler.py: `Request` is frozen with prompt ids only. Fixed in
  reference/scheduler.py (mutable Sequence).
- model.py: forward() takes `Mapping[int, (K, V)]` dense per-layer tensors.
  Replaced in reference/paged_model.py by BatchInput(block_tables, positions).
- server/api.py: max_new_tokens has ge=1 but no le=; no prompt-length cap;
  temperature<0 passes pydantic and becomes a 500. Fix when writing the server.

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
- Claude's cloud workspace can pip-install torch/transformers (CPU) and run
  tests/test_harness.py, but huggingface.co is blocked by egress policy there
  and in the Cowork VM, so the real-model gate only runs on the Mac / Kaggle.
- Don't commit from Cowork (index.lock problem) — I commit from the Mac.
- From the Cowork VM, the worktrees show as "prunable" (their /Users/... paths
  aren't mounted). NEVER run `git worktree prune` from Cowork.
- Use `git --no-optional-locks` for read-only git commands from Cowork: plain
  `git status` can leave a stale .git/index.lock (the VM can't unlink it by
  default), which then blocks git on the Mac.
- Update the Done / Not done / Next up sections of this file at the end of any
  session where a stage completes or a benchmark runs. Never claim a stage is
  done without a benchmark row and a passing correctness test.

## Next up
1. Me: `pip install -e ".[dev]" && pytest` on the Mac — all 53 must pass
   (the 3 real-model gates in float32 included), then commit + push
2. Me: MPS batching check: `mini-vllm-benchmark --stages kv,static,continuous
   --batch-size 8 --output-lens 128 --no-write` and the mixed workload
3. Kaggle: run benchmarks/kaggle_t4.ipynb → README benchmark table
4. Me: port reference/ into engine/ following PORTING.md; keep tests green
5. Open: kv TTFT jumped 48 -> ~300 ms at len 128/256 right after naive runs
   (MPS). gc + empty_cache now run between stages; re-check.
6. Later: server + Docker (AI), speculative decoding, int8 (optional)
7. stash@{0} is superseded by reference/ — drop it once the port is done
