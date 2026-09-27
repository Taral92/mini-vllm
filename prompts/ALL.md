
## W1-A: engine core (main window)
Implement engine/sampler.py and engine/engine.py.
sampler.py: sample(logits[batch,vocab], temperature, top_k, top_p) -> [batch,1].
temperature==0 is greedy, no division. top_k masks below kth. top_p is nucleus.
Subtract max before softmax.
engine.py, both @torch.no_grad(), return list[int], handle EOS early stop:
- generate_naive(model, ids, max_new, eos_id, **sp): full re-forward each step, deliberately slow, this is the baseline
- generate_cached(model, ids, max_new, eos_id, **sp): past_key_values, prefill prompt once then one token per step
Inline comments explaining exactly what work the cached version skips.

## W1-B: scheduler + batching (main window, after W1-A)
Implement engine/scheduler.py and batched execution in engine/engine.py.
Request dataclass: id, prompt_ids, generated, max_tokens, sampling_params, finished.
Scheduler: waiting deque, running list, max_batch_size, add_request, schedule(),
finish() frees the slot immediately. Continuous batching: freed slot refills on the
NEXT step, not after the batch drains.
Engine: step() -> list[(request_id, token_id, finished)]. One batched forward for all
running requests. Per-request KV cache, left-padding, correct attention mask,
per-request sampling params.
Comment explaining why batching helps decode: memory-bandwidth bound, weights read
once and reused across the batch.

## W2-A: benchmark harness (mv-bench window)
Write benchmarks/harness.py and benchmarks/run.py.
Measure decode tokens/sec, TTFT, TPOT, end-to-end p50/p99, peak memory.
Critical: device sync before EVERY timer stop (cuda.synchronize / mps.synchronize /
no-op on cpu). 3 discarded warmup runs. Fixed seed. Prompts from
benchmarks/prompts.json (10 prompts, 20-500 tokens).
Append to results.csv: stage, device, batch_size, tokens_per_sec, ttft_ms, tpot_ms,
p50_ms, p99_ms, peak_mem_gb, git_sha, timestamp.
Interface: benchmark(engine_fn, prompts, n_runs) -> dict.
run.py flags: --device --stages --max-tokens. Prints a markdown table.

## W2-B: load test (mv-bench window, after W2-A)
Write benchmarks/load_test.py with asyncio + httpx.
N concurrent clients hitting /v1/completions, varied prompt lengths.
Sweep concurrency 1,4,8,16,32,64. Report throughput and p50/p99 per level.
Identify the saturation point. Output a markdown table.

## W3-A: server + docker (mv-server window)
Write server/api.py: FastAPI, OpenAI-compatible POST /v1/completions with SSE
streaming, plus GET /health.
asyncio.Queue for incoming requests. Background task looping engine.step(),
streaming tokens per request id. Graceful shutdown, request-id logging.
Engine interface: add_request(prompt, params) -> id; step() -> list[(id, token, finished)].
Also Dockerfile (multi-stage, CUDA 12.1 runtime base, non-root user, healthcheck on
/health, expose 8000), a CPU-only variant, and docker-compose.yml.

## W4: vLLM study notes (mv-docs window) - run 4 times, swap the file each time
Read vLLM's vllm/core/scheduler.py. Explain as if teaching me:
1) exact sequence of decisions in one scheduler step
2) how a sequence is preempted and what happens to its KV blocks
3) how the block table maps logical positions to physical blocks
4) how waiting/running/swapped queues are managed
Use concrete numbers in examples. Write to docs/notes-vllm-scheduler.md.
[then repeat with: vllm/core/block_manager_v2.py -> docs/notes-vllm-blocks.md,
 vllm/model_executor/layers/sampler.py -> docs/notes-vllm-sampler.md,
 vllm/attention/backends/flash_attn.py -> docs/notes-vllm-attention.md]

## BONUS-1: int8 quantization (any free window, branch feat/quant)
Implement engine/quant.py: int8 weight-only quantization for Linear layers.
Per-channel symmetric, calibration on 128 samples, dequantize-on-the-fly matmul.
Skip lm_head and embeddings. Measure perplexity drift on wikitext-2 before/after.
Comments on why weight-only specifically helps decode.

## BONUS-2: paged KV cache (any free window, branch feat/paged)
Implement engine/paged_cache.py: block-based KV cache.
Fixed block size 16 tokens, global block pool. BlockTable maps request_id ->
list[physical_block_id]. allocate/free/append APIs, fragmentation metrics.
Compare wasted memory vs contiguous preallocation at batch 32.

## FINAL: readme (main window, last)
Write README.md framing this as an engineering cost problem, not a tutorial.
Open with the results table, then the key insight (decode is memory-bandwidth
bound: one token requires reading all 1GB of weights from HBM to produce 150KB of
logits), then KV cache memory math, then architecture, then "what broke" and
"not done yet". No marketing language.
