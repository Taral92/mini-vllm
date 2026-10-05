# mini-vLLM

A small LLM inference engine: KV cache, static batching, continuous batching
and a paged KV cache, behind an OpenAI-style streaming server. One model
(`Qwen/Qwen2.5-0.5B`, pinned to hub commit `060db64`), one GPU, every
optimization checked token-for-token against the Hugging Face baseline and
measured on a Tesla T4.

## Results (Tesla T4, fp16, greedy, 3 timed runs after 3 warmups)

**Batching: same per-token latency, 34x the throughput.** 256 output tokens
per request, all requests arriving at once.

| requests in flight | static tok/s | continuous tok/s | static TPOT ms |
|---:|---:|---:|---:|
| 1 | 31.8 | 34.0 | 31.5 |
| 4 | 139.7 | 137.6 | 28.6 |
| 8 | 265.3 | 258.2 | 30.2 |
| 16 | 540.8 | 512.6 | 29.6 |
| 32 | 1,080.8 | 971.6 | 29.6 |

**Mixed output lengths: continuous batching beats static.** 32 requests
arriving at once, 8 in flight, output lengths cycling 32 / 256 / 64 / 128
tokens.

| | static | continuous | |
|---|---:|---:|---|
| throughput, tok/s | 126.8 | 212.9 | 1.7x |
| time to first token, ms | 11,367 | 4,640 | 2.4x faster |
| p50 request latency, ms | 18,892 | 8,147 | 2.3x faster |

**KV cache vs full re-forward (batch 1).**

| output tokens | naive tok/s | kv tok/s | kv / naive |
|---:|---:|---:|---:|
| 32 | 28.0 | 30.4 | 1.08x |
| 128 | 29.8 | 31.1 | 1.04x |
| 512 | 24.9 | 30.9 | 1.24x |

Raw rows are in `results.csv` (`device=cuda`, run at commit `7de9a3c`);
`benchmarks/kaggle_t4.ipynb` reproduces them. These T4 rows were transcribed
from the notebook's printed table (2 decimals) because the Kaggle session
ended before the CSV was downloaded. M1 rows (`device=mps`) are directional only.

## What the numbers say

**1. Per-step cost is fixed, so batching is almost free.** TPOT stays at
~30 ms from 1 to 32 requests. Each decode step pays the same cost whether it
produces 1 token or 32, so throughput scales with batch size until the GPU
runs out of compute. At batch 32 it has not.

**2. At 0.5B parameters, decode is overhead-bound, not bandwidth-bound.**
Reading Qwen2.5-0.5B's ~1 GB of fp16 weights from the T4's 320 GB/s memory
takes ~3 ms. Measured TPOT is ~32 ms, 10x that floor. The rest is per-step
overhead: hundreds of small kernel launches and Python dispatch per token,
each one tiny for a model this size. That is also why the KV cache only buys
1.04-1.24x here: re-running a few hundred tokens is cheap on a T4, and the
step overhead dominates either way. For a 7B model the weight read alone is
~45 ms per step and the classic memory-bound picture returns. The next lever
for this engine is CUDA graphs (or `torch.compile`) to collapse the launch
overhead, not more memory tricks.

**3. Continuous batching wins when request lengths differ.** A static batch
ends when its longest request ends, so short requests leave seats idle. In
the mixed workload above, each static batch of 8 runs 256 steps (2,048
seat-steps) to produce 960 tokens: 53% of seats sit empty. Continuous
batching refills a seat on the very next step. With equal lengths there is
nothing to refill, and continuous costs 3-10% more than static (gathering K/V
from scattered cache blocks grows with batch size).

## KV cache memory

```
bytes per token = 2 (K and V) x 24 layers x 2 KV heads x 64 head_dim x 2 bytes = 12 KB
2,048-token request   = 25 MB
64 such requests      = 1.6 GB   (more than the 1 GB of weights)
```

That is why the cache is paged: 16-token blocks from a shared pool, allocated
as a request grows instead of reserving its maximum length up front. A request
that ends at 50 tokens holds 4 blocks (64 slots), not 2,048.

## How a request flows

```
POST /v1/completions
  -> tokenize, check it fits                         server/api.py
  -> inbox -> worker admits it between steps         server/api.py (EngineWorker)
  -> scheduler gives it a seat and cache blocks      engine/scheduler.py
  -> step(): one forward for every running request   engine.py
       prefill (whole prompt) or decode (1 token)
       new K/V written into its blocks, attention
       reads its blocks back                          model.py, cache.py
  -> sampler picks a token per request (own RNG)     engine/sampler.py
  -> token streamed back as SSE; finished requests
     free their blocks, the next waiting one joins   scheduler.py
```

If the cache runs out mid-generation, the newest request is preempted: its
blocks are freed and it re-runs prefill on its full text when readmitted, so
its output does not change.

## Correctness

Every path must produce the same greedy tokens as Hugging Face's own forward,
in float32, on the real model: naive, cached, static-batched and paged
continuous-batched (`tests/test_correctness.py`). On failure the test reports
the first divergent token index. Offline tests on a tiny random Qwen2 cover
the block allocator, slot mapping, preemption, EOS, per-request seeding, the
harness and the server (`tests/test_batching.py`, `tests/test_harness.py`,
`tests/test_server.py`).

## What broke along the way

- **Fake peak memory.** The first harness reported 0.988 GB for every stage:
  the size of the weights, read after the run instead of a true peak. Fixed;
  peaks now differ by stage as they should.
- **naive and kv disagreed in bf16.** Same prompt, same greedy decoding, and
  the outputs split at token 7 ("If" vs "When"). Both paths are correct; they
  sum in a different order and bf16 rounding flipped a near-tie. The
  correctness gate runs in float32 for this reason.
- **bf16 on a T4.** `torch.cuda.is_bf16_supported()` returns true on a T4
  because it counts software emulation, which would have made every T4 number
  slow. The loader now checks compute capability (bf16 needs 8.0+).
- **Batch-dependent output in bf16.** Streaming one long greedy request while
  a second joined mid-generation changed the long one's text at token 33:
  the batch went from 1 row to 2, the rounding order changed, and a near-tied
  token flipped. In float32 the solo and batched outputs are identical
  (`diff` clean), so batching itself is correct; bf16 batch invariance would
  need batch-invariant kernels.
- **Continuous batching 3.3x slower than static on the M1.** On the T4 the
  gap is 3-10%. The remaining cost is gathering K/V from scattered blocks.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                         # all correctness and offline tests
mini-vllm --prompt "The capital of France is"  # one prompt, KV cache path

mini-vllm-serve --port 8000                    # OpenAI-style server
curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
curl -N localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "Once upon a time", "max_tokens": 64, "stream": true}'
```

Qwen2.5-0.5B is a base model: give it text to continue, not a question.

Docker (CPU by default; pass a CUDA wheel index for GPU):

```bash
docker build -t mini-vllm .
docker run -p 8000:8000 mini-vllm
```

Benchmarks:

```bash
mini-vllm-benchmark --stages naive,kv --output-lens 32,128,512
mini-vllm-benchmark --stages static,continuous --batch-size 8 --output-lens 256
mini-vllm-benchmark --stages static,continuous --workload mixed --batch-size 8 --output-lens 256
```

Each run syncs the device before every timer stop, discards 3 warmups, uses
greedy decoding with seed 2026, ignores EOS so every request emits exactly
`output_len` tokens, and appends one row per stage to `results.csv`. For
batched stages all requests arrive at once, so TTFT and p50/p99 include
queueing.

## Layout

- `engine/engine.py`: `generate_naive`, `generate_cached`, `generate_static`
  and the continuous-batching `LLMEngine` (`add_request`, `step`, `generate`)
- `engine/scheduler.py`: waiting/running queues, admission, preemption
- `engine/cache.py`: block allocator and paged K/V slot pools
- `engine/model.py`: Qwen2 forward over the paged cache (HF weights, own attention)
- `engine/sampler.py`: greedy, temperature, top-k, top-p
- `engine/loader.py`, `engine/device.py`: pinned model loading, dtype, device sync
- `server/api.py`: `/v1/completions` (JSON or SSE streaming) and `/health`
- `benchmarks/`: harness CLI and the Kaggle T4 notebook
- `tests/`: real-model gates and offline tests

## Not done

- CUDA graphs / `torch.compile` for the per-step overhead found above.
- Speculative decoding; int8 weight-only quantization.
- A client that disconnects mid-stream is not cancelled; its request runs to
  completion.
- The GPU Docker image is untested.
