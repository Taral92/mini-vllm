# Porting the reference engine into your four files

`reference/` is a complete, tested implementation of static batching, paged KV
cache and continuous batching. It exists so you can **port it by hand** into
`engine/engine.py`, `engine/cache.py`, `engine/scheduler.py`,
`engine/sampler.py` (plus `engine/model.py`) and be able to explain every line.
Don't copy files over. Retype each piece, then run its tests.

Tests that already pass against `reference/`:

- `tests/test_batching.py`: 15 offline tests on a tiny Qwen2
- `tests/test_correctness.py`: real-model float32 gates (Mac or Kaggle)

When you port a piece, point its test imports at `engine.*` instead of
`reference.*`. The test must stay green.

## Order

### 1. Sampler seeding fix → `engine/sampler.py`

- **Bug:** `Sampler.sample` creates and seeds a new `torch.Generator` on every
  call, so every decode step replays the same random draw.
- **Reference:** `LLMEngine.add_request` creates one generator per request;
  `LLMEngine._sample` reuses it every step.
- **Test:** `test_engine_seed_draws_advance_each_step`.
- **Explain:** with uniform logits every token is a pure random draw. Re-seeding
  every step returns the same token 12 times in a row.

### 2. Static batching → `engine/engine.py` (`generate_static`)

- **Reference:** `reference/static_batch.py` (~70 lines).
- **Key idea:** left-pad so every row's last token lines up; `attention_mask`
  hides the pads; `position_ids = cumsum(mask) - 1` so each prompt starts at
  position 0.
- **Tests:** `test_static_batch_matches_single`,
  `test_static_batch_stops_rows_at_eos`, real-model
  `test_static_batch_matches_single`.
- **Explain:** prompt A has 7 tokens and prompt B has 11. Without the position
  fix, A's first real token would get RoPE position 4 instead of 0, and its
  output changes.

### 3. Paged KV cache → `engine/cache.py`

- **Reference:** `reference/block_cache.py` (`BlockAllocator`, `PagedKVCache`).
- **Key line:** `slot = block_table[p // block_size] * block_size + p % block_size`.
- **Tests:** `test_allocator_hands_out_and_takes_back`,
  `test_cache_slot_mapping`, `test_bytes_per_token_matches_formula`.
- **Explain:** with `block_table = [5, 2]` and `block_size = 4`, position 5
  lives in slot `2*4 + 1 = 9`. Blocks don't need to be contiguous.
- **Your stub's interface changes:** `allocate(request_id, n)` becomes
  "grow a block table to fit n tokens". The per-layer tensors are flat slot
  pools shaped `[layers, num_blocks*block_size, kv_heads, head_dim]`.

### 4. Paged model runner → `engine/model.py`

- **Reference:** `reference/paged_model.py`. It keeps HF's weights, norms,
  RoPE and MLP, and replaces only attention.
- **Per layer:**
  1. project q, k, v
  2. apply RoPE at absolute positions
  3. **write** new K/V into their slots
  4. **read** each request's whole context by gathering its slots
  5. repeat the 2 KV heads to 14 query heads
  6. run SDPA with a causal + length mask
- **Tests:** `test_paged_prefill_logits_match_hf`,
  `test_paged_decode_uses_absolute_positions`.
- **This replaces the dense `Mapping[int, (K, V)]` interface,** which would
  have copied the whole cache every token.

### 5. Scheduler → `engine/scheduler.py`

- **Reference:** `reference/scheduler.py`.
- **`Request` becomes a mutable `Sequence`,** tracking status, output ids,
  block table, `num_computed` and its generator.
- **`schedule()` has two phases:**
  1. Running requests get room for one more token. If blocks run out, the
     newest request is preempted: its blocks are freed and it goes to the
     front of the queue.
  2. Waiting requests are admitted while batch slots, token budget and blocks
     last.
- **Tests:** `test_scheduler_respects_batch_and_token_limits`,
  `test_engine_preemption_keeps_output`.
- **Explain:** a preempted request loses its cache but not its tokens. On
  re-admission `pending_ids()` returns prompt + output, so it re-prefills
  everything and continues with identical output.

### 6. Continuous batching step loop → `engine/engine.py` (`LLMEngine`)

- **Reference:** `reference/llm_engine.py`.
- **`step()`:**
  1. schedule
  2. one forward for all scheduled requests (prefill and decode mixed)
  3. sample per request
  4. `num_computed = num_tokens`
  5. finish and free blocks
- **Tests:** `test_engine_matches_single`,
  `test_finished_request_frees_its_seat_next_step`, `test_engine_stops_at_eos`,
  real-model `test_paged_engine_matches_single`.
- **Explain:** with batch size 2, r0 needs 3 tokens and r1 needs 24. At step 4,
  r2 takes r0's seat while r1 is still running. Static batching would leave
  that seat empty for 21 steps.

## Interview questions you should be able to answer after porting

- **Why left padding and not right padding for batched decode?**
  - The next token is read from position `-1` of every row.
- **What's in the cache after `step()` for a decode request?**
  - Every token except the one just sampled.
  - That's why `pending_ids()` is exactly 1 token on the next step.
- **How much KV does one token cost for Qwen2.5-0.5B, and how many requests
  fit in 1,024 blocks?**
  - 12 KB per token.
  - 16,384 slots ≈ 200 MB.
  - That's 32 requests of 512 tokens.
- **Why recompute on preemption instead of swapping to CPU?**
  - Simpler, and prefill is compute-bound and fast.
  - Swapping costs PCIe bandwidth.
- **Where does continuous batching beat static?**
  - Mixed output lengths. Compare `static-mixed` and `continuous-mixed` in the
    harness.
