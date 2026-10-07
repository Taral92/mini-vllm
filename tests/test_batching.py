"""Offline correctness tests for static batching, paged KV and continuous batching.

All run on a tiny random Qwen2 in float32 on CPU. Every batched path must
produce exactly the tokens that single-request cached decoding produces.
The same checks on the real model live in test_correctness.py.
"""

from typing import Any

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from engine.engine import generate_cached
from engine.sampler import SamplingParams
from engine import (
    BatchInput,
    BlockAllocator,
    EngineConfig,
    LLMEngine,
    OutOfBlocksError,
    PagedKVCache,
    PagedModelRunner,
    Scheduler,
    SeqStatus,
    Sequence,
    build_cache,
    generate_static,
)

NO_EOS = -1
MAX_NEW = 24


@pytest.fixture(scope="module")
def model() -> Any:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model_cls: Any = Qwen2ForCausalLM
    return model_cls(config).eval()


@pytest.fixture(scope="module")
def prompts() -> list[list[int]]:
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(1, 128, (n,), generator=generator).tolist()
        for n in (7, 11, 3, 20)
    ]


@pytest.fixture(scope="module")
def expected(model: Any, prompts: list[list[int]]) -> list[list[int]]:
    """Single-request cached decoding: the baseline every batched path must match."""
    return [
        generate_cached(model, torch.tensor([p]), MAX_NEW, NO_EOS, temperature=0)
        for p in prompts
    ]


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _assert_same(actual: list[list[int]], expected: list[list[int]]) -> None:
    for row, (got, want) in enumerate(zip(actual, expected, strict=True)):
        index = _first_divergence(got, want)
        assert index is None, f"row {row} diverged first at token {index}"


# --- block allocator and cache ------------------------------------------------


def test_allocator_hands_out_and_takes_back() -> None:
    allocator = BlockAllocator(4)
    first = allocator.allocate(3)
    assert len(set(first)) == 3 and allocator.num_free == 1
    with pytest.raises(OutOfBlocksError):
        allocator.allocate(2)
    assert allocator.num_free == 1  # a failed allocation takes nothing
    allocator.free(first)
    assert allocator.num_free == 4
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free([first[0]])  # double free


def test_cache_slot_mapping() -> None:
    cache = PagedKVCache(1, 8, 4, 2, 8, torch.float32, torch.device("cpu"))
    table = [5, 2]  # blocks need not be contiguous or ordered
    # position 0..3 -> block 5 (slots 20..23), 4..5 -> block 2 (slots 8, 9)
    assert cache.slots(table, 0, 6) == [20, 21, 22, 23, 8, 9]
    assert cache.blocks_needed(0) == 0
    assert cache.blocks_needed(4) == 1
    assert cache.blocks_needed(5) == 2
    grown: list[int] = []
    cache.grow(grown, 9)
    assert len(grown) == 3
    cache.release(grown)
    assert grown == [] and cache.allocator.num_free == 8


def test_bytes_per_token_matches_formula() -> None:
    # Qwen2.5-0.5B: 2 (K,V) x 24 layers x 2 KV heads x 64 dims x 2 bytes = 12,288
    cache = PagedKVCache(24, 1, 16, 2, 64, torch.bfloat16, torch.device("cpu"))
    assert cache.bytes_per_token() == 12_288


# --- paged model runner --------------------------------------------------------


def test_paged_prefill_logits_match_hf(model: Any, prompts: list[list[int]]) -> None:
    cache = build_cache(model, num_blocks=64, block_size=16)
    runner = PagedModelRunner(model, cache)
    tables: list[list[int]] = []
    for prompt in prompts:
        table: list[int] = []
        cache.grow(table, len(prompt))
        tables.append(table)
    logits = runner.forward(BatchInput(prompts, [0] * len(prompts), tables))
    for row, prompt in enumerate(prompts):
        reference = model(input_ids=torch.tensor([prompt])).logits[0, -1]
        torch.testing.assert_close(logits[row], reference, atol=1e-5, rtol=1e-5)


def test_paged_decode_uses_absolute_positions(
    model: Any, prompts: list[list[int]]
) -> None:
    """Prefill then one decode step must equal a full forward over prompt + token."""
    prompt = prompts[1]
    cache = build_cache(model, num_blocks=8, block_size=4)
    runner = PagedModelRunner(model, cache)
    table: list[int] = []
    cache.grow(table, len(prompt))
    first = int(runner.forward(BatchInput([prompt], [0], [table])).argmax())
    cache.grow(table, len(prompt) + 1)
    logits = runner.forward(BatchInput([[first]], [len(prompt)], [table]))[0]
    reference = model(input_ids=torch.tensor([prompt + [first]])).logits[0, -1]
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)


# --- static batching -----------------------------------------------------------


def test_static_batch_matches_single(
    model: Any, prompts: list[list[int]], expected: list[list[int]]
) -> None:
    _assert_same(
        generate_static(model, prompts, MAX_NEW, NO_EOS, temperature=0), expected
    )


def test_static_batch_stops_rows_at_eos(
    model: Any, prompts: list[list[int]], expected: list[list[int]]
) -> None:
    eos = expected[0][2]  # third greedy token of row 0
    out = generate_static(model, prompts, MAX_NEW, eos, temperature=0)
    assert out[0] == expected[0][:3]  # row 0 stops at its EOS
    for row in range(
        1, len(prompts)
    ):  # other rows run on, cut at their first EOS if any
        want = expected[row]
        cut = want.index(eos) + 1 if eos in want else len(want)
        assert out[row] == want[:cut]


# --- continuous batching engine -------------------------------------------------


def test_engine_matches_single(
    model: Any, prompts: list[list[int]], expected: list[list[int]]
) -> None:
    engine = LLMEngine(model, config=EngineConfig(max_batch_size=3, num_blocks=64))
    _assert_same(
        engine.generate(prompts, max_new_tokens=MAX_NEW, ignore_eos=True), expected
    )
    assert engine.cache.allocator.num_free == 64  # every block returned


def test_engine_abort_frees_seat_and_blocks(
    model: Any, prompts: list[list[int]], expected: list[list[int]]
) -> None:
    """Aborting a running and a waiting request frees everything; others finish."""
    engine = LLMEngine(model, config=EngineConfig(max_batch_size=2, num_blocks=64))
    for i, prompt in enumerate(prompts[:3]):
        engine.add_request(f"r{i}", prompt, max_new_tokens=MAX_NEW, ignore_eos=True)
    engine.step()  # r0, r1 running; r2 waiting
    engine.abort("r0")  # running
    engine.abort("r2")  # waiting
    engine.abort("missing")  # unknown ids are ignored
    assert [s.request_id for s in engine.scheduler.running] == ["r1"]
    assert not engine.scheduler.waiting
    while engine.has_unfinished():
        engine.step()
    assert engine.sequences["r1"].output_ids == expected[1]
    assert engine.cache.allocator.num_free == 64


def test_engine_preemption_keeps_output(
    model: Any, prompts: list[list[int]], expected: list[list[int]]
) -> None:
    """A cache too small for all four requests forces preemption and recompute."""
    engine = LLMEngine(
        model, config=EngineConfig(max_batch_size=4, num_blocks=12, block_size=4)
    )
    _assert_same(
        engine.generate(prompts, max_new_tokens=MAX_NEW, ignore_eos=True), expected
    )
    assert engine.scheduler.num_preemptions > 0
    assert engine.cache.allocator.num_free == 12


def test_finished_request_frees_its_seat_next_step(
    model: Any, prompts: list[list[int]], expected: list[list[int]]
) -> None:
    """The continuous-batching property: r2 starts as soon as r0 finishes, before r1 does."""
    engine = LLMEngine(model, config=EngineConfig(max_batch_size=2, num_blocks=64))
    lengths = (3, 24, 5)
    for index, length in enumerate(lengths):
        engine.add_request(
            f"r{index}", prompts[index], max_new_tokens=length, ignore_eos=True
        )
    steps: list[set[str]] = []
    while engine.has_unfinished():
        steps.append(set(engine.step()))
    assert steps[:3] == [{"r0", "r1"}] * 3
    assert steps[3] == {"r1", "r2"}  # r0 done after 3 tokens; r2 takes its seat at once
    for index, length in enumerate(lengths):
        assert engine.sequences[f"r{index}"].output_ids == expected[index][:length]


def test_engine_stops_at_eos(
    model: Any, prompts: list[list[int]], expected: list[list[int]]
) -> None:
    tokenizer = type("Tok", (), {"eos_token_id": expected[0][4]})()
    engine = LLMEngine(model, tokenizer, EngineConfig(num_blocks=64))
    engine.add_request("a", prompts[0], max_new_tokens=MAX_NEW)
    while engine.has_unfinished():
        engine.step()
    assert engine.sequences["a"].output_ids == expected[0][:5]
    assert engine.sequences["a"].status is SeqStatus.FINISHED


def test_engine_seeded_sampling_is_reproducible(
    model: Any, prompts: list[list[int]]
) -> None:
    params = SamplingParams(temperature=0.9, top_p=0.95, seed=7)
    runs = []
    for _ in range(2):
        engine = LLMEngine(model, config=EngineConfig(num_blocks=64))
        runs.append(
            engine.generate(prompts[:2], params, max_new_tokens=16, ignore_eos=True)
        )
    assert runs[0] == runs[1]


def test_engine_seed_draws_advance_each_step(
    model: Any, prompts: list[list[int]]
) -> None:
    """One generator per request, seeded once: draws differ step to step.

    With uniform logits every token is a pure random draw, so a generator
    re-seeded every step (the Sampler.sample bug) would repeat one token.
    """
    engine = LLMEngine(model, config=EngineConfig(num_blocks=64))
    engine.add_request(
        "s", prompts[0], SamplingParams(temperature=1.0, seed=3), 12, ignore_eos=True
    )
    seq = engine.sequences["s"]
    engine.scheduler.schedule()
    uniform = torch.zeros(1, 128)
    tokens = {engine._sample(uniform, [seq])[0] for _ in range(12)}
    assert len(tokens) > 1


def test_engine_rejects_bad_requests(model: Any) -> None:
    engine = LLMEngine(
        model, config=EngineConfig(num_blocks=2, block_size=4, max_tokens_per_batch=16)
    )
    with pytest.raises(ValueError, match="cache holds"):
        engine.add_request(
            "big", list(range(1, 6)), max_new_tokens=10
        )  # 15 tokens > 8 slots
    with pytest.raises(ValueError, match="text prompts"):
        engine.add_request("text", "hello")
    with pytest.raises(ValueError, match="at least 1"):
        engine.add_request("zero", [1, 2], max_new_tokens=0)
    engine.add_request("ok", [1, 2], max_new_tokens=2)
    with pytest.raises(ValueError, match="duplicate"):
        engine.add_request("ok", [1, 2], max_new_tokens=2)


def test_scheduler_respects_batch_and_token_limits(model: Any) -> None:
    cache = build_cache(model, num_blocks=64, block_size=16)
    scheduler = Scheduler(cache, max_batch_size=2, max_tokens_per_batch=10)
    greedy = SamplingParams(temperature=0)
    for index, length in enumerate((4, 5, 3)):
        scheduler.add(
            Sequence(f"q{index}", list(range(1, length + 1)), 4, greedy, None, index)
        )
    batch = scheduler.schedule()
    assert [s.request_id for s in batch] == [
        "q0",
        "q1",
    ]  # 4 + 5 tokens <= 10, batch full
    assert [s.request_id for s in scheduler.waiting] == ["q2"]


# --- GPU-sync guard ------------------------------------------------------------

# Ops that make the GPU report a value or a size back to Python. Each one
# stalls the device queue; inside the per-layer loop they cost ~2 ms each on
# MPS. A boolean-mask index (x[mask]) is one of them: it has to count the Trues.
_SYNC_OPS = {"nonzero", "_local_scalar_dense", "masked_select", "item"}


def test_paged_forward_has_no_device_syncs(
    model: Any, prompts: list[list[int]]
) -> None:
    """Regression guard: 48 hidden syncs per step made continuous 3x slower than static on MPS."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class SyncRecorder(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.hits: list[str] = []

        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            name = func.__name__.split(".")[0]
            if name in _SYNC_OPS:
                self.hits.append(name)
            if (
                name.startswith("index")
                and len(args) > 1
                and isinstance(args[1], (list, tuple))
            ):
                if any(
                    isinstance(t, torch.Tensor) and t.dtype == torch.bool
                    for t in args[1]
                ):
                    self.hits.append(f"{name}[bool mask]")
            return func(*args, **(kwargs or {}))

    cache = build_cache(model, num_blocks=64, block_size=16)
    runner = PagedModelRunner(model, cache)
    tables: list[list[int]] = []
    for prompt in prompts:
        table: list[int] = []
        cache.grow(table, len(prompt) + 1)
        tables.append(table)
    runner.forward(BatchInput(prompts, [0] * len(prompts), tables))  # prefill
    decode = BatchInput([[5]] * len(prompts), [len(p) for p in prompts], tables)
    with SyncRecorder() as recorder:
        runner.forward(decode)
    assert recorder.hits == []
