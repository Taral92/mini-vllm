"""Correctness tests for generation and token sampling."""

from typing import Any

import pytest
import torch

from engine.engine import generate_cached, generate_naive
from engine.loader import MODEL_NAME, load_model
from engine.sampler import sample

PROMPT = "Continue this exact counting sequence: 1, 2, 3, 4, 5, 6, 7, 8,"
SEED = 2026


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Choose the fastest available torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model_and_tokenizer(device: torch.device) -> tuple[Any, Any]:
    """Load the pinned model and tokenizer once per test run, in float32.

    float32 because token-for-token equality is only a fair demand when both
    paths round the same way. In bf16 (3 significant digits) naive and cached
    decoding can flip a near-tied greedy choice (seen at token ~33 on MPS).

    Skips (with the reason printed, see pytest addopts) when the weights are
    neither cached nor downloadable, e.g. in an offline sandbox.
    """
    try:
        model, tokenizer, _ = load_model(
            MODEL_NAME,
            device.type,  # type: ignore[arg-type]
            dtype="float32",
        )
    except OSError as error:
        pytest.skip(f"{MODEL_NAME} unavailable: {error}")
    return model, tokenizer


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def test_naive_and_cached_generation_match(
    model_and_tokenizer: tuple[Any, Any],
    device: torch.device,
) -> None:
    """Cached decoding must preserve the baseline's greedy token sequence."""
    model, tokenizer = model_and_tokenizer
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None

    _seed(SEED)
    naive = generate_naive(
        model,
        ids,
        max_new=32,
        eos_id=eos_id,
        temperature=0,
    )
    _seed(SEED)
    cached = generate_cached(
        model,
        ids,
        max_new=32,
        eos_id=eos_id,
        temperature=0,
    )

    divergence = _first_divergence(naive, cached)
    if divergence is not None:
        naive_token = naive[divergence] if divergence < len(naive) else "<missing>"
        cached_token = cached[divergence] if divergence < len(cached) else "<missing>"
        pytest.fail(
            f"generation diverged first at index {divergence}: "
            f"naive={naive_token}, cached={cached_token}"
        )
    assert len(naive) == len(cached) == 32


def test_sampling_is_deterministic_for_same_seed(device: torch.device) -> None:
    _seed(SEED)
    logits = torch.randn(16, 128, device=device)

    _seed(SEED)
    first = sample(logits, temperature=0.8, top_k=32, top_p=0.9)
    _seed(SEED)
    second = sample(logits, temperature=0.8, top_k=32, top_p=0.9)

    assert torch.equal(first, second)


def test_top_k_one_equals_greedy(device: torch.device) -> None:
    _seed(SEED)
    logits = torch.randn(16, 128, device=device)

    greedy = sample(logits, temperature=0)
    top_k_one = sample(logits, temperature=1, top_k=1, top_p=1)

    assert torch.equal(top_k_one, greedy)


def test_sample_output_shape_and_dtype(device: torch.device) -> None:
    batch_size = 7
    logits = torch.randn(batch_size, 128, device=device)

    output = sample(logits, temperature=1, top_k=32, top_p=0.9)

    assert output.shape == (batch_size, 1)
    assert output.dtype == torch.long


# --- batched paths on the real model -------------------------------------------

BATCH_PROMPTS = (
    PROMPT,
    "Explain paged attention in simple terms.",
    "Write a short function that reverses a string.",
    "Summarize the benefits of continuous batching.",
)
NO_EOS = -1


@pytest.fixture(scope="session")
def single_outputs(
    model_and_tokenizer: tuple[Any, Any], device: torch.device
) -> tuple[list[list[int]], list[list[int]]]:
    """(prompt ids, single-request cached outputs) for the batch prompts."""
    model, tokenizer = model_and_tokenizer
    prompt_ids = [tokenizer(p).input_ids for p in BATCH_PROMPTS]
    outputs = [
        generate_cached(
            model, torch.tensor([ids], device=device), 32, NO_EOS, temperature=0
        )
        for ids in prompt_ids
    ]
    return prompt_ids, outputs


def _assert_rows_match(actual: list[list[int]], expected: list[list[int]]) -> None:
    for row, (got, want) in enumerate(zip(actual, expected, strict=True)):
        index = _first_divergence(got, want)
        if index is not None:
            pytest.fail(f"prompt {row} diverged first at token {index}")


def test_static_batch_matches_single(
    model_and_tokenizer: tuple[Any, Any],
    single_outputs: tuple[list[list[int]], list[list[int]]],
) -> None:
    """Left-padded static batching must not change any prompt's greedy tokens."""
    from engine import generate_static

    model, tokenizer = model_and_tokenizer
    prompt_ids, expected = single_outputs
    pad = tokenizer.pad_token_id or 0
    actual = generate_static(model, prompt_ids, 32, NO_EOS, pad_id=pad, temperature=0)
    _assert_rows_match(actual, expected)


def test_paged_engine_matches_single(
    model_and_tokenizer: tuple[Any, Any],
    single_outputs: tuple[list[list[int]], list[list[int]]],
) -> None:
    """Continuous batching over the paged KV cache must match single requests."""
    from engine import EngineConfig, LLMEngine

    model, tokenizer = model_and_tokenizer
    prompt_ids, expected = single_outputs
    engine = LLMEngine(model, tokenizer, EngineConfig(max_batch_size=3, num_blocks=64))
    actual = engine.generate(prompt_ids, max_new_tokens=32, ignore_eos=True)
    _assert_rows_match(actual, expected)
    assert engine.cache.allocator.num_free == 64
