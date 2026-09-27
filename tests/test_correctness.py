"""Correctness tests for generation and token sampling."""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.engine import generate_cached, generate_naive
from engine.sampler import sample


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
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
    """Load the shared correctness model and tokenizer once per test run."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
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
        cached_token = (
            cached[divergence] if divergence < len(cached) else "<missing>"
        )
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
