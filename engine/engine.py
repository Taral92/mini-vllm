"""Top-level inference engine interfaces."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .device import DeviceName
from .sampler import SamplingParams, sample


@torch.no_grad()
def generate_naive(
    model: Any,
    ids: Tensor,
    max_new: int,
    eos_id: int,
    **sampling: Any,
) -> list[int]:
    """Generate tokens by forwarding the complete sequence on every step."""
    generated: list[int] = []
    input_ids = ids

    for _ in range(max_new):
        outputs = model(input_ids=input_ids, use_cache=False)
        next_token = sample(outputs.logits[:, -1, :], **sampling)
        token_id = int(next_token.item())
        generated.append(token_id)

        if token_id == eos_id:
            break
        input_ids = torch.cat((input_ids, next_token), dim=-1)

    return generated


@torch.no_grad()
def generate_cached(
    model: Any,
    ids: Tensor,
    max_new: int,
    eos_id: int,
    **sampling: Any,
) -> list[int]:
    """Generate tokens using the model's key/value cache."""
    if max_new <= 0:
        return []

    # Prefill computes and caches keys and values for every prompt token once.
    outputs = model(input_ids=ids, use_cache=True)
    past_key_values = outputs.past_key_values
    generated: list[int] = []

    for step in range(max_new):
        next_token = sample(outputs.logits[:, -1, :], **sampling)
        token_id = int(next_token.item())
        generated.append(token_id)

        if token_id == eos_id or step == max_new - 1:
            break

        # Feeding only the new token skips recomputing projections and
        # attention states for the prompt and all earlier generated tokens;
        # their keys and values are reused directly from past_key_values.
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    return generated


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Configuration required to construct an inference engine."""

    model_name: str
    device: DeviceName = "auto"
    max_batch_size: int = 8
    max_tokens_per_batch: int = 2048
    cache_block_size: int = 16
    num_cache_blocks: int = 256


class LLMEngine:
    """Coordinate model execution, scheduling, caching, and sampling."""

    def __init__(self, config: EngineConfig) -> None:
        """Configure the inference engine.

        Args:
            config: Model, device, scheduler, and cache configuration.

        Raises:
            NotImplementedError: Engine construction is pending.
        """
        raise NotImplementedError

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        max_new_tokens: int,
    ) -> None:
        """Tokenize and enqueue a generation request.

        Args:
            request_id: Stable identifier for the request.
            prompt: Input text to continue.
            sampling_params: Controls for next-token selection.
            max_new_tokens: Maximum number of tokens to generate.

        Raises:
            NotImplementedError: Request submission is pending.
        """
        raise NotImplementedError

    def step(self) -> dict[str, str]:
        """Execute one scheduled prefill or decode step.

        Returns:
            Newly decoded text grouped by request identifier.

        Raises:
            NotImplementedError: Engine stepping is pending.
        """
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        max_new_tokens: int,
    ) -> str:
        """Generate a complete response for one prompt.

        Args:
            prompt: Input text to continue.
            sampling_params: Controls for next-token selection.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Generated text.

        Raises:
            NotImplementedError: Synchronous generation is pending.
        """
        raise NotImplementedError

    def stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Yield generated text fragments for one prompt.

        Args:
            prompt: Input text to continue.
            sampling_params: Controls for next-token selection.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            An iterator of generated text fragments.

        Raises:
            NotImplementedError: Streaming generation is pending.
        """
        raise NotImplementedError
