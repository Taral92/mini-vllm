"""Token sampling interfaces."""

from dataclasses import dataclass

import torch
from torch import Tensor


def _sample(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one vocabulary entry")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    # Zero temperature means exact greedy decoding; importantly, do not divide
    # by zero (or apply sampling filters) on this path.
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    scores = logits / temperature

    if top_k is not None:
        k = min(top_k, scores.shape[-1])
        kth_values = torch.topk(scores, k, dim=-1).values[..., -1, None]
        # A strict comparison preserves ties at the kth-largest value.
        scores = scores.masked_fill(scores < kth_values, -torch.inf)

    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(
            scores, dim=-1, descending=True
        )
        stable_sorted_scores = sorted_scores - sorted_scores.amax(
            dim=-1, keepdim=True
        )
        sorted_probs = torch.softmax(stable_sorted_scores, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Keep the first token that takes the cumulative mass over top_p, then
        # remove every lower-probability token after it.
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False

        remove = torch.zeros_like(sorted_remove)
        remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        scores = scores.masked_fill(remove, -torch.inf)

    # Centering avoids overflow in exp for very large logits.
    stable_scores = scores - scores.amax(dim=-1, keepdim=True)
    probabilities = torch.softmax(stable_scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def sample(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> Tensor:
    """Sample one token per row from logits shaped ``[batch, vocab]``."""
    return _sample(logits, temperature, top_k, top_p)


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters controlling next-token sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None

