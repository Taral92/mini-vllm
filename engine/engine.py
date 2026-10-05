"""Generation paths, from slowest to fastest.

- generate_naive:  re-runs the whole sequence for every token (baseline)
- generate_cached: prefill once, then one token per step from the HF KV cache
- generate_static: B prompts per forward pass, left-padded (static batching)
- LLMEngine:       continuous batching over the paged KV cache

LLMEngine.step():
    1. scheduler picks up to max_batch_size requests and reserves cache blocks
    2. model runs one forward for all of them (prefill and decode mixed)
    3. each request samples one token with its own seeded generator
    4. finished requests free their blocks; their seats go to waiting requests
"""

from collections.abc import Sequence as Seq
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .model import BatchInput, PagedModelRunner, build_cache
from .sampler import SamplingParams, _sample, sample
from .scheduler import Scheduler, SeqStatus, Sequence


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


# Static batching with the Hugging Face KV cache: B prompts, one forward per step.
#
# Analogy: a tour bus. Everyone boards together and the bus only returns when
# the last passenger is done, so seats of people who finished early ride empty.
# That wasted seat is exactly what continuous batching fixes.
#
# Prompts have different lengths, so they are LEFT-padded to line up their last
# tokens, with an attention mask hiding the padding and position ids that start
# at 0 for each prompt's first real token.
@torch.no_grad()
def generate_static(
    model: Any,
    prompts: list[list[int]],
    max_new: int,
    eos_id: int,
    pad_id: int = 0,
    **sampling: Any,
) -> list[list[int]]:
    """Generate for all prompts in one batch; return each row's tokens up to EOS."""
    if max_new <= 0:
        return [[] for _ in prompts]
    device = next(model.parameters()).device
    lengths = [len(p) for p in prompts]
    width = max(lengths)

    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), width, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt) :] = torch.tensor(prompt)
        mask[row, width - len(prompt) :] = 1
    input_ids, mask = input_ids.to(device), mask.to(device)
    # Position 0 = first real token of each row, whatever its padding.
    positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)

    outputs = model(
        input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=True
    )
    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)

    for step in range(max_new):
        next_tokens: Tensor = sample(outputs.logits[:, -1, :], **sampling)  # [B, 1]
        for row, token in enumerate(next_tokens.squeeze(-1).tolist()):
            if not finished[row]:
                generated[row].append(int(token))
                finished[row] = token == eos_id
        if all(finished) or step == max_new - 1:
            break
        # Finished rows keep decoding (their tokens are dropped): the cost of static batching.
        mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=-1)
        positions = positions[:, -1:] + 1
        outputs = model(
            input_ids=next_tokens,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
    return generated


# Continuous batching over the paged KV cache.

GREEDY = SamplingParams(temperature=0.0)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Batch, token and cache limits."""

    max_batch_size: int = 8
    max_tokens_per_batch: int = 2048
    block_size: int = 16
    num_blocks: int = 512  # 8,192 token slots; ~100 MB for Qwen2.5-0.5B in bf16


class LLMEngine:
    """Continuous-batching engine over a Hugging Face causal LM."""

    def __init__(
        self, model: Any, tokenizer: Any = None, config: EngineConfig | None = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        config = config or EngineConfig()
        self.config = config
        self.cache = build_cache(model, config.num_blocks, config.block_size)
        self.runner = PagedModelRunner(model, self.cache)
        self.scheduler = Scheduler(
            self.cache, config.max_batch_size, config.max_tokens_per_batch
        )
        self.sequences: dict[str, Sequence] = {}
        self._arrivals = 0

    def add_request(
        self,
        request_id: str,
        prompt: str | Seq[int],
        params: SamplingParams = GREEDY,
        max_new_tokens: int = 64,
        ignore_eos: bool = False,
    ) -> None:
        """Tokenize (if needed) and queue a request."""
        if request_id in self.sequences:
            raise ValueError(f"duplicate request id {request_id!r}")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("a tokenizer is needed for text prompts")
            prompt_ids = list(self.tokenizer(prompt).input_ids)
        else:
            prompt_ids = list(prompt)
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")

        eos = (
            None
            if ignore_eos or self.tokenizer is None
            else self.tokenizer.eos_token_id
        )
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.runner.device)
            generator.manual_seed(params.seed)
        seq = Sequence(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            params=params,
            eos_id=eos,
            arrival=self._arrivals,
            generator=generator,
        )
        self._arrivals += 1
        self.scheduler.add(seq)
        self.sequences[request_id] = seq

    def has_unfinished(self) -> bool:
        return self.scheduler.has_pending()

    def step(self) -> dict[str, int]:
        """Run one model step. Returns the new token id for each request that ran."""
        batch = self.scheduler.schedule()
        if not batch:
            return {}
        logits = self.runner.forward(
            BatchInput(
                token_ids=[seq.pending_ids() for seq in batch],
                start_positions=[seq.num_computed for seq in batch],
                block_tables=[seq.block_table for seq in batch],
            )
        )
        tokens = self._sample(logits, batch)

        emitted: dict[str, int] = {}
        for seq, token in zip(batch, tokens, strict=True):
            seq.num_computed = seq.num_tokens  # everything fed so far is cached
            seq.output_ids.append(token)
            emitted[seq.request_id] = token
            if seq.is_done():
                self.scheduler.finish(seq)
        return emitted

    def _sample(self, logits: torch.Tensor, batch: list[Sequence]) -> list[int]:
        if all(seq.params.temperature == 0 for seq in batch):
            return [
                int(t) for t in logits.argmax(dim=-1).tolist()
            ]  # one sync for the batch
        tokens: list[int] = []
        for row, seq in enumerate(batch):
            p = seq.params
            token = _sample(
                logits[row : row + 1], p.temperature, p.top_k, p.top_p, seq.generator
            )
            tokens.append(int(token.item()))
        return tokens

    def generate(
        self,
        prompts: Seq[str | Seq[int]],
        params: SamplingParams = GREEDY,
        max_new_tokens: int = 64,
        ignore_eos: bool = False,
    ) -> list[list[int]]:
        """Run every prompt to completion; return output token ids in prompt order."""
        ids = [f"gen-{self._arrivals + i}" for i in range(len(prompts))]
        for request_id, prompt in zip(ids, prompts, strict=True):
            self.add_request(request_id, prompt, params, max_new_tokens, ignore_eos)
        while any(self.sequences[r].status is not SeqStatus.FINISHED for r in ids):
            self.step()
        return [list(self.sequences.pop(r).output_ids) for r in ids]
