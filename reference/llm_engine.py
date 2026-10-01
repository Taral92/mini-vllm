"""LLMEngine: the step loop that ties scheduler, paged cache, model and sampler together.

One call to ``step()``:
    1. scheduler picks up to max_batch_size requests and reserves cache blocks
    2. model runs one forward for all of them (prefill and decode mixed)
    3. each request samples one token with its own seeded generator
    4. finished requests free their blocks; their seats go to waiting requests
"""

from collections.abc import Sequence as Seq
from dataclasses import dataclass
from typing import Any

import torch

from engine.sampler import SamplingParams, _sample

from .paged_model import BatchInput, PagedModelRunner, build_cache
from .scheduler import Scheduler, SeqStatus, Sequence

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
