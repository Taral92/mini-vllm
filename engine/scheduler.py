"""Continuous-batching scheduler.

Analogy: a restaurant with 8 tables. Every step the waiter first makes sure
each seated guest has room for one more dish, then seats people from the queue
while tables and kitchen space last. A guest who finishes leaves right away and
their table goes to the next person at the very next step.

If the cache runs out of blocks mid-meal, the newest guest is asked to step
out (preempted): their blocks are freed and they rejoin the front of the queue.
When readmitted they recompute their KV from their full text, so their output
is unchanged.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from .cache import PagedKVCache
from .sampler import SamplingParams


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    """Mutable state for one request as it moves through the engine."""

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    params: SamplingParams
    eos_id: int | None
    arrival: int
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_computed: int = 0  # tokens whose K/V are already in the cache
    generator: torch.Generator | None = None  # seeded once per request

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def pending_ids(self) -> list[int]:
        """Tokens to feed this step: whole text on (re)prefill, else the last token."""
        return self.all_ids[self.num_computed :]

    def is_done(self) -> bool:
        if len(self.output_ids) >= self.max_new_tokens:
            return True
        return bool(self.output_ids) and self.output_ids[-1] == self.eos_id


class Scheduler:
    """Pick which requests run in the next step, within batch, token and block limits."""

    def __init__(
        self, cache: PagedKVCache, max_batch_size: int, max_tokens_per_batch: int
    ) -> None:
        self.cache = cache
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        """Queue a request. Rejects requests that could never fit."""
        worst_case = seq.num_tokens + seq.max_new_tokens
        if self.cache.blocks_needed(worst_case) > self.cache.num_blocks:
            raise ValueError(
                f"request {seq.request_id} needs up to {worst_case} tokens, "
                f"cache holds {self.cache.num_blocks * self.cache.block_size}"
            )
        if len(seq.prompt_ids) > self.max_tokens_per_batch:
            raise ValueError(
                f"prompt of {len(seq.prompt_ids)} tokens exceeds "
                f"max_tokens_per_batch={self.max_tokens_per_batch}"
            )
        self.waiting.append(seq)

    def has_pending(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[Sequence]:
        """Return the requests to run this step, with cache blocks reserved."""
        scheduled: list[Sequence] = []
        budget = self.max_tokens_per_batch

        # 1. Running requests first (oldest first): each needs room for the
        #    token it is about to produce. Preempt the newest if blocks run out.
        for seq in sorted(self.running, key=lambda s: s.arrival):
            if seq.status is not SeqStatus.RUNNING:
                continue  # preempted earlier in this loop
            need = self.cache.extra_blocks(seq.block_table, seq.num_tokens)
            while not self.cache.allocator.can_allocate(need):
                victim = max(
                    (s for s in self.running if s.status is SeqStatus.RUNNING),
                    key=lambda s: s.arrival,
                )
                self._preempt(victim)
                if victim is seq:
                    break
            if seq.status is not SeqStatus.RUNNING:
                continue
            self.cache.grow(seq.block_table, seq.num_tokens)
            scheduled.append(seq)
            budget -= len(seq.pending_ids())

        # 2. Admit waiting requests while batch slots, token budget and blocks last.
        while self.waiting and len(scheduled) < self.max_batch_size:
            seq = self.waiting[0]
            cost = len(seq.pending_ids())
            # Over budget waits for the next step, unless the batch is empty
            # (a long preempted request must still be able to run alone).
            if cost > budget and scheduled:
                break
            if not self.cache.allocator.can_allocate(
                self.cache.extra_blocks(seq.block_table, seq.num_tokens)
            ):
                break
            self.waiting.popleft()
            self.cache.grow(seq.block_table, seq.num_tokens)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            scheduled.append(seq)
            budget -= cost

        return scheduled

    def finish(self, seq: Sequence) -> None:
        """Release a finished request's blocks and remove it from the batch."""
        self.cache.release(seq.block_table)
        self.running.remove(seq)
        seq.status = SeqStatus.FINISHED

    def _preempt(self, seq: Sequence) -> None:
        self.cache.release(seq.block_table)
        self.running.remove(seq)
        seq.num_computed = 0  # recompute K/V from prompt + output on readmission
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1
