"""Reference implementations to port into the hand-written engine files.

- block_cache.py  -> engine/cache.py       (block allocator + paged KV pool)
- scheduler.py    -> engine/scheduler.py   (mutable Sequence, admission, preemption)
- llm_engine.py   -> engine/engine.py      (LLMEngine.step loop, per-request RNG)
- static_batch.py -> engine/engine.py      (left-padded static batching)
- paged_model.py  -> engine/model.py       (Qwen2 forward over paged KV)

See PORTING.md at the repo root for the order and what to be able to explain.
"""

from .block_cache import BlockAllocator, OutOfBlocksError, PagedKVCache
from .llm_engine import EngineConfig, LLMEngine
from .paged_model import BatchInput, PagedModelRunner, build_cache
from .scheduler import Scheduler, SeqStatus, Sequence
from .static_batch import generate_static

__all__ = [
    "BatchInput",
    "BlockAllocator",
    "EngineConfig",
    "LLMEngine",
    "OutOfBlocksError",
    "PagedKVCache",
    "PagedModelRunner",
    "Scheduler",
    "SeqStatus",
    "Sequence",
    "build_cache",
    "generate_static",
]
