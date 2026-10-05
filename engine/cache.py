"""Paged KV cache: one preallocated pool of fixed-size blocks shared by all requests.

Analogy: a car park with numbered bays. A request gets bays one at a time as
its text grows and hands them all back when it finishes. Nobody reserves 2,048
bays "just in case".

Layout: for each layer, keys and values live in flat tensors of
``num_blocks * block_size`` slots. Token position ``p`` of a request lives in

    slot = block_table[p // block_size] * block_size + p % block_size

so a request's tokens can sit in any blocks, in any order.
"""

from collections import deque

import torch
from torch import Tensor
from torch import device as TorchDevice
from torch import dtype as TorchDType


class OutOfBlocksError(RuntimeError):
    """Raised when an allocation asks for more free blocks than exist."""


class BlockAllocator:
    """Hand out and take back block ids from a fixed pool."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def can_allocate(self, count: int) -> bool:
        return count <= len(self._free)

    def allocate(self, count: int) -> list[int]:
        """Take ``count`` free blocks, or raise without taking any."""
        if count > len(self._free):
            raise OutOfBlocksError(f"need {count} blocks, {len(self._free)} free")
        blocks = [self._free.popleft() for _ in range(count)]
        self._used.update(blocks)
        return blocks

    def free(self, blocks: list[int]) -> None:
        """Return blocks to the pool. Freeing a block twice is a bug."""
        for block in blocks:
            if block not in self._used:
                raise ValueError(f"block {block} is not allocated")
            self._used.remove(block)
            self._free.append(block)


class PagedKVCache:
    """Per-layer key/value slot pools plus the allocator that owns them."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: TorchDType,
        device: TorchDevice,
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.allocator = BlockAllocator(num_blocks)
        shape = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def num_blocks(self) -> int:
        return self.allocator.num_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        """Blocks required to hold ``num_tokens`` tokens (ceil division)."""
        return -(-num_tokens // self.block_size)

    def extra_blocks(self, block_table: list[int], num_tokens: int) -> int:
        """How many more blocks ``block_table`` needs to hold ``num_tokens``."""
        return max(0, self.blocks_needed(num_tokens) - len(block_table))

    def grow(self, block_table: list[int], num_tokens: int) -> None:
        """Append blocks to ``block_table`` in place so it fits ``num_tokens``."""
        block_table.extend(
            self.allocator.allocate(self.extra_blocks(block_table, num_tokens))
        )

    def release(self, block_table: list[int]) -> None:
        """Free every block in ``block_table`` and empty it."""
        self.allocator.free(block_table)
        block_table.clear()

    def slots(self, block_table: list[int], start: int, end: int) -> list[int]:
        """Flat slot index for each token position in ``[start, end)``."""
        size = self.block_size
        return [block_table[p // size] * size + p % size for p in range(start, end)]

    def bytes_per_token(self) -> int:
        """K and V bytes for one token across all layers."""
        _, _, heads, dim = self.keys.shape
        return 2 * self.num_layers * heads * dim * self.keys.element_size()

    def layer(self, index: int) -> tuple[Tensor, Tensor]:
        """Key and value slot pools for one layer, shaped [slots, kv_heads, head_dim]."""
        return self.keys[index], self.values[index]
