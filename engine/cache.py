"""Key/value cache interfaces."""

from torch import Tensor, device as TorchDevice, dtype as TorchDType


class KVCache:
    """Manage fixed-size key/value cache blocks for active requests."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: TorchDType,
        device: TorchDevice,
    ) -> None:
        """Configure cache storage.

        Args:
            num_layers: Number of transformer layers.
            num_blocks: Number of allocatable cache blocks.
            block_size: Number of token slots in each block.
            num_kv_heads: Number of key/value attention heads.
            head_size: Width of each attention head.
            dtype: Tensor data type used by cache storage.
            device: Device on which cache tensors will reside.

        Raises:
            NotImplementedError: Cache construction is pending.
        """
        raise NotImplementedError

    def allocate(self, request_id: str, token_count: int) -> tuple[int, ...]:
        """Allocate enough cache blocks for a request.

        Args:
            request_id: Stable identifier for the request.
            token_count: Number of token slots required.

        Returns:
            Allocated block indices.

        Raises:
            NotImplementedError: Block allocation is pending.
        """
        raise NotImplementedError

    def get(self, layer: int, block_indices: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        """Return key and value tensors for selected blocks.

        Args:
            layer: Transformer layer index.
            block_indices: Cache blocks to retrieve.

        Returns:
            Key and value tensors.

        Raises:
            NotImplementedError: Cache retrieval is pending.
        """
        raise NotImplementedError

    def update(
        self,
        layer: int,
        block_indices: tuple[int, ...],
        keys: Tensor,
        values: Tensor,
    ) -> None:
        """Write key and value tensors into selected cache blocks.

        Args:
            layer: Transformer layer index.
            block_indices: Cache blocks to update.
            keys: Key tensor values.
            values: Value tensor values.

        Raises:
            NotImplementedError: Cache updates are pending.
        """
        raise NotImplementedError

    def free(self, request_id: str) -> None:
        """Release all cache blocks owned by a request.

        Args:
            request_id: Stable identifier for the completed request.

        Raises:
            NotImplementedError: Cache release is pending.
        """
        raise NotImplementedError
