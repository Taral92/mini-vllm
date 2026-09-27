"""Model loading and execution interfaces."""

from collections.abc import Mapping
from typing import Any

from torch import Tensor, device as TorchDevice


class ModelRunner:
    """Load and execute a causal language model on one device."""

    def __init__(self, model_name: str, device: TorchDevice) -> None:
        """Configure a model runner.

        Args:
            model_name: Hugging Face model identifier or local model path.
            device: Device on which model execution will occur.

        Raises:
            NotImplementedError: Model-runner construction is pending.
        """
        raise NotImplementedError

    def load(self) -> None:
        """Load model weights and prepare the model for inference.

        Raises:
            NotImplementedError: Model loading is pending.
        """
        raise NotImplementedError

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_cache: Mapping[int, tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Run a model forward pass.

        Args:
            input_ids: Token IDs for the current batch.
            positions: Absolute positions corresponding to each token.
            kv_cache: Layer-indexed key/value tensors.

        Returns:
            Next-token logits for the current batch.

        Raises:
            NotImplementedError: Model execution is pending.
        """
        raise NotImplementedError

    @property
    def tokenizer(self) -> Any:
        """Return the tokenizer associated with the loaded model.

        Raises:
            NotImplementedError: Tokenizer access is pending.
        """
        raise NotImplementedError
