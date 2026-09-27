"""Device selection interfaces for the inference engine."""

from typing import Literal, TypeAlias

from torch import device as TorchDevice

DeviceName: TypeAlias = Literal["auto", "mps", "cuda", "cpu"]


def resolve_device(requested: DeviceName = "auto") -> TorchDevice:
    """Resolve a requested device, auto-detecting MPS, CUDA, then CPU.

    Args:
        requested: Explicit device name or ``"auto"`` for capability detection.

    Returns:
        The selected PyTorch device.

    Raises:
        NotImplementedError: The device-selection implementation is pending.
    """
    raise NotImplementedError
