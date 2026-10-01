"""Device selection and synchronization helpers for the inference engine."""

from typing import Literal, TypeAlias

import torch
from torch import device as TorchDevice

DeviceName: TypeAlias = Literal["auto", "mps", "cuda", "cpu"]


def _available(name: str) -> bool:
    if name == "mps":
        return torch.backends.mps.is_available()
    if name == "cuda":
        return torch.cuda.is_available()
    return name == "cpu"


def resolve_device(requested: DeviceName = "auto") -> TorchDevice:
    """Resolve a requested device, auto-detecting MPS, CUDA, then CPU.

    Args:
        requested: Explicit device name or ``"auto"`` for capability detection.

    Returns:
        The selected PyTorch device.

    Raises:
        ValueError: The name is unknown.
        RuntimeError: An explicitly requested device is not available.
    """
    if requested == "auto":
        for name in ("mps", "cuda", "cpu"):
            if _available(name):
                return torch.device(name)
    if requested not in ("mps", "cuda", "cpu"):
        raise ValueError(f"unknown device {requested!r}; use auto, mps, cuda or cpu")
    if not _available(requested):
        raise RuntimeError(f"device {requested!r} was requested but is not available")
    return torch.device(requested)


def synchronize(device: TorchDevice) -> None:
    """Block until all queued work on ``device`` has finished.

    GPU kernels launch asynchronously, so a timer stopped without this measures
    launch time, not compute time. Call it before every timer stop.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
