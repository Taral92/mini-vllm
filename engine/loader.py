"""Load the pinned Hugging Face model and tokenizer onto one device.

One place for the model id, the revision pin and the dtype choice, so tests,
the CLI and the benchmark harness all run exactly the same weights.
"""

from typing import Any, Literal, TypeAlias

import torch
from torch import device as TorchDevice

from .device import DeviceName, resolve_device

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
# Hub commit of Qwen/Qwen2.5-0.5B (last modified 2024-09-25). Pinning it means
# the M1 runs and the Kaggle T4 runs load byte-identical weights.
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

DTypeName: TypeAlias = Literal["auto", "bfloat16", "float16", "float32"]

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(device: TorchDevice, requested: DTypeName = "auto") -> torch.dtype:
    """Pick the weight dtype for ``device``.

    ``auto`` keeps the checkpoint's bfloat16 where the hardware supports it,
    falls back to float16 on GPUs without bf16 (the Kaggle T4 is sm_75), and
    uses float32 on CPU, where bf16 matmuls are slow.
    """
    if requested != "auto":
        return _DTYPES[requested]
    if device.type == "cuda":
        # Native bf16 needs compute capability 8.0+ (Ampere). Don't use
        # torch.cuda.is_bf16_supported(): it also counts slow software
        # emulation, so it says yes on a T4 (7.5).
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def load_model(
    model_name: str = MODEL_NAME,
    device: DeviceName = "auto",
    revision: str | None = None,
    dtype: DTypeName = "auto",
) -> tuple[Any, Any, TorchDevice]:
    """Load model and tokenizer in eval mode on the resolved device.

    Returns:
        ``(model, tokenizer, device)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = resolve_device(device)
    torch_dtype = resolve_dtype(torch_device, dtype)
    # Default model -> pinned commit. Any other model id or local path uses the
    # revision passed in, or its latest/local files when none is given.
    pin = revision or (MODEL_REVISION if model_name == MODEL_NAME else None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=pin)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name, revision=pin, dtype=torch_dtype
    )
    model.to(torch_device)
    model.eval()
    return model, tokenizer, torch_device
