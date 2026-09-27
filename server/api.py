"""HTTP API interfaces for mini-vLLM."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from engine.device import DeviceName
from engine.sampler import SamplingParams


class GenerateRequest(BaseModel):
    """Payload accepted by the text-generation endpoint."""

    prompt: str
    max_new_tokens: int = Field(default=32, ge=1)
    sampling: SamplingParams = SamplingParams()


class GenerateResponse(BaseModel):
    """Payload returned by the text-generation endpoint."""

    text: str


def create_app(model_name: str, device: DeviceName = "auto") -> FastAPI:
    """Create an HTTP application backed by one inference engine.

    Args:
        model_name: Hugging Face model identifier or local model path.
        device: Explicit execution device or ``"auto"``.

    Returns:
        Configured FastAPI application.

    Raises:
        NotImplementedError: API construction is pending.
    """
    raise NotImplementedError
