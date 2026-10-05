"""Offline tests for the HTTP server, on a tiny random Qwen2 model.

The server must return exactly what the engine produces on its own, whether
the request is streamed, sent alone, or sent alongside other requests.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import Qwen2Config, Qwen2ForCausalLM

from engine import EngineConfig, LLMEngine
from server.api import create_app

CONFIG = EngineConfig(max_batch_size=4, max_tokens_per_batch=256, num_blocks=64)


class CharTokenizer:
    """One token per character; decode maps ids back to letters."""

    eos_token_id = 0

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[1 + (ord(ch) % 100) for ch in text])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(
            chr(97 + t % 26) for t in ids if not (skip_special_tokens and t == 0)
        )


@pytest.fixture(scope="module")
def tiny_model() -> Any:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model_cls: Any = Qwen2ForCausalLM
    model = model_cls(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def client(tiny_model: Any) -> Any:
    app = create_app(
        "tiny", "cpu", config=CONFIG, model=tiny_model, tokenizer=CharTokenizer()
    )
    with TestClient(app) as test_client:
        yield test_client


def _expected(model: Any, prompt: str, max_tokens: int) -> str:
    """What the engine alone generates for this prompt (greedy)."""
    tok = CharTokenizer()
    engine = LLMEngine(model, tok, CONFIG)
    [ids] = engine.generate([tok(prompt).input_ids], max_new_tokens=max_tokens)
    return tok.decode(ids)


def test_health(client: Any) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["running"] == 0


def test_completion_matches_engine(client: Any, tiny_model: Any) -> None:
    response = client.post(
        "/v1/completions", json={"prompt": "hello world", "max_tokens": 12}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == _expected(tiny_model, "hello world", 12)
    assert body["usage"]["prompt_tokens"] == len("hello world")
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_stream_matches_non_stream(client: Any) -> None:
    payload = {"prompt": "stream me", "max_tokens": 10}
    full = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]

    deltas: list[str] = []
    finish = None
    lines: list[str] = []
    with client.stream(
        "POST", "/v1/completions", json={**payload, "stream": True}
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                lines.append(line[len("data: ") :])
    assert lines[-1] == "[DONE]"
    for raw in lines[:-1]:
        choice = json.loads(raw)["choices"][0]
        deltas.append(choice["text"])
        finish = choice["finish_reason"] or finish
    assert "".join(deltas) == full
    assert finish in ("stop", "length")


def test_concurrent_requests_match_solo(client: Any, tiny_model: Any) -> None:
    """Six requests at once (batch size 4): every answer equals its solo run."""
    prompts = [f"request number {i} " * (i + 1) for i in range(6)]

    def call(prompt: str) -> str:
        response = client.post(
            "/v1/completions", json={"prompt": prompt, "max_tokens": 8}
        )
        assert response.status_code == 200
        text: str = response.json()["choices"][0]["text"]
        return text

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(call, prompts))
    for prompt, text in zip(prompts, results, strict=True):
        assert text == _expected(tiny_model, prompt, 8)
    assert client.get("/health").json()["running"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "x", "temperature": -1},
        {"prompt": "x", "max_tokens": 0},
        {"prompt": "x", "max_tokens": 5000},
        {"prompt": "x", "top_p": 0},
        {"prompt": ""},
    ],
)
def test_invalid_params_rejected(client: Any, payload: dict[str, Any]) -> None:
    assert client.post("/v1/completions", json=payload).status_code == 422


def test_prompt_too_long_rejected(client: Any) -> None:
    response = client.post(
        "/v1/completions", json={"prompt": "a" * 300, "max_tokens": 4}
    )
    assert response.status_code == 400
    assert "limit" in response.json()["detail"]
