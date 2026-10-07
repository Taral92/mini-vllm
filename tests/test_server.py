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


# --- chat endpoint, UI, abort ------------------------------------------------


class ChatTokenizer(CharTokenizer):
    """CharTokenizer plus a toy chat template."""

    chat_template = "toy"

    def apply_chat_template(
        self,
        turns: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        text = "".join(f"[{t['role']}]{t['content']}" for t in turns)
        return text + ("[assistant]" if add_generation_prompt else "")


@pytest.fixture(scope="module")
def chat_client(tiny_model: Any) -> Any:
    app = create_app(
        "tiny-chat", "cpu", config=CONFIG, model=tiny_model, tokenizer=ChatTokenizer()
    )
    with TestClient(app) as test_client:
        yield test_client


CHAT = {
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi there"},
    ],
    "max_tokens": 12,
    "temperature": 0,
}


def test_chat_matches_engine(chat_client: Any, tiny_model: Any) -> None:
    body = chat_client.post("/v1/chat/completions", json=CHAT).json()
    assert body["object"] == "chat.completion"
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    prompt = ChatTokenizer().apply_chat_template(CHAT["messages"])
    assert message["content"] == _expected(tiny_model, prompt, 12)
    assert body["usage"]["prompt_tokens"] == len(prompt)
    assert body["metrics"]["ttft_ms"] >= 0


def test_chat_stream_matches_non_stream(chat_client: Any) -> None:
    full = chat_client.post("/v1/chat/completions", json=CHAT).json()
    events: list[dict[str, Any]] = []
    with chat_client.stream(
        "POST", "/v1/chat/completions", json={**CHAT, "stream": True}
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: ") :]))
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == full["choices"][0]["message"]["content"]
    assert events[-1]["usage"]["completion_tokens"] == 12
    assert events[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_chat_needs_chat_template(client: Any) -> None:
    response = client.post("/v1/chat/completions", json=CHAT)
    assert response.status_code == 400
    assert "chat template" in response.json()["detail"]


def test_long_chat_drops_oldest_turns(chat_client: Any) -> None:
    """A history longer than max_tokens_per_batch is trimmed, not rejected."""
    turns = [{"role": "user", "content": "x" * 60} for _ in range(8)]
    body = chat_client.post(
        "/v1/chat/completions", json={"messages": turns, "max_tokens": 4}
    ).json()
    assert body["usage"]["prompt_tokens"] <= CONFIG.max_tokens_per_batch


def test_ui_and_health(chat_client: Any) -> None:
    page = chat_client.get("/")
    assert page.status_code == 200
    assert "mini-vLLM" in page.text and "/v1/chat/completions" in page.text
    health = chat_client.get("/health").json()
    assert health["chat"] is True
    assert health["max_batch_size"] == CONFIG.max_batch_size
    assert health["kv_blocks_total"] == CONFIG.num_blocks
    assert health["kv_blocks_used"] == 0


def test_worker_abort_frees_seat(tiny_model: Any) -> None:
    """A client hang-up mid-generation frees the request's seat and blocks."""
    import asyncio

    from engine.sampler import SamplingParams
    from server.api import EngineWorker

    async def scenario() -> None:
        engine = LLMEngine(tiny_model, CharTokenizer(), CONFIG)
        worker = EngineWorker(engine)
        worker.start()
        greedy = SamplingParams(temperature=0)
        queue = worker.submit("long", [5, 6, 7], greedy, 200)
        await queue.get()  # first token: it is running
        worker.abort("long")
        for _ in range(200):
            await asyncio.sleep(0.005)
            if not engine.has_unfinished():
                break
        assert not engine.has_unfinished()
        assert engine.cache.allocator.num_free == CONFIG.num_blocks
        await worker.stop()

    asyncio.run(scenario())
