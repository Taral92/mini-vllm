"""OpenAI-style HTTP server and chat UI over the continuous-batching engine.

    mini-vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct    # then open http://localhost:8000
    curl localhost:8000/v1/completions -H 'content-type: application/json' \
        -d '{"prompt": "The capital of France is", "max_tokens": 16}'

Endpoints:
    GET  /                      chat UI (server/static/index.html)
    POST /v1/completions        raw text continuation (any model)
    POST /v1/chat/completions   chat turns, rendered with the model's chat template
    GET  /health                engine load: running/waiting requests, KV blocks in use

How a request flows:
    1. The handler tokenizes the prompt, checks it fits, and hands it to the
       worker's inbox. It never touches the engine itself.
    2. One background task owns the engine. Between steps it admits everything
       in the inbox and drops aborted requests, then runs ``engine.step()`` in a
       thread so the event loop keeps accepting requests while the GPU works.
    3. Each new token goes onto that request's own queue. The handler turns the
       queue into one JSON response, or into Server-Sent Events when
       ``stream`` is true. If the client hangs up, the request is aborted and
       its seat and KV blocks are freed on the next step.

Only the worker calls the engine, so the engine needs no locks. Requests that
arrive while a step runs join the batch on the next step: that is continuous
batching seen from the outside.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine import EngineConfig, LLMEngine, SeqStatus
from engine.device import DeviceName
from engine.loader import MODEL_NAME, DTypeName
from engine.sampler import SamplingParams

logger = logging.getLogger("mini_vllm.server")

MAX_TOKENS_LIMIT = 1024  # per request; keeps one caller from holding a seat forever
UI_PATH = Path(__file__).parent / "static" / "index.html"


class CompletionRequest(BaseModel):
    """Subset of OpenAI's /v1/completions body."""

    prompt: str = Field(min_length=1, max_length=16_000)
    max_tokens: int = Field(default=64, ge=1, le=MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)  # 0 = greedy
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stream: bool = False
    model: str | None = None  # accepted for client compatibility, ignored


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=16_000)


class ChatRequest(BaseModel):
    """Subset of OpenAI's /v1/chat/completions body."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=512, ge=1, le=MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stream: bool = False
    model: str | None = None


# --------------------------------------------------------------------------
# Worker: the only code that calls the engine
# --------------------------------------------------------------------------


class _Finished:
    """Queue sentinel: the request is done."""


@dataclass(frozen=True)
class _Failed:
    """Queue sentinel: the engine raised while this request was in flight."""

    message: str


_DONE = _Finished()
QueueItem = int | _Finished | _Failed


@dataclass
class _Submission:
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    max_tokens: int
    queue: asyncio.Queue[QueueItem] = field(default_factory=asyncio.Queue)


class EngineWorker:
    """Runs the engine step loop in the background and fans tokens out."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._inbox: asyncio.Queue[_Submission] = asyncio.Queue()
        self._queues: dict[str, asyncio.Queue[QueueItem]] = {}
        self._aborts: set[str] = set()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="engine-worker")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._fail_all("server shutting down")

    def submit(
        self,
        request_id: str,
        prompt_ids: list[int],
        params: SamplingParams,
        max_tokens: int,
    ) -> asyncio.Queue[QueueItem]:
        """Queue a request for the next step; returns the queue its tokens land on."""
        sub = _Submission(request_id, prompt_ids, params, max_tokens)
        self._inbox.put_nowait(sub)
        return sub.queue

    def abort(self, request_id: str) -> None:
        """Ask the worker to drop a request before its next step."""
        self._aborts.add(request_id)

    def load(self) -> dict[str, int]:
        scheduler = self.engine.scheduler
        allocator = self.engine.cache.allocator
        return {
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting) + self._inbox.qsize(),
            "max_batch_size": scheduler.max_batch_size,
            "kv_blocks_used": allocator.num_blocks - allocator.num_free,
            "kv_blocks_total": allocator.num_blocks,
            "preemptions": scheduler.num_preemptions,
        }

    async def _run(self) -> None:
        while True:
            if not self.engine.has_unfinished():
                self._admit(await self._inbox.get())  # idle: sleep until work
            while not self._inbox.empty():
                self._admit(self._inbox.get_nowait())
            self._drop_aborted()
            if not self.engine.has_unfinished():
                continue
            try:
                emitted: dict[str, int] = await asyncio.to_thread(self.engine.step)
            except Exception as error:  # engine state is unknown now; fail loudly
                logger.exception("engine step failed")
                self._fail_all(f"engine error: {error}")
                raise
            for request_id, token in emitted.items():
                queue = self._queues[request_id]
                queue.put_nowait(token)
                seq = self.engine.sequences[request_id]
                if seq.status is SeqStatus.FINISHED:
                    queue.put_nowait(_DONE)
                    del self._queues[request_id]
                    self.engine.sequences.pop(request_id)

    def _admit(self, sub: _Submission) -> None:
        if sub.request_id in self._aborts:  # hung up before it was admitted
            self._aborts.discard(sub.request_id)
            return
        try:
            self.engine.add_request(
                sub.request_id, sub.prompt_ids, sub.params, sub.max_tokens
            )
        except ValueError as error:  # handler pre-checks make this rare
            sub.queue.put_nowait(_Failed(str(error)))
            return
        self._queues[sub.request_id] = sub.queue

    def _drop_aborted(self) -> None:
        for request_id in self._aborts:
            if self._queues.pop(request_id, None) is not None:
                self.engine.abort(request_id)
                logger.info("%s aborted (client disconnected)", request_id)
        self._aborts.clear()

    def _fail_all(self, message: str) -> None:
        for queue in self._queues.values():
            queue.put_nowait(_Failed(message))
        self._queues.clear()


# --------------------------------------------------------------------------
# Streaming detokenization
# --------------------------------------------------------------------------


class _Detokenizer:
    """Turns a growing token list into text deltas.

    Decodes the full list each time and sends only the new suffix: a single
    token can be half of a multi-byte character, so token-by-token decoding
    would emit broken text. A trailing U+FFFD means "wait for the next token".
    Quadratic in length, which is fine for max_tokens <= 1024.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.sent = ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        return self._delta(text)

    def flush(self) -> str:
        return self._delta(self.tokenizer.decode(self.ids, skip_special_tokens=True))

    def _delta(self, text: str) -> str:
        delta = text[len(self.sent) :] if text.startswith(self.sent) else ""
        self.sent = text
        return delta


# --------------------------------------------------------------------------
# Response shapes: /v1/completions vs /v1/chat/completions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Format:
    """How one endpoint shapes its JSON. ``kind`` is "text" or "chat"."""

    kind: Literal["text", "chat"]

    def chunk(
        self, meta: dict[str, Any], text: str, finish: str | None
    ) -> dict[str, Any]:
        if self.kind == "text":
            choice: dict[str, Any] = {"index": 0, "text": text}
        else:
            choice = {"index": 0, "delta": {"content": text} if text else {}}
        choice["finish_reason"] = finish
        object_name = (
            "text_completion" if self.kind == "text" else "chat.completion.chunk"
        )
        return {**meta, "object": object_name, "choices": [choice]}

    def first_chunk(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        if self.kind == "text":
            return None
        choice = {"index": 0, "delta": {"role": "assistant", "content": ""}}
        return {
            **meta,
            "object": "chat.completion.chunk",
            "choices": [{**choice, "finish_reason": None}],
        }

    def full(self, meta: dict[str, Any], text: str, finish: str) -> dict[str, Any]:
        if self.kind == "text":
            choice: dict[str, Any] = {"index": 0, "text": text}
            object_name = "text_completion"
        else:
            choice = {"index": 0, "message": {"role": "assistant", "content": text}}
            object_name = "chat.completion"
        choice["finish_reason"] = finish
        return {**meta, "object": object_name, "choices": [choice]}


TEXT = _Format("text")
CHAT = _Format("chat")


def _data(payload: dict[str, Any] | str) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {body}\n\n"


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def create_app(
    model_name: str = MODEL_NAME,
    device: DeviceName = "auto",
    dtype: DTypeName = "auto",
    config: EngineConfig | None = None,
    *,
    model: Any = None,
    tokenizer: Any = None,
) -> FastAPI:
    """Build the app. Weights load at startup unless ``model``/``tokenizer`` are given."""
    engine_config = config or EngineConfig()
    capacity = engine_config.num_blocks * engine_config.block_size

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded_model, loaded_tokenizer = model, tokenizer
        if loaded_model is None or loaded_tokenizer is None:
            from engine.loader import load_model

            loaded_model, loaded_tokenizer, _ = load_model(
                model_name, device, None, dtype
            )
        engine = LLMEngine(loaded_model, loaded_tokenizer, engine_config)
        worker = EngineWorker(engine)
        worker.start()
        app.state.tokenizer = loaded_tokenizer
        app.state.worker = worker
        app.state.device = str(next(loaded_model.parameters()).device.type)
        app.state.dtype = str(next(loaded_model.parameters()).dtype).replace(
            "torch.", ""
        )
        logger.info("serving %s with %s", model_name, engine_config)
        yield
        await worker.stop()

    app = FastAPI(title="mini-vllm", lifespan=lifespan)

    def check_fits(prompt_ids: list[int], max_tokens: int) -> None:
        if not prompt_ids:
            raise HTTPException(400, "prompt tokenized to zero tokens")
        if len(prompt_ids) > engine_config.max_tokens_per_batch:
            raise HTTPException(
                400,
                f"prompt is {len(prompt_ids)} tokens; "
                f"limit is {engine_config.max_tokens_per_batch}",
            )
        if len(prompt_ids) + max_tokens > capacity:
            raise HTTPException(
                400,
                f"prompt + max_tokens = {len(prompt_ids) + max_tokens} tokens; "
                f"KV cache holds {capacity}",
            )

    def fits(prompt_ids: list[int], max_tokens: int) -> bool:
        return (
            len(prompt_ids) <= engine_config.max_tokens_per_batch
            and len(prompt_ids) + max_tokens <= capacity
        )

    async def serve(
        request: Request,
        prompt_ids: list[int],
        params: SamplingParams,
        max_tokens: int,
        stream: bool,
        fmt: _Format,
    ) -> dict[str, Any] | StreamingResponse:
        worker: EngineWorker = request.app.state.worker
        tok = request.app.state.tokenizer
        prefix = "cmpl" if fmt.kind == "text" else "chatcmpl"
        request_id = f"{prefix}-{uuid.uuid4().hex[:16]}"
        queue = worker.submit(request_id, prompt_ids, params, max_tokens)
        started = time.perf_counter()
        meta = {"id": request_id, "created": int(time.time()), "model": model_name}

        def finish_reason(ids: list[int]) -> str:
            return "stop" if ids and ids[-1] == tok.eos_token_id else "length"

        def report(ids: list[int], first_at: float | None) -> dict[str, Any]:
            """Usage plus server-side timings (an extension to OpenAI's format)."""
            now = time.perf_counter()
            ttft = (first_at - started) if first_at is not None else 0.0
            decode = now - first_at if first_at is not None else 0.0
            n = len(ids)
            logger.info(
                "%s prompt=%d completion=%d ttft=%.0fms total=%.0fms",
                request_id,
                len(prompt_ids),
                n,
                ttft * 1000,
                (now - started) * 1000,
            )
            return {
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": n,
                    "total_tokens": len(prompt_ids) + n,
                },
                "metrics": {
                    "ttft_ms": round(ttft * 1000, 1),
                    "tpot_ms": round(decode / (n - 1) * 1000, 1) if n > 1 else None,
                    "tokens_per_sec": round((n - 1) / decode, 1) if n > 1 else None,
                    "total_ms": round((now - started) * 1000, 1),
                },
            }

        async def next_token(
            on_token: Callable[[int], None],
        ) -> AsyncIterator[int]:
            while True:
                item = await queue.get()
                if isinstance(item, _Failed):
                    raise RuntimeError(item.message)
                if isinstance(item, _Finished):
                    return
                on_token(item)
                yield item

        if stream:

            async def events() -> AsyncIterator[str]:
                detok = _Detokenizer(tok)
                first_at: list[float] = []
                done = False

                def mark(_: int) -> None:
                    if not first_at:
                        first_at.append(time.perf_counter())

                try:
                    first = fmt.first_chunk(meta)
                    if first is not None:
                        yield _data(first)
                    async for token in next_token(mark):
                        delta = detok.push(token)
                        if delta:
                            yield _data(fmt.chunk(meta, delta, None))
                    done = True
                    final = fmt.chunk(meta, detok.flush(), finish_reason(detok.ids))
                    yield _data(
                        {
                            **final,
                            **report(detok.ids, first_at[0] if first_at else None),
                        }
                    )
                    yield _data("[DONE]")
                except RuntimeError as error:
                    done = True
                    yield _data({"error": {"message": str(error)}})
                finally:
                    if not done:  # client hung up mid-stream: free the seat
                        worker.abort(request_id)

            return StreamingResponse(events(), media_type="text/event-stream")

        ids: list[int] = []
        first_at: list[float] = []
        done = False

        def mark_first(_: int) -> None:
            if not first_at:
                first_at.append(time.perf_counter())

        try:
            async for token in next_token(mark_first):
                ids.append(token)
            done = True
        except RuntimeError as error:
            done = True
            raise HTTPException(500, str(error)) from error
        finally:
            if not done:
                worker.abort(request_id)
        text = tok.decode(ids, skip_special_tokens=True)
        return {
            **fmt.full(meta, text, finish_reason(ids)),
            **report(ids, first_at[0] if first_at else None),
        }

    @app.get("/", include_in_schema=False)
    async def ui() -> HTMLResponse:
        return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        worker: EngineWorker = request.app.state.worker
        tok = request.app.state.tokenizer
        return {
            "status": "ok",
            "model": model_name,
            "device": request.app.state.device,
            "dtype": request.app.state.dtype,
            "chat": getattr(tok, "chat_template", None) is not None,
            **worker.load(),
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(
        body: CompletionRequest, request: Request
    ) -> dict[str, Any] | StreamingResponse:
        tok = request.app.state.tokenizer
        prompt_ids = list(tok(body.prompt).input_ids)
        check_fits(prompt_ids, body.max_tokens)
        params = SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            seed=body.seed,
        )
        return await serve(
            request, prompt_ids, params, body.max_tokens, body.stream, TEXT
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatRequest, request: Request
    ) -> dict[str, Any] | StreamingResponse:
        tok = request.app.state.tokenizer
        if getattr(tok, "chat_template", None) is None:
            raise HTTPException(
                400,
                f"{model_name} has no chat template; use /v1/completions or "
                "start the server with an -Instruct model",
            )
        messages = [m.model_dump() for m in body.messages]

        def encode(turns: list[dict[str, str]]) -> list[int]:
            text = tok.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=True
            )
            return list(tok(text).input_ids)

        # Long chats: drop the oldest turns (keeping any system prompt) until the
        # prompt fits the cache, so the conversation keeps going.
        prompt_ids = encode(messages)
        while not fits(prompt_ids, body.max_tokens) and len(messages) > 1:
            drop = 1 if messages[0]["role"] == "system" and len(messages) > 2 else 0
            messages.pop(drop)
            prompt_ids = encode(messages)
        check_fits(prompt_ids, body.max_tokens)

        params = SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            seed=body.seed,
        )
        return await serve(
            request, prompt_ids, params, body.max_tokens, body.stream, CHAT
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Run the server with uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="mini-vllm-serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="use Qwen/Qwen2.5-0.5B-Instruct for the chat UI",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"]
    )
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=512)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = EngineConfig(
        max_batch_size=args.max_batch_size, num_blocks=args.num_blocks
    )
    app = create_app(args.model, args.device, args.dtype, config)
    print(f"\n  mini-vLLM chat UI: http://{args.host}:{args.port}\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
