"""OpenAI-style HTTP server over the continuous-batching engine.

    mini-vllm-serve --port 8000
    curl localhost:8000/v1/completions -H 'content-type: application/json' \
        -d '{"prompt": "The capital of France is", "max_tokens": 16}'

How a request flows:
    1. The handler tokenizes the prompt, checks it fits, and hands it to the
       worker's inbox. It never touches the engine itself.
    2. One background task owns the engine. Between steps it admits everything
       in the inbox, then runs ``engine.step()`` in a thread so the event loop
       keeps accepting requests while the GPU works.
    3. Each new token goes onto that request's own queue. The handler turns the
       queue into one JSON response, or into Server-Sent Events when
       ``stream`` is true.

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
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.device import DeviceName
from engine.loader import MODEL_NAME, DTypeName
from engine.sampler import SamplingParams
from engine import EngineConfig, LLMEngine, SeqStatus

logger = logging.getLogger("mini_vllm.server")

MAX_TOKENS_LIMIT = 1024  # per request; keeps one caller from holding a seat forever


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

    def load(self) -> dict[str, int]:
        scheduler = self.engine.scheduler
        return {
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting) + self._inbox.qsize(),
        }

    async def _run(self) -> None:
        while True:
            if not self.engine.has_unfinished():
                self._admit(await self._inbox.get())  # idle: sleep until work
            while not self._inbox.empty():
                self._admit(self._inbox.get_nowait())
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
        try:
            self.engine.add_request(
                sub.request_id, sub.prompt_ids, sub.params, sub.max_tokens
            )
        except ValueError as error:  # handler pre-checks make this rare
            sub.queue.put_nowait(_Failed(str(error)))
            return
        self._queues[sub.request_id] = sub.queue

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
        logger.info("serving %s with %s", model_name, engine_config)
        yield
        await worker.stop()

    app = FastAPI(title="mini-vllm", lifespan=lifespan)
    capacity = engine_config.num_blocks * engine_config.block_size

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        worker: EngineWorker = request.app.state.worker
        return {"status": "ok", "model": model_name, **worker.load()}

    @app.post("/v1/completions", response_model=None)
    async def completions(
        body: CompletionRequest, request: Request
    ) -> dict[str, Any] | StreamingResponse:
        worker: EngineWorker = request.app.state.worker
        tok = request.app.state.tokenizer
        prompt_ids = list(tok(body.prompt).input_ids)
        if not prompt_ids:
            raise HTTPException(400, "prompt tokenized to zero tokens")
        if len(prompt_ids) > engine_config.max_tokens_per_batch:
            raise HTTPException(
                400,
                f"prompt is {len(prompt_ids)} tokens; "
                f"limit is {engine_config.max_tokens_per_batch}",
            )
        if len(prompt_ids) + body.max_tokens > capacity:
            raise HTTPException(
                400,
                f"prompt + max_tokens = {len(prompt_ids) + body.max_tokens} tokens; "
                f"KV cache holds {capacity}",
            )

        request_id = f"cmpl-{uuid.uuid4().hex[:16]}"
        params = SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            seed=body.seed,
        )
        queue = worker.submit(request_id, prompt_ids, params, body.max_tokens)
        started = time.perf_counter()
        meta = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
        }

        def finish_reason(ids: list[int]) -> str:
            return "stop" if ids and ids[-1] == tok.eos_token_id else "length"

        def log_done(ids: list[int]) -> None:
            logger.info(
                "%s prompt=%d completion=%d %.0fms",
                request_id,
                len(prompt_ids),
                len(ids),
                (time.perf_counter() - started) * 1000,
            )

        if body.stream:

            async def events() -> AsyncIterator[str]:
                detok = _Detokenizer(tok)
                while True:
                    item = await queue.get()
                    if isinstance(item, _Failed):
                        error = {"error": {"message": item.message}}
                        yield f"data: {json.dumps(error)}\n\n"
                        return
                    if isinstance(item, _Finished):
                        break
                    delta = detok.push(item)
                    if delta:
                        yield _sse(meta, delta, None)
                yield _sse(meta, detok.flush(), finish_reason(detok.ids))
                yield "data: [DONE]\n\n"
                log_done(detok.ids)

            return StreamingResponse(events(), media_type="text/event-stream")

        ids: list[int] = []
        while True:
            item = await queue.get()
            if isinstance(item, _Failed):
                raise HTTPException(500, item.message)
            if isinstance(item, _Finished):
                break
            ids.append(item)
        log_done(ids)
        return {
            **meta,
            "choices": [
                {
                    "index": 0,
                    "text": tok.decode(ids, skip_special_tokens=True),
                    "finish_reason": finish_reason(ids),
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(ids),
                "total_tokens": len(prompt_ids) + len(ids),
            },
        }

    return app


def _sse(meta: dict[str, Any], text: str, finish: str | None) -> str:
    chunk = {**meta, "choices": [{"index": 0, "text": text, "finish_reason": finish}]}
    return f"data: {json.dumps(chunk)}\n\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the server with uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="mini-vllm-serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=MODEL_NAME)
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
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
