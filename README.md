# mini-vLLM

A typed, educational skeleton for a compact LLM inference engine. The project
defines interfaces for model execution, KV-cache management, request
scheduling, token sampling, serving, and benchmarking.

All executable interfaces are intentionally stubs that raise
`NotImplementedError`.

## Layout

- `engine/`: core model, cache, scheduler, sampler, and engine interfaces
- `server/api.py`: FastAPI request, response, and application interfaces
- `benchmarks/harness.py`: benchmark input and result interfaces
- `tests/test_correctness.py`: skipped correctness-test placeholders
- `run.py`: generation CLI entry point
- `prompts.json`: sample benchmark prompts

## Device selection

The `engine.device.resolve_device` interface accepts `auto`, `mps`, `cuda`, or
`cpu`. Automatic selection is specified to prefer MPS, then CUDA, then CPU.
Both CLI entry points reserve `--device` as the explicit override.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The intended CLI shapes are:

```bash
mini-vllm --model MODEL_NAME --device auto
mini-vllm-benchmark --model MODEL_NAME --device cpu
```

These commands remain non-functional until their stubs are implemented.
