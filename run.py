"""Command-line entry point for mini-vLLM.

mini-vllm --prompt "Explain paged attention in simple terms."
mini-vllm --prompt "Write a haiku" --temperature 0.8 --top-p 0.9 --device cpu
"""

import argparse
import sys
import time
from collections.abc import Sequence

import torch


def _build_parser() -> argparse.ArgumentParser:
    from engine.loader import MODEL_NAME

    parser = argparse.ArgumentParser(
        prog="mini-vllm", description="Generate text with mini-vLLM."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--revision",
        default=None,
        help="hub commit; default is the pin in engine/loader.py",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--mode",
        default="kv",
        choices=["kv", "naive"],
        help="kv = cached decode, naive = full re-forward",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run text generation with an optional ``--device`` override.

    The device option accepts ``auto``, ``mps``, ``cuda``, or ``cpu`` and
    defaults to automatic selection in that priority order.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit status.
    """
    args = _build_parser().parse_args(argv)
    if args.max_new_tokens < 1:
        print("--max-new-tokens must be at least 1", file=sys.stderr)
        return 2

    from engine.device import synchronize
    from engine.engine import generate_cached, generate_naive
    from engine.loader import load_model

    model, tokenizer, device = load_model(
        args.model, args.device, args.revision, args.dtype
    )
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    generate = generate_cached if args.mode == "kv" else generate_naive

    torch.manual_seed(args.seed)
    synchronize(device)
    start = time.perf_counter()
    tokens = generate(
        model,
        ids,
        max_new=args.max_new_tokens,
        eos_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start

    print(tokenizer.decode(tokens, skip_special_tokens=True))
    print(
        f"\n[{len(tokens)} tokens in {elapsed:.2f}s = {len(tokens) / elapsed:.1f} tok/s, "
        f"mode={args.mode}, device={device.type}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
