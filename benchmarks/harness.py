"""Benchmark harness interfaces."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from engine.device import DeviceName


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate latency and throughput measurements."""

    request_count: int
    elapsed_seconds: float
    tokens_per_second: float


def load_prompts(path: Path) -> tuple[str, ...]:
    """Load benchmark prompts from a JSON file.

    Args:
        path: JSON file containing prompt strings.

    Returns:
        Prompt strings in source order.

    Raises:
        NotImplementedError: Prompt loading is pending.
    """
    raise NotImplementedError


def run_benchmark(
    model_name: str,
    prompts: Sequence[str],
    device: DeviceName = "auto",
) -> BenchmarkResult:
    """Benchmark generation for a sequence of prompts.

    Args:
        model_name: Hugging Face model identifier or local model path.
        prompts: Input prompts to generate from.
        device: Explicit execution device or ``"auto"``.

    Returns:
        Aggregate benchmark measurements.

    Raises:
        NotImplementedError: Benchmark execution is pending.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI, including its ``--device`` override.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit status.

    Raises:
        NotImplementedError: Benchmark CLI handling is pending.
    """
    raise NotImplementedError
