"""Command-line entry point for mini-vLLM."""

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run text generation with an optional ``--device`` override.

    The device option accepts ``auto``, ``mps``, ``cuda``, or ``cpu`` and
    defaults to automatic selection in that priority order.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit status.

    Raises:
        NotImplementedError: CLI handling is pending.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
