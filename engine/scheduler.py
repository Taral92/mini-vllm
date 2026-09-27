"""Request scheduling interfaces."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Request:
    """A tokenized generation request waiting for engine execution."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int


class Scheduler:
    """Select requests for prefill and decode batches."""

    def __init__(self, max_batch_size: int, max_tokens_per_batch: int) -> None:
        """Configure scheduler capacity.

        Args:
            max_batch_size: Maximum requests in one execution batch.
            max_tokens_per_batch: Maximum tokens processed in one batch.

        Raises:
            NotImplementedError: Scheduler construction is pending.
        """
        raise NotImplementedError

    def add(self, request: Request) -> None:
        """Add a request to the scheduling queue.

        Args:
            request: Tokenized request to enqueue.

        Raises:
            NotImplementedError: Request enqueueing is pending.
        """
        raise NotImplementedError

    def schedule(self) -> tuple[Request, ...]:
        """Select the next batch of requests.

        Returns:
            Requests selected for the next model step.

        Raises:
            NotImplementedError: Batch scheduling is pending.
        """
        raise NotImplementedError

    def complete(self, request_id: str) -> None:
        """Mark a request complete and remove its scheduler state.

        Args:
            request_id: Stable identifier for the completed request.

        Raises:
            NotImplementedError: Request completion is pending.
        """
        raise NotImplementedError

    def has_pending(self) -> bool:
        """Return whether queued or active requests remain.

        Raises:
            NotImplementedError: Pending-state inspection is pending.
        """
        raise NotImplementedError
