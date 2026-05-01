"""Lifecycle ABC shared by spawn-based producer pools.

Owns the machinery that every background producer needs and that has
nothing to do with the decoded payload shape: process/thread worker
management, warmup/stop events, error-queue draining, optional live
update queue.

The inner dispatch loop (asyncio in video, threaded in text) is NOT
part of the base — subclasses supply it via ``_create_worker()`` by
returning a fresh ``mp.Process`` or ``threading.Thread`` that runs the
modality-specific producer loop.

Extension points:

- ``_create_worker()`` — abstract.  Return a fresh
  ``mp.Process``/``threading.Thread`` whose ``target`` is the
  subclass's spawn entry function.  Called once per ``start()``.
- ``_DEFAULT_WARMUP_TIMEOUT_S`` — class attribute.  Override to raise
  or lower the default ``wait_for_warmup`` timeout.

State the subclass must set before the base's methods work:

- ``self._buffer`` — any object with ``capacity`` and ``__len__``.
- ``self._warmup_target`` — int.
- ``self._warmup_event`` — ``mp.Event`` or ``threading.Event``.
- ``self._stop_event`` — same.
- ``self._error_queue`` — ``mp.Queue`` or ``None`` (thread mode).
- ``self._update_queue`` — ``mp.Queue`` or ``None`` (no live updates).
- ``self._worker`` — initialized to ``None``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseProducerPool(ABC):
    """Spawn-child (or thread) producer pool lifecycle.

    Subclasses supply:

    - ``_create_worker() -> mp.Process | threading.Thread``: returns a
      fresh worker object (not yet started) whose ``target`` is the
      modality-specific spawn-entry function.  Called once per
      ``start()``.

    The subclass is responsible for setting ``self._buffer``,
    ``self._warmup_target``, and the event/queue attributes *before*
    any base method is called.  The canonical place is the subclass's
    ``__init__``.
    """

    # Default warmup timeout in seconds.  Can be overridden per-subclass.
    # 1200s accommodates slow cold starts (e.g. ffmpeg's first-decode
    # codec init at scale) without firing spuriously.
    _DEFAULT_WARMUP_TIMEOUT_S: float = 1200.0

    # Subclass-initialized attributes (type hints only — the subclass
    # must set these in its __init__).
    _buffer: Any
    _warmup_target: int
    _warmup_event: Any
    _stop_event: Any
    _error_queue: Any | None
    _update_queue: Any | None
    _worker: Any | None

    # ------------------------------------------------------------------
    # Subclass extension point
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_worker(self):
        """Return a fresh worker (Process or Thread), not yet started.

        Called exactly once per ``start()``.  Must set up whatever
        ``target`` + ``args`` the subclass needs for its modality.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker.  Raises if already running."""
        if self.is_alive:
            raise RuntimeError(f"{type(self).__name__} already running")
        self._stop_event.clear()
        self._warmup_event.clear()
        self._worker = self._create_worker()
        self._worker.start()

    def stop(self) -> None:
        """Signal stop and reap the worker.

        ``join`` waits up to 10s; if the worker has a ``terminate``
        method (Process) and is still alive, we terminate it.  Threads
        have no terminate — they exit when ``_stop_event`` is honored
        by the producer loop.
        """
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=10)
            if hasattr(self._worker, "terminate") and self._worker.is_alive():
                self._worker.terminate()
            self._worker = None

    def wait_for_warmup(self, timeout: float | None = None) -> None:
        """Block until the buffer holds ``warmup_target`` items.

        Errors reported by the child through ``error_queue`` are
        re-raised here so they surface in the parent's training log —
        child-process logger output often doesn't reach the parent's
        handler.

        Args:
            timeout: Seconds to wait.  ``None`` uses
                :attr:`_DEFAULT_WARMUP_TIMEOUT_S`.

        Raises:
            TimeoutError: if the warmup event doesn't fire in time and
                no child error is queued.
            RuntimeError: if the child reported an error, with or
                without timing out.
        """
        t = timeout if timeout is not None else self._DEFAULT_WARMUP_TIMEOUT_S
        if not self._warmup_event.wait(timeout=t):
            err = self._drain_error_queue()
            if err:
                raise RuntimeError(
                    f"{type(self).__name__} child failed: {err}"
                )
            raise TimeoutError(
                f"{type(self).__name__} didn't fill {self._warmup_target} "
                f"items in {t}s (have {len(self._buffer)})"
            )
        # Success path — still check: the child may have set warmup in
        # its ``finally`` block after an error.
        err = self._drain_error_queue()
        if err:
            raise RuntimeError(
                f"{type(self).__name__} child failed: {err}"
            )

    def _drain_error_queue(self) -> str | None:
        """Read the first queued child error, if any."""
        if self._error_queue is None:
            return None
        try:
            return self._error_queue.get_nowait()
        except Exception:
            return None

    @property
    def is_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # ------------------------------------------------------------------
    # Live update queue (optional — set ``self._update_queue`` to wire)
    # ------------------------------------------------------------------

    def update_episodes(
        self, source_name: str, new_episodes: list[int],
    ) -> None:
        """Push a fresh work queue for a running source.

        Sends ``(source_name, new_episodes)`` through ``_update_queue``;
        the child's update poller (if any) swaps its iterator
        atomically.  No-op with warning if the pool wasn't constructed
        with an update queue (thread mode, or subclasses that disable
        live updates).

        Safe to call from forked DataLoader workers: ``mp.Queue`` put
        handles survive fork, and we deliberately *don't* check
        ``is_alive`` here because that would invoke
        ``mp.Process.is_alive()`` on a handle owned by the parent,
        which raises ``AssertionError`` from any non-owning process.
        If the pool is actually dead, ``put_nowait`` fails loudly with
        the real exception — more informative than a pre-flight
        ownership assertion.

        Args:
            source_name: Routing key — must match a
                ``ProducerConfig.source_name`` the child knows about.
            new_episodes: Full replacement list.  Currently-dispatched
                decodes finish on the old list; subsequent dispatches
                use the new list.
        """
        if self._update_queue is None:
            logger.warning(
                f"{type(self).__name__}.update_episodes: no-op "
                "(pool constructed without an update queue)"
            )
            return
        try:
            self._update_queue.put_nowait(
                (source_name, list(new_episodes)),
            )
        except Exception as e:
            logger.warning(f"update_episodes: failed to enqueue: {e}")
