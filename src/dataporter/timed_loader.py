"""Instrumented DataLoader wrapper that measures per-batch fetch time.

Wraps a DataLoader iterator to timestamp each ``__next__`` call,
giving the true data pipeline latency without framework overhead.

Usage with PyTorch Lightning::

    loader = TimedDataLoader(original_loader)
    # In your callback:
    pl_module.log("perf/dl_fetch_ms", loader.last_fetch_ms)
    pl_module.log("perf/dl_fetch_ema_ms", loader.fetch_ema_ms)

The key metric is ``dl_fetch_ms`` — the wall time of the actual
DataLoader ``__next__`` call. Compare with ``data_wait_ms`` (which
includes all framework overhead between batch_end and batch_start)
to see how much of the "data wait" is actually data vs. framework.
"""

from __future__ import annotations

import time
from typing import Iterator

from torch.utils.data import DataLoader


class TimedDataLoader:
    """DataLoader wrapper that measures per-batch fetch time.

    Delegates all DataLoader attributes to the wrapped loader.
    Adds timing instrumentation on the iterator's ``__next__``.

    Args:
        loader: The DataLoader to wrap.
        ema_alpha: Smoothing factor for EMA (0-1, higher = more recent).
    """

    def __init__(self, loader: DataLoader, ema_alpha: float = 0.1):
        self._loader = loader
        self._ema_alpha = ema_alpha
        self._last_fetch_ms: float = 0.0
        self._fetch_ema_ms: float | None = None

    @property
    def last_fetch_ms(self) -> float:
        """Wall time of the most recent __next__ call (milliseconds)."""
        return self._last_fetch_ms

    @property
    def fetch_ema_ms(self) -> float:
        """EMA-smoothed fetch time (milliseconds)."""
        return self._fetch_ema_ms if self._fetch_ema_ms is not None else 0.0

    def __iter__(self) -> Iterator:
        return _TimedIterator(self, iter(self._loader))

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


class _TimedIterator:
    """Iterator that timestamps each __next__ call."""

    def __init__(self, parent: TimedDataLoader, inner: Iterator):
        self._parent = parent
        self._inner = inner

    def __next__(self):
        t0 = time.perf_counter()
        batch = next(self._inner)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._parent._last_fetch_ms = elapsed_ms
        if self._parent._fetch_ema_ms is None:
            self._parent._fetch_ema_ms = elapsed_ms
        else:
            a = self._parent._ema_alpha
            self._parent._fetch_ema_ms = (
                a * elapsed_ms + (1 - a) * self._parent._fetch_ema_ms
            )

        return batch

    def __iter__(self):
        return self
