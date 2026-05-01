"""Tests for TimedDataLoader."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataporter.timed_loader import TimedDataLoader


class TestTimedDataLoader:

    def _make_loader(self, n: int = 100, batch_size: int = 16):
        data = TensorDataset(torch.randn(n, 32))
        return DataLoader(data, batch_size=batch_size)

    def test_produces_same_batches(self):
        loader = self._make_loader()
        timed = TimedDataLoader(loader)

        original_batches = [b for b in DataLoader(loader.dataset, batch_size=16)]
        timed_batches = list(timed)

        assert len(timed_batches) == len(original_batches)
        for a, b in zip(original_batches, timed_batches):
            assert torch.equal(a[0], b[0])

    def test_records_fetch_time(self):
        timed = TimedDataLoader(self._make_loader())
        for batch in timed:
            pass
        assert timed.last_fetch_ms > 0
        assert timed.fetch_ema_ms > 0

    def test_ema_smooths(self):
        timed = TimedDataLoader(self._make_loader(n=200), ema_alpha=0.5)
        times = []
        for batch in timed:
            times.append(timed.last_fetch_ms)

        # EMA should be between min and max of raw times
        assert timed.fetch_ema_ms >= min(times) * 0.5
        assert timed.fetch_ema_ms <= max(times) * 2

    def test_len_delegates(self):
        loader = self._make_loader(n=100, batch_size=10)
        timed = TimedDataLoader(loader)
        assert len(timed) == 10

    def test_attribute_delegation(self):
        loader = self._make_loader(batch_size=32)
        timed = TimedDataLoader(loader)
        assert timed.batch_size == 32

    def test_initial_ema_is_zero(self):
        timed = TimedDataLoader(self._make_loader())
        assert timed.fetch_ema_ms == 0.0
