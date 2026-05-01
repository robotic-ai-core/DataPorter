"""Unit tests for WeightedRoundRobinDispatcher.

Locks in the long-run ratio property: given weights ``{a: 3, b: 1}``,
a stream of picks must contain ``a`` 3× as often as ``b`` with zero
variance over complete cycles.
"""

from __future__ import annotations

from collections import Counter

import pytest

from dataporter._blending import WeightedRoundRobinDispatcher


class TestWeightedRoundRobinDispatcher:

    def test_single_source(self):
        d = WeightedRoundRobinDispatcher({"a": 1.0})
        picks = [d.next() for _ in range(20)]
        assert picks == ["a"] * 20

    def test_equal_weights(self):
        d = WeightedRoundRobinDispatcher({"a": 1.0, "b": 1.0})
        # 20 picks = 10 complete 2-long cycles; each source hits exactly 10.
        picks = [d.next() for _ in range(20)]
        c = Counter(picks)
        assert c["a"] == 10
        assert c["b"] == 10

    def test_three_to_one(self):
        d = WeightedRoundRobinDispatcher({"a": 3.0, "b": 1.0})
        # 40 picks = 10 complete 4-long cycles; a=30, b=10.
        picks = [d.next() for _ in range(40)]
        c = Counter(picks)
        assert c["a"] == 30
        assert c["b"] == 10

    def test_three_sources_varied_weights(self):
        d = WeightedRoundRobinDispatcher(
            {"a": 5.0, "b": 2.0, "c": 1.0},
        )
        # 40 picks = 5 complete 8-long cycles.
        picks = [d.next() for _ in range(40)]
        c = Counter(picks)
        assert c["a"] == 25
        assert c["b"] == 10
        assert c["c"] == 5

    def test_determinism(self):
        """Two dispatchers with identical weights produce identical
        streams — no hidden RNG state."""
        d1 = WeightedRoundRobinDispatcher({"a": 2.0, "b": 1.0, "c": 1.0})
        d2 = WeightedRoundRobinDispatcher({"a": 2.0, "b": 1.0, "c": 1.0})
        s1 = [d1.next() for _ in range(64)]
        s2 = [d2.next() for _ in range(64)]
        assert s1 == s2

    def test_rejects_empty_weights(self):
        with pytest.raises(ValueError, match="weights empty"):
            WeightedRoundRobinDispatcher({})

    def test_rejects_nonpositive_weights(self):
        with pytest.raises(ValueError, match="must be > 0"):
            WeightedRoundRobinDispatcher({"a": 1.0, "b": 0.0})
        with pytest.raises(ValueError, match="must be > 0"):
            WeightedRoundRobinDispatcher({"a": 1.0, "b": -0.5})

    def test_ratio_holds_over_long_run(self):
        """Regardless of cycle boundaries, the long-run ratio converges."""
        d = WeightedRoundRobinDispatcher({"a": 7.0, "b": 3.0})
        picks = [d.next() for _ in range(10_000)]
        c = Counter(picks)
        # Exact multiple of 10 → exact ratio.
        assert c["a"] == 7000
        assert c["b"] == 3000
