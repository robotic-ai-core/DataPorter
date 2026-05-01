"""Weighted round-robin source dispatcher.

Shared utility used by multi-source producer pools (video today;
any future modality that blends from N sources with relative weights
can pick it up).  Extracted from duplicated in-line implementations
in :mod:`producer_pool`.

The dispatcher yields source names in a ratio that respects each
source's ``weight`` over the long run.  Each ``next()`` call picks
the source whose *token deficit* (weight − tokens-consumed) is
largest; tokens accumulate until every source has met its weight,
then reset.  Deterministic, lock-free, O(N) per pick where N is the
number of sources — trivial for N ≤ a few dozen.
"""

from __future__ import annotations

from typing import Iterable


class WeightedRoundRobinDispatcher:
    """Deterministic weighted source picker.

    Args:
        weights: ``{source_name: relative_weight}``.  All weights must
            be > 0.  Absolute scale doesn't matter — only the ratio.

    Example:
        >>> d = WeightedRoundRobinDispatcher({"a": 3.0, "b": 1.0})
        >>> picks = [d.next() for _ in range(8)]
        >>> picks.count("a"), picks.count("b")
        (6, 2)
    """

    def __init__(self, weights: dict[str, float]):
        if not weights:
            raise ValueError("WeightedRoundRobinDispatcher: weights empty")
        if any(w <= 0 for w in weights.values()):
            raise ValueError(
                f"WeightedRoundRobinDispatcher: all weights must be > 0, "
                f"got {weights!r}"
            )
        self._weights: dict[str, float] = dict(weights)
        self._tokens: dict[str, float] = {name: 0.0 for name in weights}

    @property
    def source_names(self) -> Iterable[str]:
        return self._weights.keys()

    def next(self) -> str:
        """Pick the next source name by maximum weight-minus-tokens.

        After the pick, increment that source's token count.  When
        every source has met or exceeded its weight, reset all tokens
        to zero so the next cycle starts fresh.
        """
        name = max(
            self._tokens,
            key=lambda n: self._weights[n] - self._tokens[n],
        )
        self._tokens[name] += 1
        if all(
            t >= self._weights[n] for n, t in self._tokens.items()
        ):
            for k in self._tokens:
                self._tokens[k] = 0.0
        return name
