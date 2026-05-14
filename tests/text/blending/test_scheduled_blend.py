"""Tests for ScheduledBlendDataset — the unified N-way weighted mixer."""

from __future__ import annotations

import random as _random

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.text.blending import ScheduledBlendDataset


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


class _StubDataset(Dataset):
    """Map-style dataset whose items report their tag + index."""

    def __init__(self, tag: str, n: int = 10):
        self._tag = tag
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {"tag": self._tag, "idx": idx}


def _w(value: float) -> torch.Tensor:
    t = torch.tensor([value], dtype=torch.float64)
    t.share_memory_()
    return t


def _mk(*specs: tuple[str, float, int]) -> ScheduledBlendDataset:
    """Build a dataset from (name, weight, n) triples."""
    return ScheduledBlendDataset(
        [(_StubDataset(name, n=n), _w(w), name) for name, w, n in specs]
    )


# ----------------------------------------------------------------------
# Construction validation
# ----------------------------------------------------------------------


class TestConstruction:
    def test_empty_sources_rejected(self):
        with pytest.raises(ValueError, match="at least one source"):
            ScheduledBlendDataset([])

    def test_non_three_tuple_rejected(self):
        with pytest.raises(ValueError, match="3-tuple"):
            ScheduledBlendDataset([(_StubDataset("A"), _w(1.0))])  # 2-tuple

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ScheduledBlendDataset([(_StubDataset("A"), _w(1.0), "")])

    def test_whitespace_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ScheduledBlendDataset([(_StubDataset("A"), _w(1.0), "   ")])

    def test_non_string_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ScheduledBlendDataset([(_StubDataset("A"), _w(1.0), 0)])

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate source name"):
            ScheduledBlendDataset([
                (_StubDataset("A"), _w(1.0), "src"),
                (_StubDataset("B"), _w(1.0), "src"),
            ])

    def test_non_scalar_weight_rejected(self):
        with pytest.raises(ValueError, match="1-element"):
            bad = torch.zeros(3)
            bad.share_memory_()
            ScheduledBlendDataset([(_StubDataset("A"), bad, "src")])

    def test_zero_weight_fallback_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="zero_weight_fallback"):
            ScheduledBlendDataset(
                [(_StubDataset("A"), _w(1.0), "src")],
                zero_weight_fallback=5,
            )

    def test_weights_are_shared_memory(self):
        # Even if caller forgot to share_memory_, constructor does.
        t = torch.tensor([1.0], dtype=torch.float64)
        assert not t.is_shared()
        ScheduledBlendDataset([(_StubDataset("A"), t, "src")])
        assert t.is_shared()


# ----------------------------------------------------------------------
# Introspection: num_sources, source_names, resolve
# ----------------------------------------------------------------------


class TestIntrospection:
    def test_num_sources(self):
        ds = _mk(("A", 1.0, 10), ("B", 1.0, 10), ("C", 1.0, 10))
        assert ds.num_sources == 3

    def test_source_names_returns_copy(self):
        ds = _mk(("A", 1.0, 10), ("B", 1.0, 10))
        names = ds.source_names
        names.append("X")
        # Mutating the returned list must not affect internal state.
        assert ds.source_names == ["A", "B"]

    def test_resolve_int_idx(self):
        ds = _mk(("A", 1.0, 10), ("B", 1.0, 10))
        assert ds.resolve(0) == 0
        assert ds.resolve(1) == 1

    def test_resolve_str_name(self):
        ds = _mk(("foo", 1.0, 10), ("bar", 1.0, 10))
        assert ds.resolve("foo") == 0
        assert ds.resolve("bar") == 1

    def test_resolve_unknown_name_raises_key_error(self):
        ds = _mk(("A", 1.0, 10))
        with pytest.raises(KeyError, match="unknown source name"):
            ds.resolve("missing")

    def test_resolve_out_of_range_int_raises_index_error(self):
        ds = _mk(("A", 1.0, 10))
        with pytest.raises(IndexError, match="out of range"):
            ds.resolve(5)
        with pytest.raises(IndexError, match="out of range"):
            ds.resolve(-1)

    def test_resolve_bool_rejected(self):
        # bool is a subclass of int but should not be silently accepted.
        ds = _mk(("A", 1.0, 10), ("B", 1.0, 10))
        with pytest.raises(TypeError, match="bool"):
            ds.resolve(True)

    def test_resolve_wrong_type_raises_type_error(self):
        ds = _mk(("A", 1.0, 10))
        with pytest.raises(TypeError, match="must be int .* or str"):
            ds.resolve(1.5)


# ----------------------------------------------------------------------
# Weight mutation
# ----------------------------------------------------------------------


class TestWeightMutation:
    def test_get_weights_initial(self):
        ds = _mk(("A", 0.7, 10), ("B", 0.3, 10))
        assert ds.get_weights() == [0.7, 0.3]

    def test_set_weight_by_idx(self):
        ds = _mk(("A", 1.0, 10), ("B", 0.0, 10))
        ds.set_weight(1, 0.5)
        assert ds.get_weight(1) == 0.5
        assert ds.get_weight(0) == 1.0

    def test_set_weight_by_name(self):
        ds = _mk(("foo", 1.0, 10), ("bar", 0.0, 10))
        ds.set_weight("bar", 0.5)
        assert ds.get_weight("bar") == 0.5

    def test_set_weight_unknown_name_raises_key_error(self):
        ds = _mk(("A", 1.0, 10))
        with pytest.raises(KeyError, match="unknown source name"):
            ds.set_weight("missing", 0.5)

    def test_set_weight_out_of_range_idx_raises_index_error(self):
        ds = _mk(("A", 1.0, 10))
        with pytest.raises(IndexError, match="out of range"):
            ds.set_weight(99, 0.5)

    def test_weight_change_visible_mid_iteration(self):
        ds = _mk(("A", 1.0, 10), ("B", 0.0, 10))
        # Initially picks A.
        assert ds[0]["tag"] == "A"
        # Flip.
        ds.set_weight("A", 0.0)
        ds.set_weight("B", 1.0)
        for _ in range(20):
            assert ds[0]["tag"] == "B"


# ----------------------------------------------------------------------
# Length / virtual length
# ----------------------------------------------------------------------


class TestLength:
    def test_default_len_is_first_source(self):
        ds = _mk(("A", 1.0, 13), ("B", 1.0, 99))
        assert len(ds) == 13

    def test_virtual_length_override(self):
        ds = ScheduledBlendDataset(
            [(_StubDataset("A", n=10), _w(1.0), "A")],
            virtual_length=1000,
        )
        assert len(ds) == 1000


# ----------------------------------------------------------------------
# Sampling behavior
# ----------------------------------------------------------------------


class TestSampling:
    def test_single_hot_weight_picks_that_source(self):
        ds = _mk(("A", 1.0, 10), ("B", 0.0, 10), ("C", 0.0, 10))
        for _ in range(30):
            sample = ds[0]
            assert sample["tag"] == "A"
            assert int(sample["source_idx"].item()) == 0

    def test_zero_weight_fallback_default_first(self):
        ds = _mk(("A", 0.0, 10), ("B", 0.0, 10))
        # All zero → fall back to index 0.
        for _ in range(10):
            sample = ds[3]
            assert sample["tag"] == "A"
            assert int(sample["source_idx"].item()) == 0

    def test_zero_weight_fallback_configurable(self):
        ds = ScheduledBlendDataset(
            [
                (_StubDataset("A"), _w(0.0), "A"),
                (_StubDataset("B"), _w(0.0), "B"),
            ],
            zero_weight_fallback=1,
        )
        for _ in range(10):
            assert ds[0]["tag"] == "B"

    def test_negative_weight_clamped_to_zero(self):
        ds = _mk(("A", -1.0, 10), ("B", 1.0, 10))
        for _ in range(20):
            assert ds[0]["tag"] == "B"

    def test_idx_modulo_per_source(self):
        # Outer idx larger than B's length should still pick valid B item.
        ds = _mk(("A", 0.0, 100), ("B", 1.0, 5))
        item = ds[42]
        assert item["tag"] == "B"
        assert 0 <= item["idx"] < 5

    def test_balanced_distribution_chi_square(self):
        """Chi-square test that empirical distribution matches uniform."""
        ds = _mk(("A", 1.0, 10), ("B", 1.0, 10), ("C", 1.0, 10))
        _random.seed(0)
        N = 6000
        counts = [0, 0, 0]
        for _ in range(N):
            counts[int(ds[0]["source_idx"].item())] += 1
        expected = N / 3
        # Pearson chi-square; df=2; critical value at α=0.01 ≈ 9.21.
        chi2 = sum((c - expected) ** 2 / expected for c in counts)
        assert chi2 < 9.21, f"counts={counts}, chi2={chi2:.3f}"

    def test_skewed_distribution_chi_square(self):
        """Verify a 70/30 split is realised."""
        ds = _mk(("A", 0.7, 10), ("B", 0.3, 10))
        _random.seed(0)
        N = 5000
        counts = [0, 0]
        for _ in range(N):
            counts[int(ds[0]["source_idx"].item())] += 1
        expected = [0.7 * N, 0.3 * N]
        chi2 = sum(
            (c - e) ** 2 / e for c, e in zip(counts, expected)
        )
        # df=1, α=0.01 critical ≈ 6.63
        assert chi2 < 6.63, f"counts={counts}, chi2={chi2:.3f}"

    def test_source_idx_dtype_int32(self):
        ds = _mk(("A", 1.0, 10))
        sample = ds[0]
        assert sample["source_idx"].dtype == torch.int32
        assert sample["source_idx"].shape == ()  # scalar

    def test_source_idx_overrides_inner_value(self):
        """If inner source emits source_idx, the outer mixer overwrites it."""

        class _Polluter(Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx):
                return {"tag": "X", "source_idx": torch.tensor(99, dtype=torch.int32)}

        ds = ScheduledBlendDataset(
            [(_Polluter(), _w(1.0), "real")]
        )
        sample = ds[0]
        assert int(sample["source_idx"].item()) == 0  # not 99

    def test_top_level_dict_is_fresh_copy(self):
        """Adding source_idx must not mutate any cache the inner returns."""
        cache = {"tag": "A", "idx": 0}

        class _Cached(Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx):
                return cache  # same dict every time

        ds = ScheduledBlendDataset([(_Cached(), _w(1.0), "src")])
        ds[0]
        # Inner's cache must NOT contain source_idx now.
        assert "source_idx" not in cache
