"""Wrapper equivalence: the four BC classes preserve their old API and
emit DeprecationWarning on construction.

These tests do NOT compare against two independent implementations —
both code paths now route to ``ScheduledBlendDataset`` underneath. The
tests assert *externally-observable behavior* matches the legacy
contract (lengths, ratio setter effect, sampling tag distribution,
shared-memory tensor exposure), plus that DeprecationWarning fires.
"""

from __future__ import annotations

import random as _random
import warnings
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.text.blending import (
    BlendedTextDataset,
    MixingScheduleCallback,
    PretrainBlendScheduleCallback,
    ScheduledBlendDataset,
    WeightedMultiSourceDataset,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


class _StubDataset(Dataset):
    def __init__(self, tag: str, n: int = 10):
        self._tag = tag
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {"tag": self._tag, "idx": idx}


def _w(v: float) -> torch.Tensor:
    t = torch.tensor([v], dtype=torch.float64)
    t.share_memory_()
    return t


def _silence(category=DeprecationWarning):
    """Context-manager that filters a given warning category."""
    return warnings.catch_warnings()


# ----------------------------------------------------------------------
# DeprecationWarning emission — required for each wrapper
# ----------------------------------------------------------------------


class TestDeprecationWarnings:
    def test_weighted_multi_source_warns(self):
        with pytest.warns(DeprecationWarning, match="WeightedMultiSourceDataset is deprecated"):
            WeightedMultiSourceDataset(
                [(_StubDataset("A"), _w(1.0)), (_StubDataset("B"), _w(1.0))]
            )

    def test_blended_text_dataset_warns(self):
        with pytest.warns(DeprecationWarning, match="BlendedTextDataset is deprecated"):
            BlendedTextDataset(
                pretrain_dataset=_StubDataset("pretrain_pad"),
                chat_dataset=_StubDataset("chat_query"),
                sample_spec=None,
            )

    def test_mixing_schedule_callback_warns(self):
        with pytest.warns(DeprecationWarning, match="MixingScheduleCallback is deprecated"):
            MixingScheduleCallback(
                blend_start_step=0, blend_end_step=100, chat_ratio_end=0.5,
            )

    def test_pretrain_blend_schedule_callback_warns(self):
        with pytest.warns(DeprecationWarning, match="PretrainBlendScheduleCallback is deprecated"):
            PretrainBlendScheduleCallback(schedules=[
                {"source_idx": 0, "points": [{"step": 0, "weight": 1.0}]},
            ])


# ----------------------------------------------------------------------
# WeightedMultiSourceDataset wrapper
# ----------------------------------------------------------------------


# Suppress the deprecation noise for tests that intentionally construct
# the wrapper; the warning is already covered above.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning"
)


class TestWeightedMultiSourceWrapper:
    def test_old_2_tuple_signature(self):
        # No name required — the wrapper synthesises "source_{i}".
        ds = WeightedMultiSourceDataset(
            [(_StubDataset("A"), _w(1.0)), (_StubDataset("B"), _w(0.0))]
        )
        for _ in range(20):
            assert ds[0]["tag"] == "A"

    def test_weight_tensors_attribute_exposes_inner_tensors(self):
        ds = WeightedMultiSourceDataset(
            [(_StubDataset("A"), _w(1.0)), (_StubDataset("B"), _w(0.0))]
        )
        # Tests / debugger code pokes ds._weight_tensors[i] directly.
        ds._weight_tensors[0].fill_(0.0)
        ds._weight_tensors[1].fill_(1.0)
        for _ in range(20):
            assert ds[0]["tag"] == "B"

    def test_datasets_attribute_exposes_inner_datasets(self):
        ds = WeightedMultiSourceDataset(
            [(_StubDataset("A"), _w(1.0)), (_StubDataset("B"), _w(0.0))]
        )
        assert isinstance(ds._datasets[0], _StubDataset)

    def test_len_pinned_to_first(self):
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A", n=10), _w(1.0)),
                (_StubDataset("B", n=99), _w(1.0)),
            ]
        )
        assert len(ds) == 10

    def test_balanced_distribution_chi_square(self):
        ds = WeightedMultiSourceDataset(
            [(_StubDataset("A"), _w(1.0)), (_StubDataset("B"), _w(1.0))]
        )
        _random.seed(0)
        counts = {"A": 0, "B": 0}
        N = 4000
        for _ in range(N):
            counts[ds[0]["tag"]] += 1
        expected = N / 2
        chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
        # df=1, α=0.01 critical ≈ 6.63
        assert chi2 < 6.63

    def test_empty_sources_rejected(self):
        with pytest.raises(ValueError, match="at least one source"):
            WeightedMultiSourceDataset([])


# ----------------------------------------------------------------------
# BlendedTextDataset wrapper
# ----------------------------------------------------------------------


class TestBlendedTextDatasetWrapper:
    def test_chat_ratio_setter_getter(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        ds.chat_ratio = 0.3
        assert ds.chat_ratio == pytest.approx(0.3)

    def test_chat_ratio_clamps(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        ds.chat_ratio = -0.5
        assert ds.chat_ratio == 0.0
        ds.chat_ratio = 1.5
        assert ds.chat_ratio == 1.0

    def test_chat_ratio_t_is_shared_memory(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        assert ds._chat_ratio_t.is_shared()

    def test_ratio_zero_yields_pretrain(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        ds.chat_ratio = 0.0
        for _ in range(20):
            assert ds[0]["tag"] == "pretrain_pad"

    def test_ratio_one_yields_chat(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        ds.chat_ratio = 1.0
        for _ in range(20):
            assert ds[0]["tag"] == "chat_query"

    def test_ratio_thirty_percent_chi_square(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        ds.chat_ratio = 0.3
        _random.seed(0)
        N = 5000
        counts = {"pretrain_pad": 0, "chat_query": 0}
        for _ in range(N):
            counts[ds[0]["tag"]] += 1
        expected = {"pretrain_pad": 0.7 * N, "chat_query": 0.3 * N}
        chi2 = sum(
            (counts[k] - expected[k]) ** 2 / expected[k]
            for k in counts
        )
        # df=1, α=0.01 ≈ 6.63
        assert chi2 < 6.63, f"counts={counts}, chi2={chi2:.3f}"

    def test_len_is_pretrain_len(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad", n=20),
            chat_dataset=_StubDataset("chat_query", n=5),
        )
        assert len(ds) == 20

    def test_source_idx_present_on_samples(self):
        # New additive metadata: source_idx is stamped by the underlying
        # ScheduledBlendDataset even when accessed via the wrapper.
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        ds.chat_ratio = 0.0
        sample = ds[0]
        assert "source_idx" in sample
        assert sample["source_idx"].dtype == torch.int32
        assert int(sample["source_idx"].item()) == 0  # pretrain

        ds.chat_ratio = 1.0
        sample = ds[0]
        assert int(sample["source_idx"].item()) == 1  # chat


# ----------------------------------------------------------------------
# MixingScheduleCallback wrapper — drives BlendedTextDataset.chat_ratio
# ----------------------------------------------------------------------


class TestMixingScheduleCallbackWrapper:
    def test_writes_chat_ratio_on_batch_start(self):
        cb = MixingScheduleCallback(
            blend_start_step=10, blend_end_step=20, chat_ratio_end=1.0,
        )
        ds = BlendedTextDataset(
            pretrain_dataset=_StubDataset("pretrain_pad"),
            chat_dataset=_StubDataset("chat_query"),
        )
        trainer = MagicMock()
        trainer.global_step = 15  # halfway
        trainer.datamodule.blended_dataset = ds
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        assert ds.chat_ratio == pytest.approx(0.5)
        # And the underlying weights are [1-r, r].
        assert ds._inner.get_weights() == pytest.approx([0.5, 0.5])


# ----------------------------------------------------------------------
# PretrainBlendScheduleCallback wrapper — writes WeightedMultiSource
# weights directly via _weight_tensors[idx].fill_().
# ----------------------------------------------------------------------


class TestPretrainBlendScheduleCallbackWrapper:
    def test_writes_weights_at_simulated_step(self):
        cb = PretrainBlendScheduleCallback(schedules=[
            {"source_idx": 0, "points": [
                {"step": 0, "weight": 1.0},
                {"step": 100, "weight": 0.0},
            ]},
            {"source_idx": 1, "points": [
                {"step": 0, "weight": 0.0},
                {"step": 100, "weight": 1.0},
            ]},
        ])
        ds = WeightedMultiSourceDataset([
            (_StubDataset("A"), _w(1.0)),
            (_StubDataset("B"), _w(0.0)),
        ])
        trainer = MagicMock()
        trainer.datamodule._pretrain_multi_dataset = ds
        trainer.global_step = 50
        trainer.max_steps = 100
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        assert ds._weight_tensors[0].item() == pytest.approx(0.5)
        assert ds._weight_tensors[1].item() == pytest.approx(0.5)

    def test_invariant_preserved_under_wrapper(self):
        """End-to-end: callback drives wrapper which drives inner mixer,
        and the resulting sampling distribution matches expectations."""
        cb = PretrainBlendScheduleCallback(schedules=[
            {"source_idx": 0, "points": [{"step": 0, "weight": 0.2}]},
            {"source_idx": 1, "points": [{"step": 0, "weight": 0.8}]},
        ])
        ds = WeightedMultiSourceDataset([
            (_StubDataset("A"), _w(1.0)),
            (_StubDataset("B"), _w(0.0)),
        ])
        trainer = MagicMock()
        trainer.datamodule._pretrain_multi_dataset = ds
        trainer.global_step = 100
        trainer.max_steps = 200
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        # Now sample many times — should hit 80/20.
        _random.seed(0)
        counts = {"A": 0, "B": 0}
        N = 4000
        for _ in range(N):
            counts[ds[0]["tag"]] += 1
        chi2 = (
            (counts["A"] - 0.2 * N) ** 2 / (0.2 * N)
            + (counts["B"] - 0.8 * N) ** 2 / (0.8 * N)
        )
        # df=1, α=0.01 critical ≈ 6.63
        assert chi2 < 6.63, f"counts={counts}, chi2={chi2:.3f}"
