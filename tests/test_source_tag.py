"""source_tag emission contract tests.

Locks in the convention that BlendedLeRobotDataModule (and its
subclasses) emit a ``source_tag`` string per sample on both data paths
(streaming train + per-source val), aligned with the
``autofpv.data.blended_text_datamodule`` / ``SampleSpec`` convention.

What models can rely on:
- ``batch["source_tag"]`` exists on every sample as a Python string.
- The set of distinct values matches the user-declared source names
  (or last path component of ``repo_id`` when no name is given).
- ``datamodule.source_tag_to_idx`` provides the canonical
  string→int mapping in declaration order — useful when models need
  numeric indices for embedding lookup or per-source loss bucketing.

These tests cover the data-side contract.  The streaming-train path's
in-line ``item["source_tag"] = self._source_names[src_idx]`` is
exercised by ``test_lerobot_shuffle_buffer_dataset.py``; this file
focuses on the wrapping behavior, the property, and the allowlist.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.dataset_wrappers import SourceTagDataset


class _StubDataset:
    """Minimal map-style stub that __len__/__getitem__ correctly."""

    def __init__(self, n: int = 4):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {"x": torch.tensor([float(idx)])}


def _build_dm(sources: list[dict] | str, **overrides):
    """Construct a BlendedLeRobotDataModule without invoking setup()."""
    from dataporter import BlendedLeRobotDataModule

    base = dict(
        repo_id=sources,
        delta_timestamps={"observation.image": [0.0]},
    )
    base.update(overrides)
    return BlendedLeRobotDataModule(**base)


class TestSourceTagDataset:
    """The wrapper that stamps a fixed string onto every sample."""

    def test_stamps_tag_on_each_sample(self):
        ds = SourceTagDataset(_StubDataset(n=3), "lewm")
        for i in range(3):
            assert ds[i]["source_tag"] == "lewm"

    def test_overwrites_existing_tag(self):
        # Idempotent on re-wrap — outermost wins (matches user mental model).
        inner = SourceTagDataset(_StubDataset(n=2), "wrong")
        outer = SourceTagDataset(inner, "right")
        assert outer[0]["source_tag"] == "right"

    def test_string_type_not_tensor(self):
        # Strings collate to list[str], not stacked tensor — the model
        # must do its own str→int conversion via source_tag_to_idx.
        ds = SourceTagDataset(_StubDataset(), "v5")
        sample = ds[0]
        assert isinstance(sample["source_tag"], str)
        assert sample["source_tag"] == "v5"


class TestSourceTagToIdx:
    """The string→int mapping the model uses for embedding lookup."""

    def test_single_source(self):
        dm = _build_dm("local/lewm-pusht-96x96-full")
        # Default name = last path component
        assert dm.source_tag_to_idx == {"lewm-pusht-96x96-full": 0}

    def test_explicit_names_preserve_declaration_order(self):
        dm = _build_dm([
            {"repo_id": "neiltan/pusht-synthetic-v5", "name": "v5"},
            {"repo_id": "local/lewm-pusht-96x96-full", "name": "lewm"},
        ])
        # Indices follow declaration order, NOT lexical / hash.
        assert dm.source_tag_to_idx == {"v5": 0, "lewm": 1}

    def test_fallback_names_match_val_source_names(self):
        # When per_source_val=True, val_source_names returns the same
        # names that source_tag_to_idx keys on.  Models can use either
        # to find which dataloader_idx corresponds to which source.
        dm = _build_dm([
            {"repo_id": "neiltan/pusht-synthetic-v5"},
            {"repo_id": "local/lewm-pusht-96x96-full"},
        ], per_source_val=True)
        # Stub the per-source val dict (setup not called, so nothing
        # else to populate).
        dm._per_source_val_datasets = {
            "pusht-synthetic-v5": _StubDataset(),
            "lewm-pusht-96x96-full": _StubDataset(),
        }
        assert list(dm.val_source_names) == list(dm.source_tag_to_idx.keys())

    def test_indices_dense_and_zero_based(self):
        dm = _build_dm([
            {"repo_id": "a/x"},
            {"repo_id": "b/y"},
            {"repo_id": "c/z"},
        ])
        idx = dm.source_tag_to_idx
        assert sorted(idx.values()) == [0, 1, 2]


class TestCommonSampleKeysIncludesSourceTag:
    """KeyFilterDataset must NOT strip source_tag from blended samples."""

    def test_source_tag_in_allowlist(self):
        dm = _build_dm([{"repo_id": "a/x"}, {"repo_id": "b/y"}])
        keys = dm._common_sample_keys()
        assert "source_tag" in keys

    def test_legacy_keys_still_present(self):
        # Refactor must not drop the existing bookkeeping fields that
        # the val/non-streaming paths' KeyFilterDataset already passes.
        dm = _build_dm([{"repo_id": "a/x"}, {"repo_id": "b/y"}])
        keys = dm._common_sample_keys()
        for legacy in (
            "episode_index", "frame_index", "timestamp",
            "index", "task_index", "observation.image",
        ):
            assert legacy in keys
