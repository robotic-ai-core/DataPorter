"""Per-source validation API tests.

Locks in the contract that callers can opt into per-source val
dataloaders via ``per_source_val=True`` on ``BlendedLeRobotDataModule``.
The model's ``validation_step`` then sees one batch at a time per
source (Lightning iterates each dataloader before moving to the next)
and can log ``val_<metric>/<source>`` keyed by
``datamodule.val_source_names[dataloader_idx]``.

The four cases covered:
  1. Single source, per_source_val=False → 1 loader, names=[<src>]
  2. Single source, per_source_val=True  → 1 loader, names=[<src>]
     (no point splitting one source — flag is a no-op here)
  3. Multi-source, per_source_val=False  → 1 ConcatDataset loader,
     names=["all"] (legacy behaviour)
  4. Multi-source, per_source_val=True   → list of N loaders, names
     aligned in declaration order

Plus the source-name normalization rule: explicit ``name`` field wins,
else fallback to last path component of ``repo_id``.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import ConcatDataset

from dataporter.dataset_wrappers import KeyFilterDataset


def _build_dm(
    sources: list[dict] | str, *, per_source_val: bool = False, **overrides,
):
    """Construct a BlendedLeRobotDataModule without invoking setup().

    setup() needs real LeRobot fixtures; for these contract tests we
    only exercise __init__ + val_dataloader's branching logic, which we
    can drive by directly populating ``_per_source_val_datasets``.
    """
    from dataporter import BlendedLeRobotDataModule

    base = dict(
        repo_id=sources,
        delta_timestamps={"observation.image": [0.0]},
        per_source_val=per_source_val,
    )
    base.update(overrides)
    return BlendedLeRobotDataModule(**base)


class _StubDataset:
    """Minimal map-style stub that __len__/__getitem__ correctly.

    ResumableDataLoader doesn't care about content for the API tests —
    it just needs a Sized + indexable Dataset.
    """

    def __init__(self, n: int = 4):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {"x": torch.tensor([float(idx)])}


class TestSourceNameNormalization:
    """The per-source label used as the metric suffix."""

    def test_explicit_name_field_wins(self):
        dm = _build_dm([
            {"repo_id": "neiltan/pusht-synthetic-v5", "name": "v5"},
        ])
        assert dm._sources[0]["name"] == "v5"

    def test_fallback_strips_namespace(self):
        dm = _build_dm([{"repo_id": "neiltan/pusht-synthetic-v5"}])
        assert dm._sources[0]["name"] == "pusht-synthetic-v5"

    def test_fallback_no_slash(self):
        # Edge case: a bare repo_id with no namespace.
        dm = _build_dm([{"repo_id": "lewm-pusht-96x96"}])
        assert dm._sources[0]["name"] == "lewm-pusht-96x96"

    def test_string_repo_id_normalizes(self):
        # When repo_id is a bare str (single-source convenience form),
        # the same naming rule applies.
        dm = _build_dm("local/lewm-pusht-96x96-full")
        assert dm._sources[0]["name"] == "lewm-pusht-96x96-full"


class TestValSourceNamesEmpty:
    """Before setup() runs, the per-source map is None."""

    def test_returns_empty_list(self):
        dm = _build_dm([{"repo_id": "a/b"}, {"repo_id": "c/d"}])
        assert dm.val_source_names == []


class TestValDataloaderBranching:
    """Drives val_dataloader by stuffing in stub per-source datasets."""

    def _setup_stub_val(self, dm, source_names: list[str]):
        """Mimic setup()'s effect on ``_per_source_val_datasets``."""
        dm._per_source_val_datasets = {
            name: _StubDataset() for name in source_names
        }
        # val_dataset is the legacy concat-across-sources view.
        ds_list = list(dm._per_source_val_datasets.values())
        dm.val_dataset = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)

    def test_single_source_default(self):
        dm = _build_dm("local/lewm-pusht-96x96-full")
        self._setup_stub_val(dm, ["lewm-pusht-96x96-full"])
        loader = dm.val_dataloader()
        assert not isinstance(loader, list), "single-source path returns one loader"
        assert dm.val_source_names == ["lewm-pusht-96x96-full"]

    def test_single_source_flag_is_noop(self):
        # per_source_val=True on a 1-source dm is a no-op: no point
        # splitting one source into multiple loaders.
        dm = _build_dm("local/lewm-pusht-96x96-full", per_source_val=True)
        self._setup_stub_val(dm, ["lewm-pusht-96x96-full"])
        loader = dm.val_dataloader()
        assert not isinstance(loader, list)
        assert dm.val_source_names == ["lewm-pusht-96x96-full"]

    def test_multi_source_default_concats(self):
        dm = _build_dm([
            {"repo_id": "local/lewm-pusht-96x96-full"},
            {"repo_id": "neiltan/pusht-synthetic-v5"},
        ])
        self._setup_stub_val(dm, ["lewm-pusht-96x96-full", "pusht-synthetic-v5"])
        loader = dm.val_dataloader()
        assert not isinstance(loader, list), (
            "default multi-source returns ONE loader over ConcatDataset"
        )
        # Legacy behaviour: a single label "all" so the model still has
        # something to suffix metrics with even without per-source split.
        assert dm.val_source_names == ["all"]

    def test_multi_source_per_source_val_returns_list(self):
        dm = _build_dm(
            [
                {"repo_id": "local/lewm-pusht-96x96-full"},
                {"repo_id": "neiltan/pusht-synthetic-v5"},
            ],
            per_source_val=True,
        )
        self._setup_stub_val(dm, ["lewm-pusht-96x96-full", "pusht-synthetic-v5"])
        loaders = dm.val_dataloader()
        assert isinstance(loaders, list), (
            "per_source_val=True with >1 source must return a list of loaders"
        )
        assert len(loaders) == 2
        # Names align with loaders index — Lightning's dataloader_idx
        # parameter in validation_step uses this exact ordering.
        assert dm.val_source_names == [
            "lewm-pusht-96x96-full", "pusht-synthetic-v5",
        ]

    def test_multi_source_per_source_val_preserves_declaration_order(self):
        dm = _build_dm(
            [
                {"repo_id": "neiltan/pusht-synthetic-v5", "name": "v5"},
                {"repo_id": "local/lewm-pusht-96x96-full", "name": "lewm"},
            ],
            per_source_val=True,
        )
        self._setup_stub_val(dm, ["v5", "lewm"])
        loaders = dm.val_dataloader()
        assert isinstance(loaders, list)
        assert dm.val_source_names == ["v5", "lewm"], (
            "per-source val ordering must follow the user's declaration "
            "order, not lexical / hash order"
        )
