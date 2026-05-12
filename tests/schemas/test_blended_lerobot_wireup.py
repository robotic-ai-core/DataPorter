"""Schema wire-up in BlendedLeRobotDataModule.

Constructor-level tests: schema_spec / schema_validate_every are
plumbed and ``_maybe_wrap_with_schema`` wraps loaders correctly.
End-to-end (setup → train_dataloader) is exercised by the existing
LeRobot integration tests once schema_spec is supplied in YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch
from torch.utils.data import DataLoader, Dataset


pytest.importorskip("lerobot")


from dataporter import BlendedLeRobotDataModule  # noqa: E402
from dataporter.schemas import FieldSpec, Schema, SchemaError  # noqa: E402


def _dm(**overrides):
    base = dict(
        repo_id="lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
    )
    base.update(overrides)
    return BlendedLeRobotDataModule(**base)


@dataclass(frozen=True)
class _TinySpec(Schema):
    REQUIRED_KEYS: ClassVar = ("x",)
    FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long)}


class _ListDS(Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        return self._samples[i]


# ----- constructor plumbing -------------------------------------------------


class TestSchemaPlumbing:

    def test_default_is_none(self):
        dm = _dm()
        assert dm._schema_spec is None
        assert dm._schema_validate_every is None

    def test_explicit_spec_stored(self):
        spec = _TinySpec()
        dm = _dm(schema_spec=spec)
        assert dm._schema_spec is spec

    def test_validate_every_stored(self):
        dm = _dm(schema_spec=_TinySpec(), schema_validate_every=10)
        assert dm._schema_validate_every == 10


# ----- _maybe_wrap_with_schema ---------------------------------------------


class TestMaybeWrap:

    def test_no_spec_passthrough(self):
        dm = _dm()
        loader = DataLoader(_ListDS([{"x": torch.tensor([1])}]), batch_size=1)
        assert dm._maybe_wrap_with_schema(loader) is loader

    def test_with_spec_wraps(self):
        dm = _dm(schema_spec=_TinySpec())
        loader = DataLoader(_ListDS([{"x": torch.tensor([1])}]), batch_size=1)
        wrapped = dm._maybe_wrap_with_schema(loader)
        assert wrapped is not loader
        # forwards underlying attributes
        assert wrapped.batch_size == 1
        assert len(wrapped) == len(loader)

    def test_wrapped_loader_validates_first_batch(self):
        dm = _dm(schema_spec=_TinySpec())
        bad = {"x": torch.tensor([1.0])}
        loader = DataLoader(_ListDS([bad]), batch_size=1)
        wrapped = dm._maybe_wrap_with_schema(loader)
        with pytest.raises(SchemaError, match="batch 1"):
            next(iter(wrapped))

    def test_wrapped_loader_validate_every_passthrough_unwatched_batches(self):
        dm = _dm(schema_spec=_TinySpec(), schema_validate_every=None)
        good = {"x": torch.tensor([1])}
        bad = {"x": torch.tensor([1.0])}
        loader = DataLoader(_ListDS([good, bad, bad]), batch_size=1)
        wrapped = dm._maybe_wrap_with_schema(loader)
        # validate_every=None → only first batch checked; later bad
        # batches must NOT raise (they pass through unwatched).
        out = list(wrapped)
        assert len(out) == 3
