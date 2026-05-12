"""Unit tests for the generic Schema framework.

Covers FieldSpec dtype/shape resolution, Schema.validate behavior with
REQUIRED_KEYS + FIELDS + source-tag matching, probe_dataset /
probe_dataloader, and the _ValidatingLoader wrapper's first-batch
guarantee + every-N cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from dataporter.schemas import FieldSpec, Schema, SchemaError


# ----- FieldSpec ------------------------------------------------------------


@dataclass(frozen=True)
class _ToySchema(Schema):
    """Toy schema with a per-instance dimension name."""

    rows: int

    REQUIRED_KEYS: ClassVar = ("x",)
    FIELDS: ClassVar = {
        "x": FieldSpec(dtype=torch.float32, shape=("rows", 3)),
    }


def test_field_spec_validates_dtype():
    spec = _ToySchema(rows=2)
    bad = {"x": torch.zeros(2, 3, dtype=torch.float64)}
    with pytest.raises(SchemaError, match="dtype"):
        spec.validate(bad)


def test_field_spec_validates_shape_via_named_dim():
    spec = _ToySchema(rows=2)
    bad = {"x": torch.zeros(5, 3, dtype=torch.float32)}
    with pytest.raises(SchemaError, match="dim 0"):
        spec.validate(bad)


def test_field_spec_wildcard_dim_passes():
    @dataclass(frozen=True)
    class WildSchema(Schema):
        REQUIRED_KEYS: ClassVar = ("x",)
        FIELDS: ClassVar = {"x": FieldSpec(dtype=None, shape=(None, 3))}

    s = WildSchema()
    s.validate({"x": torch.zeros(11, 3)})
    s.validate({"x": torch.zeros(99, 3, dtype=torch.float64)})  # any dtype
    with pytest.raises(SchemaError, match="dim 1"):
        s.validate({"x": torch.zeros(11, 7)})


def test_field_spec_rejects_non_tensor():
    spec = _ToySchema(rows=2)
    with pytest.raises(SchemaError, match="must be a torch.Tensor"):
        spec.validate({"x": [[0, 0, 0], [0, 0, 0]]})


def test_field_spec_rejects_wrong_rank():
    spec = _ToySchema(rows=2)
    with pytest.raises(SchemaError, match="2D shape"):
        spec.validate({"x": torch.zeros(2, 3, 1, dtype=torch.float32)})


def test_field_spec_no_shape_skips_shape_check():
    @dataclass(frozen=True)
    class DtypeOnly(Schema):
        REQUIRED_KEYS: ClassVar = ("x",)
        FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long, shape=None)}

    s = DtypeOnly()
    s.validate({"x": torch.zeros(99, 99, dtype=torch.long)})  # rank irrelevant
    with pytest.raises(SchemaError, match="dtype"):
        s.validate({"x": torch.zeros(1, dtype=torch.float32)})


# ----- Schema.validate ------------------------------------------------------


def test_validate_raises_on_missing_required():
    spec = _ToySchema(rows=2)
    with pytest.raises(SchemaError, match="missing required keys"):
        spec.validate({"y": torch.zeros(2, 3)})


def test_validate_skips_present_check_when_no_required_keys():
    @dataclass(frozen=True)
    class Lax(Schema):
        FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long)}

    Lax().validate({})  # empty data, no required keys, no FIELDS to check


def test_validate_source_tag_check_when_known_sources_set():
    @dataclass(frozen=True)
    class TaggedSchema(Schema):
        REQUIRED_KEYS: ClassVar = ("source_tag",)
        KNOWN_SOURCES: ClassVar = frozenset({"a", "b"})

    s = TaggedSchema()
    s.validate({"source_tag": "a"}, source="a")
    with pytest.raises(SchemaError, match="source_tag"):
        s.validate({"source_tag": "a"}, source="b")
    with pytest.raises(SchemaError, match="unknown source"):
        s.validate({"source_tag": "a"}, source="c")


def test_validate_runs_invariants():
    calls = []

    def _check(data, schema):
        calls.append(data["x"].sum().item())

    @dataclass(frozen=True)
    class InvSchema(Schema):
        REQUIRED_KEYS: ClassVar = ("x",)
        KNOWN_SOURCES: ClassVar = frozenset({"s"})
        FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long)}
        _INVARIANTS: ClassVar = {"s": _check}

    s = InvSchema()
    s.validate({"x": torch.tensor([1, 2, 3])}, source="s")
    assert calls == [6]


# ----- probe_dataset --------------------------------------------------------


class _ListDataset(Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def test_probe_dataset_validates_first_n():
    spec = _ToySchema(rows=2)
    samples = [{"x": torch.zeros(2, 3, dtype=torch.float32)} for _ in range(4)]
    spec.probe_dataset(_ListDataset(samples), n=4)


def test_probe_dataset_reports_index_on_violation():
    spec = _ToySchema(rows=2)
    samples = [{"x": torch.zeros(2, 3, dtype=torch.float32)} for _ in range(4)]
    samples[2]["x"] = torch.zeros(2, 3, dtype=torch.float64)  # wrong dtype
    with pytest.raises(SchemaError, match=r"dataset\[2\]"):
        spec.probe_dataset(_ListDataset(samples), n=4)


# ----- probe_dataloader -----------------------------------------------------


def test_probe_dataloader_validates_first_n_batches():
    spec = _ToySchema(rows=2)
    ds = _ListDataset([
        {"x": torch.zeros(2, 3, dtype=torch.float32)} for _ in range(6)
    ])
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    # Default n=1 — just check the first batch.
    spec_with_batch_rows = _ToySchema(rows=2)
    # After batch_size=2 collate, x.shape = (2, 2, 3) — rank 3, not 2.
    # Validate via a batch-aware spec instead.

    @dataclass(frozen=True)
    class BatchToy(Schema):
        rows: int
        REQUIRED_KEYS: ClassVar = ("x",)
        FIELDS: ClassVar = {
            "x": FieldSpec(dtype=torch.float32, shape=(None, "rows", 3)),
        }

    batch_spec = BatchToy(rows=2)
    batch_spec.probe_dataloader(loader, n=2)


def test_probe_dataloader_reports_batch_index():
    @dataclass(frozen=True)
    class BatchToy(Schema):
        REQUIRED_KEYS: ClassVar = ("x",)
        FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long)}

    samples = [{"x": torch.tensor([1, 2, 3])} for _ in range(4)]
    samples[1]["x"] = torch.tensor([1.0, 2.0, 3.0])  # wrong dtype
    loader = DataLoader(_ListDataset(samples), batch_size=1, shuffle=False)
    spec = BatchToy()
    with pytest.raises(SchemaError, match="batch 1"):
        spec.probe_dataloader(loader, n=2)


# ----- _ValidatingLoader wrapper -------------------------------------------


def test_wrap_dataloader_validates_first_batch_always():
    @dataclass(frozen=True)
    class BatchToy(Schema):
        REQUIRED_KEYS: ClassVar = ("x",)
        FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long)}

    # First batch is non-compliant.
    bad = {"x": torch.tensor([1.0])}
    good = {"x": torch.tensor([1])}
    loader = DataLoader(
        _ListDataset([bad, good, good]), batch_size=1, shuffle=False,
    )
    wrapped = BatchToy().wrap_dataloader(loader)
    it = iter(wrapped)
    with pytest.raises(SchemaError, match="batch 1"):
        next(it)


def test_wrap_dataloader_skips_after_first_when_validate_every_none():
    @dataclass(frozen=True)
    class BatchToy(Schema):
        REQUIRED_KEYS: ClassVar = ("x",)
        FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long)}

    # First batch good, second batch bad → wrapper should NOT raise.
    good = {"x": torch.tensor([1])}
    bad = {"x": torch.tensor([1.0])}
    loader = DataLoader(
        _ListDataset([good, bad, bad]), batch_size=1, shuffle=False,
    )
    wrapped = BatchToy().wrap_dataloader(loader)
    out = list(wrapped)
    assert len(out) == 3  # no raise; bad batches passed through


def test_wrap_dataloader_validate_every_catches_later_batch():
    @dataclass(frozen=True)
    class BatchToy(Schema):
        REQUIRED_KEYS: ClassVar = ("x",)
        FIELDS: ClassVar = {"x": FieldSpec(dtype=torch.long)}

    good = {"x": torch.tensor([1])}
    bad = {"x": torch.tensor([1.0])}
    loader = DataLoader(
        _ListDataset([good, good, bad, good, good]),
        batch_size=1, shuffle=False,
    )
    wrapped = BatchToy().wrap_dataloader(loader, validate_every=3)
    # Batches 1 (always) + 3 (every 3rd) get validated; 3 is bad.
    with pytest.raises(SchemaError, match="batch 3"):
        list(wrapped)


def test_wrap_dataloader_forwards_len_and_attrs():
    @dataclass(frozen=True)
    class Pass(Schema):
        pass

    loader = DataLoader(_ListDataset([1, 2, 3]), batch_size=1)
    wrapped = Pass().wrap_dataloader(loader)
    assert len(wrapped) == len(loader)
    # The wrapped loader exposes the underlying DataLoader attrs.
    assert wrapped.batch_size == 1
