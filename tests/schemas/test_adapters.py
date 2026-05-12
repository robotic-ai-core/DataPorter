"""Unit tests for data adapter primitives and the three source recipes.

Covers:
  - Each adapter primitive in isolation.
  - The end-to-end pretrain_pad chain handling a TokenShuffleBufferDataset-shaped
    input (extra keys, no labels, bool-compatible mask).
  - Recipe outputs validate against SampleSpec.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.schemas import (
    AddCausalLabels,
    DropExtras,
    EnsureLossMask,
    SchemaError,
    StampSourceTag,
    TextSampleSpec,
    ValidateSpec,
    chat_query_adapter,
    pretrain_pad_adapter,
    val_full_adapter,
)


SEQ_LEN = 8
PAD_ID = 0
VOCAB = 100


def _spec(pad_id: int = PAD_ID) -> TextSampleSpec:
    return TextSampleSpec(
        pad_token_id=pad_id,
        seq_len=SEQ_LEN,
        tokenizer_id="test-tokenizer",
        vocab_size=VOCAB,
    )


class _FixedDataset(Dataset):
    def __init__(self, sample):
        self._sample = sample

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        # Return a fresh copy each time so mutations don't leak across tests.
        return {k: v.clone() if torch.is_tensor(v) else v
                for k, v in self._sample.items()}


def _buffer_shaped_sample(pad_count: int = 3):
    """Mimics TokenShuffleBufferDataset output: {input_ids, loss_mask, length, key}."""
    real = SEQ_LEN - pad_count
    ids = torch.cat([
        torch.randint(1, VOCAB, (real,), dtype=torch.long),
        torch.full((pad_count,), PAD_ID, dtype=torch.long),
    ])
    mask = (ids != PAD_ID).to(torch.uint8)  # DataPorter emits uint8
    return {
        "input_ids": ids,
        "loss_mask": mask,
        "length": torch.tensor(real, dtype=torch.int32),
        "key": "doc-123",
    }


def _parquet_shaped_sample():
    """Mimics ParquetTokenDataset output: {input_ids, labels}."""
    ids = torch.randint(1, VOCAB, (SEQ_LEN,), dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


def _chat_shaped_sample(query_len: int = 3):
    """Mimics ChatDataset output: {input_ids, labels, loss_mask}."""
    ids = torch.randint(1, VOCAB, (SEQ_LEN,), dtype=torch.long)
    mask = torch.zeros(SEQ_LEN, dtype=torch.bool)
    mask[query_len:] = True
    return {"input_ids": ids, "labels": ids.clone(), "loss_mask": mask}


# ----- DropExtras ------------------------------------------------------------


def test_drop_extras_removes_unknown_keys():
    ds = DropExtras(_FixedDataset(_buffer_shaped_sample()))
    sample = ds[0]
    assert set(sample.keys()) == {"input_ids", "loss_mask"}
    assert "length" not in sample
    assert "key" not in sample


def test_drop_extras_preserves_default_keys():
    full = {
        "input_ids": torch.zeros(SEQ_LEN, dtype=torch.long),
        "labels": torch.zeros(SEQ_LEN, dtype=torch.long),
        "loss_mask": torch.ones(SEQ_LEN, dtype=torch.bool),
        "source_tag": "pretrain_pad",
        "extra": "junk",
    }
    ds = DropExtras(_FixedDataset(full))
    sample = ds[0]
    assert set(sample.keys()) == {"input_ids", "labels", "loss_mask", "source_tag"}


def test_drop_extras_custom_keep():
    ds = DropExtras(
        _FixedDataset(_buffer_shaped_sample()),
        keep={"input_ids", "length"},
    )
    sample = ds[0]
    assert set(sample.keys()) == {"input_ids", "length"}


# ----- AddCausalLabels ------------------------------------------------------


def test_add_causal_labels_adds_when_missing():
    ds = AddCausalLabels(_FixedDataset(_buffer_shaped_sample()))
    sample = ds[0]
    assert "labels" in sample
    assert torch.equal(sample["labels"], sample["input_ids"])
    # Cloned, not shared.
    sample["labels"][0] = 99
    assert sample["input_ids"][0] != 99


def test_add_causal_labels_noop_when_present():
    parquet_sample = _parquet_shaped_sample()
    original_labels = parquet_sample["labels"].clone()
    ds = AddCausalLabels(_FixedDataset(parquet_sample))
    sample = ds[0]
    assert torch.equal(sample["labels"], original_labels)


# ----- EnsureLossMask -------------------------------------------------------


def test_ensure_loss_mask_synthesizes_from_pad_id():
    parquet_sample = _parquet_shaped_sample()
    # Inject some pad_id tokens
    parquet_sample["input_ids"][-2:] = PAD_ID
    ds = EnsureLossMask(_FixedDataset(parquet_sample), pad_token_id=PAD_ID)
    sample = ds[0]
    assert sample["loss_mask"].dtype == torch.bool
    expected = sample["input_ids"] != PAD_ID
    assert torch.equal(sample["loss_mask"], expected)
    assert not sample["loss_mask"][-2:].any()


def test_ensure_loss_mask_all_true_when_pad_id_none():
    parquet_sample = _parquet_shaped_sample()
    ds = EnsureLossMask(_FixedDataset(parquet_sample), pad_token_id=None)
    sample = ds[0]
    assert sample["loss_mask"].dtype == torch.bool
    assert sample["loss_mask"].all()


def test_ensure_loss_mask_casts_uint8_to_bool():
    """DataPorter's TokenShuffleBufferDataset emits uint8; must become bool."""
    ds = EnsureLossMask(_FixedDataset(_buffer_shaped_sample()), pad_token_id=PAD_ID)
    sample = ds[0]
    assert sample["loss_mask"].dtype == torch.bool


def test_ensure_loss_mask_noop_when_bool_present():
    chat = _chat_shaped_sample()
    original_mask = chat["loss_mask"].clone()
    ds = EnsureLossMask(_FixedDataset(chat), pad_token_id=PAD_ID)
    sample = ds[0]
    assert torch.equal(sample["loss_mask"], original_mask)


# ----- StampSourceTag -------------------------------------------------------


def test_stamp_source_tag():
    ds = StampSourceTag(_FixedDataset(_parquet_shaped_sample()), tag="pretrain_pad")
    assert ds[0]["source_tag"] == "pretrain_pad"


def test_stamp_source_tag_overwrites_existing():
    sample = _parquet_shaped_sample()
    sample["source_tag"] = "wrong"
    ds = StampSourceTag(_FixedDataset(sample), tag="val_full")
    assert ds[0]["source_tag"] == "val_full"


# ----- ValidateSpec ---------------------------------------------------------


def test_validate_spec_passes_on_compliant():
    spec = _spec()
    ds = pretrain_pad_adapter(_FixedDataset(_buffer_shaped_sample()), spec)
    vs = ValidateSpec(ds, spec, source="pretrain_pad", first_n=3)
    for i in range(3):
        vs[i]  # should not raise


def test_validate_spec_raises_on_non_compliant():
    spec = _spec()
    # Non-compliant: all-True mask but sample has pads
    bad = _buffer_shaped_sample()
    bad["loss_mask"] = torch.ones(SEQ_LEN, dtype=torch.uint8)
    # Pass through a chain that keeps the bad mask
    ds = StampSourceTag(
        AddCausalLabels(DropExtras(_FixedDataset(bad))),
        tag="pretrain_pad",
    )
    # Cast uint8→bool manually since we skipped EnsureLossMask here
    class _CastBool(Dataset):
        def __init__(self, inner):
            self._inner = inner
        def __len__(self):
            return len(self._inner)
        def __getitem__(self, idx):
            s = dict(self._inner[idx])
            s["loss_mask"] = s["loss_mask"].bool()
            return s
    ds_bool = _CastBool(ds)
    vs = ValidateSpec(ds_bool, spec, source="pretrain_pad", first_n=1)
    with pytest.raises(SchemaError):
        vs[0]


# ----- end-to-end recipes ---------------------------------------------------


def test_pretrain_pad_recipe_makes_buffer_sample_compliant():
    """The exact bug scenario: DataPorter emits correct pad-mask; adapter chain
    must preserve it (not clobber to all-True) and validate clean."""
    spec = _spec()
    sample = _buffer_shaped_sample(pad_count=3)
    ds = pretrain_pad_adapter(_FixedDataset(sample), spec)
    got = ds[0]
    assert set(got.keys()) == {"input_ids", "labels", "loss_mask", "source_tag"}
    assert got["source_tag"] == "pretrain_pad"
    # Real mask preserved: False at pad positions
    assert not got["loss_mask"][-3:].any()
    assert got["loss_mask"][:-3].all()
    spec.validate(got, "pretrain_pad")


def test_pretrain_pad_recipe_handles_parquet_sample():
    """Parquet emits {input_ids, labels}; recipe must synthesize loss_mask from pad_id."""
    spec = _spec()
    parquet = _parquet_shaped_sample()
    parquet["input_ids"][-2:] = PAD_ID  # inject pads
    ds = pretrain_pad_adapter(_FixedDataset(parquet), spec)
    got = ds[0]
    spec.validate(got, "pretrain_pad")
    assert not got["loss_mask"][-2:].any()


def test_chat_query_recipe():
    spec = _spec()
    ds = chat_query_adapter(_FixedDataset(_chat_shaped_sample()), spec)
    got = ds[0]
    assert got["source_tag"] == "chat_query"
    spec.validate(got, "chat_query")


def test_val_full_recipe_on_parquet_sample():
    """Val data has no padding; recipe produces all-True mask regardless of tokens."""
    spec = _spec()
    parquet = _parquet_shaped_sample()
    # Even if EOS happens to equal pad_id mid-stream, val_full ignores pad_id
    parquet["input_ids"][3] = PAD_ID
    ds = val_full_adapter(_FixedDataset(parquet), spec)
    got = ds[0]
    assert got["source_tag"] == "val_full"
    assert got["loss_mask"].all()
    spec.validate(got, "val_full")


def test_recipes_preserve_dataset_length():
    spec = _spec()
    inner = _FixedDataset(_buffer_shaped_sample())
    assert len(pretrain_pad_adapter(inner, spec)) == len(inner)
    assert len(chat_query_adapter(inner, spec)) == len(inner)
    assert len(val_full_adapter(inner, spec)) == len(inner)
