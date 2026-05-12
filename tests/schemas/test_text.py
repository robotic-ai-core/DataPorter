"""Unit tests for TextSampleSpec — the text sample contract.

The canonical regression test here is ``test_pretrain_pad_catches_all_true_with_pads``
which encodes the historical blended-text bug (pretrain branch overwriting the
real pad-mask with all-True, making pad predictions count in training loss).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.schemas import SchemaError, TextSampleSpec


SEQ_LEN = 8
PAD_ID = 0
VOCAB = 100


def _mk_spec(pad_id: int = PAD_ID, seq_len: int = SEQ_LEN) -> TextSampleSpec:
    return TextSampleSpec(
        pad_token_id=pad_id,
        seq_len=seq_len,
        tokenizer_id="test-tokenizer",
        vocab_size=VOCAB,
    )


def _mk_pretrain_sample(pad_count: int = 3, seq_len: int = SEQ_LEN):
    real_count = seq_len - pad_count
    ids = torch.cat([
        torch.randint(1, VOCAB, (real_count,), dtype=torch.long),
        torch.full((pad_count,), PAD_ID, dtype=torch.long),
    ])
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "loss_mask": (ids != PAD_ID),
        "source_tag": "pretrain_pad",
    }


def _mk_chat_sample(query_len: int = 3, seq_len: int = SEQ_LEN):
    ids = torch.randint(1, VOCAB, (seq_len,), dtype=torch.long)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[query_len:] = True  # response-only loss
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "loss_mask": mask,
        "source_tag": "chat_query",
    }


def _mk_val_sample(seq_len: int = SEQ_LEN):
    ids = torch.randint(1, VOCAB, (seq_len,), dtype=torch.long)
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "loss_mask": torch.ones(seq_len, dtype=torch.bool),
        "source_tag": "val_full",
    }


# ----- happy-path validations -----------------------------------------------


def test_validate_pretrain_pad_ok():
    spec = _mk_spec()
    spec.validate(_mk_pretrain_sample(), "pretrain_pad")


def test_validate_chat_query_ok():
    spec = _mk_spec()
    spec.validate(_mk_chat_sample(), "chat_query")


def test_validate_val_full_ok():
    spec = _mk_spec()
    spec.validate(_mk_val_sample(), "val_full")


# ----- bug regression -------------------------------------------------------


def test_pretrain_pad_catches_all_true_with_pads():
    """The exact historical bug: pretrain sample with pads but all-True loss_mask."""
    spec = _mk_spec()
    sample = _mk_pretrain_sample(pad_count=3)
    sample["loss_mask"] = torch.ones(SEQ_LEN, dtype=torch.bool)  # clobber real mask
    with pytest.raises(SchemaError, match="loss_mask disagrees"):
        spec.validate(sample, "pretrain_pad")


def test_pretrain_pad_with_no_pads_all_true_is_valid():
    """Edge case: if sequence has no pads, real mask IS all-True — should pass."""
    spec = _mk_spec()
    sample = _mk_pretrain_sample(pad_count=0)
    assert sample["loss_mask"].all()
    spec.validate(sample, "pretrain_pad")


# ----- per-source invariants ------------------------------------------------


def test_chat_query_rejects_all_true_mask():
    spec = _mk_spec()
    sample = _mk_chat_sample()
    sample["loss_mask"] = torch.ones(SEQ_LEN, dtype=torch.bool)
    with pytest.raises(SchemaError, match="all-True"):
        spec.validate(sample, "chat_query")


def test_chat_query_rejects_all_false_mask():
    spec = _mk_spec()
    sample = _mk_chat_sample()
    sample["loss_mask"] = torch.zeros(SEQ_LEN, dtype=torch.bool)
    with pytest.raises(SchemaError, match="all-False"):
        spec.validate(sample, "chat_query")


def test_val_full_rejects_any_false():
    spec = _mk_spec()
    sample = _mk_val_sample()
    sample["loss_mask"][2] = False
    with pytest.raises(SchemaError, match="False entries"):
        spec.validate(sample, "val_full")


# ----- common violations ----------------------------------------------------


def test_unknown_source_raises():
    spec = _mk_spec()
    with pytest.raises(SchemaError, match="unknown source"):
        spec.validate(_mk_val_sample(), "bogus_source")


def test_missing_required_key_raises():
    spec = _mk_spec()
    sample = _mk_val_sample()
    del sample["loss_mask"]
    with pytest.raises(SchemaError, match="missing required keys"):
        spec.validate(sample, "val_full")


def test_source_tag_mismatch_raises():
    spec = _mk_spec()
    sample = _mk_pretrain_sample()
    sample["source_tag"] = "val_full"
    with pytest.raises(SchemaError, match="source_tag"):
        spec.validate(sample, "pretrain_pad")


def test_wrong_dtype_input_ids_raises():
    spec = _mk_spec()
    sample = _mk_val_sample()
    sample["input_ids"] = sample["input_ids"].float()
    with pytest.raises(SchemaError, match="dtype"):
        spec.validate(sample, "val_full")


def test_wrong_dtype_loss_mask_raises():
    spec = _mk_spec()
    sample = _mk_val_sample()
    sample["loss_mask"] = sample["loss_mask"].to(torch.uint8)
    with pytest.raises(SchemaError, match="dtype"):
        spec.validate(sample, "val_full")


def test_wrong_shape_raises():
    spec = _mk_spec()
    sample = _mk_val_sample()
    sample["input_ids"] = sample["input_ids"][:4]
    sample["labels"] = sample["labels"][:4]
    sample["loss_mask"] = sample["loss_mask"][:4]
    with pytest.raises(SchemaError, match="dim"):
        spec.validate(sample, "val_full")


# ----- probe_dataset --------------------------------------------------------


class _ListDataset(Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def test_probe_dataset_passes_compliant():
    spec = _mk_spec()
    ds = _ListDataset([_mk_pretrain_sample() for _ in range(16)])
    spec.probe_dataset(ds, "pretrain_pad", n=8)


def test_probe_dataset_reports_index_on_first_bad_sample():
    spec = _mk_spec()
    good = [_mk_pretrain_sample() for _ in range(4)]
    bad = _mk_pretrain_sample()
    bad["loss_mask"] = torch.ones(SEQ_LEN, dtype=torch.bool)  # the bug
    ds = _ListDataset(good + [bad] + good)
    with pytest.raises(SchemaError, match=r"dataset\[4\]"):
        spec.probe_dataset(ds, "pretrain_pad", n=8)


def test_probe_dataset_unknown_source_raises():
    spec = _mk_spec()
    ds = _ListDataset([_mk_val_sample()])
    with pytest.raises(SchemaError, match="unknown source"):
        spec.probe_dataset(ds, "nope", n=1)


# ----- from_tokenizer -------------------------------------------------------


class _MockTokenizer:
    def __init__(self, pad=0, eos=None, vocab=1000):
        self.pad_token_id = pad
        self.eos_token_id = eos
        self.vocab_size = vocab


def test_from_tokenizer_uses_pad_id():
    tok = _MockTokenizer(pad=7, eos=50256, vocab=8000)
    spec = TextSampleSpec.from_tokenizer(tok, seq_len=256, tokenizer_id="mock")
    assert spec.pad_token_id == 7
    assert spec.seq_len == 256
    assert spec.vocab_size == 8000


def test_from_tokenizer_falls_back_to_eos():
    tok = _MockTokenizer(pad=None, eos=50256)
    spec = TextSampleSpec.from_tokenizer(tok, seq_len=128, tokenizer_id="gpt2")
    assert spec.pad_token_id == 50256


def test_from_tokenizer_raises_when_both_missing():
    tok = _MockTokenizer(pad=None, eos=None)
    with pytest.raises(ValueError, match="pad_token_id"):
        TextSampleSpec.from_tokenizer(tok, seq_len=128, tokenizer_id="broken")
