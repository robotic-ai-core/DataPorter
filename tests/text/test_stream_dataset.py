"""Tests for PretrainStreamDataset.

Ported from autofpv/data/tokenization/tests/test_stream_dataset.py.
Drops the autofpv.data.text_datamodule tests (those belong on the
autofpv side until Phase 4).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from dataporter.text.stream_dataset import PretrainStreamDataset, _to_tensors


class _FakeTokenizer:
    """Minimal tokenizer stub for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        # Each word becomes a token (1-indexed)
        return [hash(w) % 1000 + 1 for w in text.split()]

    @classmethod
    def from_pretrained(cls, name):
        return cls()


def _make_fake_dataset(n_docs: int = 100, words_per_doc: int = 50):
    """Create a fake HF-style streaming dataset."""
    docs = []
    for i in range(n_docs):
        text = " ".join(f"word{j}" for j in range(words_per_doc))
        docs.append({"text": text})
    return docs


class TestToTensors:
    def test_returns_expected_keys(self):
        chunk = np.array([1, 2, 3, 4], dtype=np.uint16)
        result = _to_tensors(chunk)
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert torch.equal(result["input_ids"], result["labels"])

    def test_shape_preserved(self):
        chunk = np.zeros(512, dtype=np.uint16)
        result = _to_tensors(chunk)
        assert result["input_ids"].shape == (512,)


class TestPretrainStreamDataset:
    def test_init(self):
        ds = PretrainStreamDataset(
            dataset="test/dataset",
            seq_len=32,
        )
        assert ds.seq_len == 32

    def test_set_epoch(self):
        ds = PretrainStreamDataset(dataset="test/dataset")
        ds.set_epoch(5)
        assert ds._epoch == 5

    @patch("transformers.AutoTokenizer")
    @patch("datasets.load_dataset")
    def test_yields_correct_format(self, mock_load, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = _FakeTokenizer()
        mock_load.return_value = _make_fake_dataset(n_docs=50, words_per_doc=100)

        ds = PretrainStreamDataset(
            dataset="test/dataset",
            seq_len=32,
            shuffle_buffer_size=10,
            max_docs=50,
            tokenizer_batch_size=10,
        )

        batches = list(ds)
        assert len(batches) > 0

        for batch in batches:
            assert "input_ids" in batch
            assert "labels" in batch
            assert batch["input_ids"].shape == (32,)
            assert batch["input_ids"].dtype == torch.long

    @patch("transformers.AutoTokenizer")
    @patch("datasets.load_dataset")
    def test_max_docs_limits_output(self, mock_load, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = _FakeTokenizer()
        # Provide 1000 docs but cap at 50
        mock_load.return_value = _make_fake_dataset(n_docs=1000, words_per_doc=100)

        ds_capped = PretrainStreamDataset(
            dataset="test/dataset",
            seq_len=32,
            shuffle_buffer_size=10,
            max_docs=50,
            tokenizer_batch_size=10,
        )
        chunks_capped = list(ds_capped)

        mock_load.return_value = _make_fake_dataset(n_docs=1000, words_per_doc=100)
        ds_full = PretrainStreamDataset(
            dataset="test/dataset",
            seq_len=32,
            shuffle_buffer_size=10,
            max_docs=1000,
            tokenizer_batch_size=10,
        )
        chunks_full = list(ds_full)

        assert len(chunks_capped) < len(chunks_full)

    @patch("transformers.AutoTokenizer")
    @patch("datasets.load_dataset")
    def test_deterministic_with_same_seed(self, mock_load, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = _FakeTokenizer()

        def make_ds():
            mock_load.return_value = _make_fake_dataset(n_docs=30, words_per_doc=50)
            return PretrainStreamDataset(
                dataset="test/dataset",
                seq_len=16,
                shuffle_buffer_size=5,
                max_docs=30,
                tokenizer_batch_size=5,
                seed=42,
            )

        a = [b["input_ids"].tolist() for b in make_ds()]
        b = [b["input_ids"].tolist() for b in make_ds()]
        assert a == b

    @patch("transformers.AutoTokenizer")
    @patch("datasets.load_dataset")
    def test_empty_docs_skipped(self, mock_load, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = _FakeTokenizer()
        mock_load.return_value = [
            {"text": ""},
            {"text": "   "},
            {"text": "hello world " * 50},
        ]

        ds = PretrainStreamDataset(
            dataset="test/dataset",
            seq_len=16,
            shuffle_buffer_size=5,
            max_docs=10,
            tokenizer_batch_size=1,
        )
        chunks = list(ds)
        assert len(chunks) > 0  # Only the non-empty doc produces chunks
