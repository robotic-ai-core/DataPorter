"""Round-trip tests for ParquetTokenDataset.

The acceptance test for Phase 2 migration: write a tiny Parquet shard,
read it back via ParquetTokenDataset, verify samples match what we wrote
in both preload and streaming modes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataporter.text import ParquetTokenDataset


SEQ_LEN = 16


def _write_shard(path: Path, rows: np.ndarray) -> None:
    """Write a uint16 [n_rows, seq_len] array to a Parquet shard.

    Mirrors the layout ParquetTokenDataset expects: list<uint16>
    column named ``input_ids``.
    """
    n_rows, seq_len = rows.shape
    list_of_lists = [r.tolist() for r in rows]
    table = pa.Table.from_pydict(
        {"input_ids": list_of_lists},
        schema=pa.schema([("input_ids", pa.list_(pa.uint16(), seq_len))]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


@pytest.fixture
def shard_dir(tmp_path):
    """A directory with two shards in ``train/`` and one in ``val/``."""
    root = tmp_path / "ds"
    shard_a = np.arange(0, 10 * SEQ_LEN, dtype=np.uint16).reshape(10, SEQ_LEN)
    shard_b = np.arange(10 * SEQ_LEN, 25 * SEQ_LEN, dtype=np.uint16).reshape(15, SEQ_LEN)
    val_shard = np.arange(50 * SEQ_LEN, 53 * SEQ_LEN, dtype=np.uint16).reshape(3, SEQ_LEN)
    _write_shard(root / "train" / "shard_0000.parquet", shard_a)
    _write_shard(root / "train" / "shard_0001.parquet", shard_b)
    _write_shard(root / "val" / "shard_0000.parquet", val_shard)
    return root


# ----- preload mode ---------------------------------------------------------


class TestPreload:

    def test_preload_round_trip(self, shard_dir):
        ds = ParquetTokenDataset(shard_dir, split="train", preload=True)
        assert len(ds) == 25
        assert ds.seq_len == SEQ_LEN
        # Sample 0 should be the first row of shard_a.
        s = ds[0]
        assert s["input_ids"].dtype == torch.long
        assert s["input_ids"].tolist() == list(range(SEQ_LEN))
        # Last sample should be the final row of shard_b.
        s_last = ds[24]
        assert s_last["input_ids"].tolist() == list(
            range(24 * SEQ_LEN, 25 * SEQ_LEN),
        )

    def test_labels_are_independent_clone_of_input_ids(self, shard_dir):
        ds = ParquetTokenDataset(shard_dir, split="train", preload=True)
        s = ds[0]
        assert torch.equal(s["input_ids"], s["labels"])
        s["labels"][0] = 9999
        assert s["input_ids"][0] != 9999

    def test_out_of_bounds_raises(self, shard_dir):
        ds = ParquetTokenDataset(shard_dir, split="train", preload=True)
        with pytest.raises(IndexError):
            ds[-1]
        with pytest.raises(IndexError):
            ds[len(ds)]


# ----- streaming mode -------------------------------------------------------


class TestStreaming:

    def test_streaming_round_trip(self, shard_dir):
        ds = ParquetTokenDataset(shard_dir, split="train", preload=False)
        assert len(ds) == 25
        s0 = ds[0]
        assert s0["input_ids"].tolist() == list(range(SEQ_LEN))
        s_mid = ds[10]  # First row of shard_b
        assert s_mid["input_ids"].tolist() == list(
            range(10 * SEQ_LEN, 11 * SEQ_LEN),
        )

    def test_streaming_matches_preload(self, shard_dir):
        preloaded = ParquetTokenDataset(shard_dir, split="train", preload=True)
        streaming = ParquetTokenDataset(shard_dir, split="train", preload=False)
        assert len(preloaded) == len(streaming)
        for i in range(len(preloaded)):
            assert torch.equal(
                preloaded[i]["input_ids"], streaming[i]["input_ids"],
            )


# ----- discovery + errors ---------------------------------------------------


def test_val_split(shard_dir):
    ds = ParquetTokenDataset(shard_dir, split="val", preload=True)
    assert len(ds) == 3


def test_missing_split_raises(tmp_path):
    (tmp_path / "ds").mkdir()
    with pytest.raises(FileNotFoundError, match="Split directory not found"):
        ParquetTokenDataset(tmp_path / "ds", split="train")


def test_empty_split_raises(tmp_path):
    (tmp_path / "ds" / "train").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No shard files found"):
        ParquetTokenDataset(tmp_path / "ds", split="train")


def test_dataset_name(shard_dir):
    ds = ParquetTokenDataset(shard_dir, split="train", preload=True)
    assert ds.dataset_name == shard_dir.name
