"""End-to-end tests for TextProducerPool.

Uses a real spawned child, real parquet shards on disk, and a real
picklable tokenize_fn — no mocks beyond the toy tokenizer itself.
"""

from __future__ import annotations

import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataporter.text_producer_pool import (
    TextProducerConfig, TextProducerPool,
)
from dataporter.token_shuffle_buffer import TokenShuffleBuffer


# ---------------------------------------------------------------------------
# Picklable tokenize_fn (top-level — spawn pickling can't see closures)
# ---------------------------------------------------------------------------


class ToyTokenize:
    """Deterministic toy tokenizer: char code % vocab_size per character.

    Picklable (only stores ints).  Lazy-inits nothing — the real usage
    pattern is to build the HF tokenizer in ``__call__``'s first call.
    """
    def __init__(self, seq_len: int = 16, vocab_size: int = 256):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __call__(self, text: str):
        ids = [ord(c) % self.vocab_size for c in text[: self.seq_len]]
        if not ids:
            return None
        tokens = torch.tensor(ids, dtype=torch.int32)
        mask = torch.ones_like(tokens, dtype=torch.uint8)
        return tokens, mask


class AlwaysNoneTokenize:
    """Kernel that rejects every input — to drive smoke-test failure."""
    def __call__(self, text):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_shard(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def _seed_shards(shard_dir: Path, n_shards: int, rows_per_shard: int) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        texts = [
            f"doc {s}-{r} hello world tokens"
            for r in range(rows_per_shard)
        ]
        _write_shard(shard_dir / f"shard_{s:06d}.parquet", texts)


# ---------------------------------------------------------------------------
# Happy-path e2e
# ---------------------------------------------------------------------------


class TestTextProducerPoolHappyPath:
    def test_fills_buffer_end_to_end(self, tmp_path):
        _seed_shards(tmp_path, n_shards=3, rows_per_shard=50)

        buffer = TokenShuffleBuffer(
            capacity=32, seq_len=16, pad_token_id=0, vocab_size=256,
        )
        config = TextProducerConfig(
            source_name="toy",
            shard_dir=str(tmp_path),
            tokenize_fn=ToyTokenize(seq_len=16, vocab_size=256),
        )
        pool = TextProducerPool(buffer, config, warmup_target=16)
        pool.start()
        try:
            pool.wait_for_warmup(timeout=30.0)
            assert len(buffer) >= 16

            import random
            key, tokens, mask, length = buffer.sample(random.Random(0))
            assert length > 0
            assert tokens.dtype == torch.int32
            assert mask.dtype == torch.uint8
            # Toy tokenizer produces ord() % 256 — every id in range.
            assert int(tokens.max()) < 256
        finally:
            pool.stop()

    def test_buffer_saturates_at_capacity(self, tmp_path):
        _seed_shards(tmp_path, n_shards=2, rows_per_shard=100)

        buffer = TokenShuffleBuffer(capacity=8, seq_len=16, vocab_size=256)
        config = TextProducerConfig(
            source_name="cap",
            shard_dir=str(tmp_path),
            tokenize_fn=ToyTokenize(seq_len=16, vocab_size=256),
        )
        pool = TextProducerPool(buffer, config, warmup_target=8)
        pool.start()
        try:
            pool.wait_for_warmup(timeout=30.0)
            # Let the producer run a bit more; buffer should sit at
            # capacity (ring-buffer behaviour).
            time.sleep(0.5)
            assert len(buffer) == buffer.capacity
        finally:
            pool.stop()


# ---------------------------------------------------------------------------
# Fail-fast scenarios
# ---------------------------------------------------------------------------


class TestFailFast:
    def test_empty_shard_dir_times_out_cleanly(self, tmp_path):
        buffer = TokenShuffleBuffer(capacity=4, seq_len=16, vocab_size=256)
        config = TextProducerConfig(
            source_name="empty",
            shard_dir=str(tmp_path),   # no shards written
            tokenize_fn=ToyTokenize(),
            shard_rescan_s=0.5,
        )
        pool = TextProducerPool(buffer, config, warmup_target=4)
        pool.start()
        try:
            # Child exits after 120s "no shards" timeout but we don't wait
            # — just check that warmup times out cleanly (not a hang).
            with pytest.raises((TimeoutError, RuntimeError)):
                pool.wait_for_warmup(timeout=3.0)
        finally:
            pool.stop()

    def test_tokenize_always_none_reports_error(self, tmp_path):
        _seed_shards(tmp_path, n_shards=1, rows_per_shard=10)

        buffer = TokenShuffleBuffer(capacity=4, seq_len=16, vocab_size=256)
        config = TextProducerConfig(
            source_name="bad-kernel",
            shard_dir=str(tmp_path),
            tokenize_fn=AlwaysNoneTokenize(),
        )
        pool = TextProducerPool(buffer, config, warmup_target=4)
        pool.start()
        try:
            with pytest.raises(RuntimeError, match="smoke-test|child failed"):
                pool.wait_for_warmup(timeout=15.0)
        finally:
            pool.stop()


# ---------------------------------------------------------------------------
# Liveness / teardown
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_stop_terminates_child(self, tmp_path):
        _seed_shards(tmp_path, n_shards=1, rows_per_shard=20)

        buffer = TokenShuffleBuffer(capacity=8, seq_len=16, vocab_size=256)
        config = TextProducerConfig(
            source_name="stop",
            shard_dir=str(tmp_path),
            tokenize_fn=ToyTokenize(),
        )
        pool = TextProducerPool(buffer, config, warmup_target=4)
        pool.start()
        pool.wait_for_warmup(timeout=20.0)
        assert pool.is_alive

        pool.stop()
        assert not pool.is_alive

    def test_is_alive_false_before_start(self, tmp_path):
        buffer = TokenShuffleBuffer(capacity=4, seq_len=16, vocab_size=256)
        config = TextProducerConfig(
            source_name="unstarted",
            shard_dir=str(tmp_path),
            tokenize_fn=ToyTokenize(),
        )
        pool = TextProducerPool(buffer, config)
        assert not pool.is_alive
