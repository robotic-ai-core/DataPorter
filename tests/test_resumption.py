"""Tests for scientific resumption: ShardStorage state_dict, freeze/unfreeze, shard pinning."""

from __future__ import annotations

import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.storage import ShardStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_text_shard(path: Path, texts: list[str]):
    schema = pa.schema([("text", pa.string())])
    table = pa.table({"text": texts}, schema=schema)
    pq.write_table(table, str(path), compression="zstd")


def _write_shards(tmp_path, n=5, docs_per_shard=10):
    for i in range(n):
        _write_text_shard(
            tmp_path / f"shard_{i:06d}.parquet",
            [f"shard{i}_doc{j}" for j in range(docs_per_shard)],
        )


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_round_trip(self, tmp_path):
        """state_dict → load_state_dict preserves shard list."""
        _write_shards(tmp_path, n=5)
        s = ShardStorage(tmp_path, refresh_interval=60)
        assert len(s) == 50

        state = s.state_dict()
        assert len(state["shard_names"]) == 5
        assert state["total_rows"] == 50

        # Create a new storage and load the state
        s2 = ShardStorage(tmp_path, refresh_interval=60)
        s2.load_state_dict(state)
        assert len(s2) == 50
        assert s2.state_dict() == state

    def test_pins_shard_list(self, tmp_path):
        """After load_state_dict, only pinned shards are visible."""
        _write_shards(tmp_path, n=5)
        s = ShardStorage(tmp_path, refresh_interval=0.01)

        # Save state with 5 shards
        state = s.state_dict()

        # Write 3 more shards
        for i in range(5, 8):
            _write_text_shard(
                tmp_path / f"shard_{i:06d}.parquet",
                [f"new{i}_doc{j}" for j in range(10)],
            )

        # Load old state — should only see the original 5 shards
        s.load_state_dict(state)
        assert s.shard_count == 5
        assert len(s) == 50  # not 80

    def test_pinning_survives_refresh(self, tmp_path):
        """Pinned shard list is stable across refreshes."""
        _write_shards(tmp_path, n=3)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        state = s.state_dict()

        # Add more shards
        _write_text_shard(tmp_path / "shard_000003.parquet", ["new"] * 10)

        s.load_state_dict(state)
        assert s.shard_count == 3

        # Refresh should still only see pinned shards
        time.sleep(0.02)
        _ = len(s)
        assert s.shard_count == 3

    def test_unfreeze_after_load_allows_new_shards(self, tmp_path):
        """After unfreeze(), new shards become visible."""
        _write_shards(tmp_path, n=3)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        state = s.state_dict()

        _write_text_shard(tmp_path / "shard_000003.parquet", ["new"] * 10)

        s.load_state_dict(state)
        assert s.shard_count == 3

        s.unfreeze()
        time.sleep(0.02)
        _ = len(s)
        assert s.shard_count == 4

    def test_data_content_stable_after_resume(self, tmp_path):
        """Same index returns same data before and after state_dict round-trip."""
        _write_shards(tmp_path, n=3, docs_per_shard=10)
        s = ShardStorage(tmp_path, refresh_interval=60)

        # Read some items
        items_before = [s.get(i) for i in range(30)]
        state = s.state_dict()

        # Reload
        s2 = ShardStorage(tmp_path, refresh_interval=60)
        s2.load_state_dict(state)
        items_after = [s2.get(i) for i in range(30)]

        assert items_before == items_after


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

class TestFreezeUnfreeze:

    def test_freeze_blocks_eviction(self, tmp_path):
        """When frozen, scheduled evictions are deferred."""
        _write_shards(tmp_path, n=5)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        s.freeze()

        s.schedule_eviction(tmp_path / "shard_000000.parquet")
        time.sleep(0.02)
        _ = len(s)  # refresh

        # Should NOT be evicted (frozen)
        assert (tmp_path / "shard_000000.parquet").exists()
        assert s.shard_count == 5

    def test_unfreeze_executes_pending(self, tmp_path):
        """After unfreeze, pending evictions execute on next refresh."""
        _write_shards(tmp_path, n=5)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        s.freeze()

        s.schedule_eviction(tmp_path / "shard_000000.parquet")
        s.unfreeze()
        time.sleep(0.02)
        _ = len(s)

        assert not (tmp_path / "shard_000000.parquet").exists()
        assert s.shard_count == 4

    def test_freeze_blocks_auto_eviction(self, tmp_path):
        """When frozen, max_cache_gb auto-eviction is deferred."""
        _write_shards(tmp_path, n=10)
        total = sum(p.stat().st_size for p in tmp_path.glob("*.parquet"))
        half_gb = (total * 0.55) / 1_073_741_824
        s = ShardStorage(tmp_path, refresh_interval=0.01, max_cache_gb=half_gb)
        initial_count = s.shard_count
        assert initial_count <= 6  # evicted to ~half

        s.freeze()
        # Write more shards — should exceed limit but not evict
        for i in range(10, 15):
            _write_text_shard(tmp_path / f"shard_{i:06d}.parquet", ["x"] * 5)
        time.sleep(0.02)
        _ = len(s)

        # Should be over limit (frozen)
        assert s.shard_count > initial_count

        s.unfreeze()
        time.sleep(0.02)
        _ = len(s)
        assert s.shard_count <= initial_count + 1

    def test_load_state_dict_freezes(self, tmp_path):
        """load_state_dict automatically freezes."""
        _write_shards(tmp_path, n=3)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        state = s.state_dict()

        s.load_state_dict(state)
        # Should be frozen after load
        s.schedule_eviction(tmp_path / "shard_000000.parquet")
        time.sleep(0.02)
        _ = len(s)
        assert (tmp_path / "shard_000000.parquet").exists()  # not evicted


# ---------------------------------------------------------------------------
# Resume scenario simulation
# ---------------------------------------------------------------------------

class TestResumeScenario:

    def test_pause_resume_sees_same_data(self, tmp_path):
        """Simulate: train → pause → shard changes → resume → same data."""
        _write_shards(tmp_path, n=5, docs_per_shard=20)
        s = ShardStorage(tmp_path, refresh_interval=0.01)

        # "Train" — read some data
        data_at_pause = [s.get(i)["text"] for i in range(100)]
        state = s.state_dict()

        # "Shards change" — prefetcher writes new, evicts old
        (tmp_path / "shard_000000.parquet").unlink()
        _write_text_shard(tmp_path / "shard_000005.parquet", ["new"] * 20)

        # "Resume" — load state, should see same data as before pause
        s2 = ShardStorage(tmp_path, refresh_interval=0.01)
        s2.load_state_dict(state)

        # shard_000000 was deleted — so we can't read it.
        # But the pinned list only includes shards that still exist.
        # We should still get a consistent view (4 of the original 5).
        assert s2.shard_count == 4
        assert s2.state_dict()["total_rows"] == 80

    def test_full_cycle(self, tmp_path):
        """Full cycle: create → freeze → read → state_dict → unfreeze → evict → load → verify."""
        _write_shards(tmp_path, n=5, docs_per_shard=10)
        s = ShardStorage(tmp_path, refresh_interval=0.01)

        # Freeze for epoch
        s.freeze()
        initial_count = len(s)

        # Read all data
        all_data = [s.get(i) for i in range(initial_count)]
        assert all(d is not None for d in all_data)

        # Save state
        state = s.state_dict()

        # Unfreeze — allow eviction
        s.unfreeze()

        # Simulate prefetcher adding shards
        _write_text_shard(tmp_path / "shard_000005.parquet", ["x"] * 10)
        time.sleep(0.02)
        _ = len(s)
        assert s.shard_count == 6

        # Resume from state — pins back to 5 original shards
        s.load_state_dict(state)
        assert s.shard_count == 5
        assert len(s) == initial_count

        # Data is the same
        for i in range(initial_count):
            assert s.get(i) == all_data[i]
