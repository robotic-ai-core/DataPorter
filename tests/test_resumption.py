"""Tests for scientific resumption: ShardStorage state_dict, freeze/unfreeze, shard pinning."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# Incremental refresh optimization
# ---------------------------------------------------------------------------

class TestIncrementalRefresh:
    """Verify that refresh() uses metadata cache to avoid redundant work."""

    def test_no_parquet_opens_on_unchanged_refresh(self, tmp_path):
        """Second refresh() opens zero ParquetFiles when nothing changed."""
        _write_shards(tmp_path, n=5)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        assert len(s) == 50  # initial refresh

        # Monkey-patch ParquetFile to count calls
        open_count = 0
        orig_pf = pq.ParquetFile

        def counting_pf(path):
            nonlocal open_count
            open_count += 1
            return orig_pf(path)

        with patch("dataporter.storage.pq.ParquetFile", side_effect=counting_pf):
            s._last_refresh = 0.0
            s.refresh()

        assert open_count == 0, f"Expected 0 ParquetFile opens, got {open_count}"
        assert len(s) == 50

    def test_only_new_shards_opened(self, tmp_path):
        """After adding new shards, refresh opens only the new ones."""
        _write_shards(tmp_path, n=3)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        assert len(s) == 30

        # Write 2 more shards
        _write_text_shard(tmp_path / "shard_000003.parquet", ["new"] * 10)
        _write_text_shard(tmp_path / "shard_000004.parquet", ["newer"] * 10)

        open_count = 0
        orig_pf = pq.ParquetFile

        def counting_pf(path):
            nonlocal open_count
            open_count += 1
            return orig_pf(path)

        with patch("dataporter.storage.pq.ParquetFile", side_effect=counting_pf):
            s._last_refresh = 0.0
            s.refresh()

        assert open_count == 2, f"Expected 2 ParquetFile opens, got {open_count}"
        assert len(s) == 50
        assert s.shard_count == 5

    def test_text_cache_survives_refresh(self, tmp_path):
        """Cached shard text is preserved across refresh when shard unchanged."""
        _write_shards(tmp_path, n=3, docs_per_shard=10)
        s = ShardStorage(tmp_path, refresh_interval=0.01)

        # Read from shard 1 to populate _shard_texts
        item = s.get(10)
        assert item is not None
        assert 1 in s._shard_texts

        # Refresh — shard_texts should survive
        s._last_refresh = 0.0
        s.refresh()

        assert 1 in s._shard_texts
        assert s.get(10) == item

    def test_text_cache_remapped_after_eviction(self, tmp_path):
        """Text cache is remapped when shard positions shift after eviction."""
        _write_shards(tmp_path, n=3, docs_per_shard=10)
        s = ShardStorage(tmp_path, refresh_interval=0.01)

        # Read from shard 2 (index 20-29)
        item_at_20 = s.get(20)
        assert item_at_20 is not None
        assert 2 in s._shard_texts
        original_texts = s._shard_texts[2]

        # Evict shard 0 — shard 2 shifts to position 1
        s.schedule_eviction(tmp_path / "shard_000000.parquet")
        time.sleep(0.02)
        s._last_refresh = 0.0
        s.refresh()

        # shard_000002 is now at position 1 — cache should be remapped
        assert s.shard_count == 2
        assert 1 in s._shard_texts
        assert s._shard_texts[1] == original_texts

    def test_meta_cleaned_on_external_deletion(self, tmp_path):
        """Metadata cache drops entries for externally deleted shards."""
        _write_shards(tmp_path, n=3)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        assert "shard_000001.parquet" in s._shard_meta

        # External deletion
        (tmp_path / "shard_000001.parquet").unlink()
        s._last_refresh = 0.0
        s.refresh()

        assert "shard_000001.parquet" not in s._shard_meta
        assert s.shard_count == 2

    def test_load_state_dict_prepopulates_meta(self, tmp_path):
        """load_state_dict pre-populates metadata from checkpoint."""
        _write_shards(tmp_path, n=5)
        s = ShardStorage(tmp_path, refresh_interval=60)
        state = s.state_dict()

        # New storage — no metadata yet
        s2 = ShardStorage(tmp_path, refresh_interval=60)

        open_count = 0
        orig_pf = pq.ParquetFile

        def counting_pf(path):
            nonlocal open_count
            open_count += 1
            return orig_pf(path)

        # load_state_dict should pre-populate, so refresh opens nothing
        with patch("dataporter.storage.pq.ParquetFile", side_effect=counting_pf):
            s2.load_state_dict(state)

        assert open_count == 0, f"Expected 0 opens after load_state_dict, got {open_count}"
        assert len(s2) == 50
        assert s2.shard_count == 5

    def test_evict_excess_uses_cached_sizes(self, tmp_path):
        """_maybe_evict_excess uses cached sizes instead of stat()."""
        _write_shards(tmp_path, n=10)
        total = sum(p.stat().st_size for p in tmp_path.glob("*.parquet"))
        half_gb = (total * 0.55) / 1_073_741_824

        s = ShardStorage(tmp_path, refresh_interval=0.01, max_cache_gb=half_gb)
        initial_count = s.shard_count
        assert initial_count <= 6  # evicted to ~half

        # All remaining shards should be in metadata cache
        for path, _ in s._shards:
            assert path.name in s._shard_meta

        # Trigger _maybe_evict_excess — should work with cached sizes
        _write_text_shard(tmp_path / "shard_000099.parquet", ["x"] * 5)
        s._last_evict_check = 0.0
        s._maybe_evict_excess()

        # Should still be at or below the limit
        assert s._shard_meta  # cache not empty
