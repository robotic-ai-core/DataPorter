"""Tests for HF cache pre-flight check and sentinel mechanism.

These helpers pair with sparkinstance's ``hf_cache_source`` feature:
  - ``check_hf_cache_populated`` detects missing datasets before the
    job hits HF XET rate limits mid-training
  - ``write_cache_sentinel`` writes ``.dataporter_cache_complete`` after
    a successful load so sparkinstance's ``skip_if_present`` can detect
    a healthy cache and skip the pre-sync rsync
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def test_hf_cache_repo_path_default():
    """Default HF_HOME is ~/.cache/huggingface."""
    from dataporter import hf_cache_repo_path
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HF_HOME", None)
        path = hf_cache_repo_path("lerobot/pusht")
        assert path.name == "datasets--lerobot--pusht"
        assert path.parent.name == "hub"
        assert ".cache/huggingface" in str(path)


def test_hf_cache_repo_path_with_hf_home(tmp_path, monkeypatch):
    """HF_HOME env var overrides the default."""
    from dataporter import hf_cache_repo_path
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    path = hf_cache_repo_path("neiltan/pusht-synthetic-v3")
    assert path == tmp_path / "hub" / "datasets--neiltan--pusht-synthetic-v3"


def test_hf_cache_repo_path_nested_repo_id(tmp_path, monkeypatch):
    """Repo IDs with slashes are escaped with double-dashes."""
    from dataporter import hf_cache_repo_path
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    path = hf_cache_repo_path("org/sub/dataset")
    assert path.name == "datasets--org--sub--dataset"


def test_check_hf_cache_populated_missing_dir(tmp_path, monkeypatch):
    """Empty HF_HOME → not populated."""
    from dataporter import check_hf_cache_populated
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    populated, reason = check_hf_cache_populated("lerobot/pusht")
    assert not populated
    assert "does not exist" in reason


def test_check_hf_cache_populated_no_snapshots(tmp_path, monkeypatch):
    """Cache dir exists but has no snapshots/ subdir → not populated."""
    from dataporter import check_hf_cache_populated
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    cache_dir = tmp_path / "hub" / "datasets--lerobot--pusht"
    cache_dir.mkdir(parents=True)
    populated, reason = check_hf_cache_populated("lerobot/pusht")
    assert not populated
    assert "no snapshots dir" in reason


def test_check_hf_cache_populated_empty_snapshots(tmp_path, monkeypatch):
    """Snapshots dir exists but has no parquet files → not populated."""
    from dataporter import check_hf_cache_populated
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshots = tmp_path / "hub" / "datasets--lerobot--pusht" / "snapshots" / "abc123"
    snapshots.mkdir(parents=True)
    populated, reason = check_hf_cache_populated("lerobot/pusht")
    assert not populated
    assert "no parquet files" in reason


def test_check_hf_cache_populated_with_parquets(tmp_path, monkeypatch):
    """Cache has parquet files → populated."""
    from dataporter import check_hf_cache_populated
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshots = tmp_path / "hub" / "datasets--lerobot--pusht" / "snapshots" / "abc123"
    (snapshots / "data" / "chunk-000").mkdir(parents=True)
    (snapshots / "data" / "chunk-000" / "episode_000000.parquet").touch()
    populated, reason = check_hf_cache_populated("lerobot/pusht")
    assert populated
    assert "1 parquet files" in reason


def test_check_hf_cache_populated_multiple_parquets(tmp_path, monkeypatch):
    """Reports total parquet count."""
    from dataporter import check_hf_cache_populated
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshots = tmp_path / "hub" / "datasets--lerobot--pusht" / "snapshots" / "abc123"
    (snapshots / "data" / "chunk-000").mkdir(parents=True)
    for i in range(5):
        (snapshots / "data" / "chunk-000" / f"episode_{i:06d}.parquet").touch()
    populated, reason = check_hf_cache_populated("lerobot/pusht")
    assert populated
    assert "5 parquet files" in reason


def test_write_cache_sentinel_creates_file(tmp_path):
    """Sentinel file is written at the expected path."""
    from dataporter import write_cache_sentinel
    write_cache_sentinel(tmp_path)
    assert (tmp_path / ".dataporter_cache_complete").exists()


def test_write_cache_sentinel_creates_missing_parent(tmp_path):
    """Creates parent dirs if they don't exist."""
    from dataporter import write_cache_sentinel
    target = tmp_path / "a" / "b" / "c"
    write_cache_sentinel(target)
    assert (target / ".dataporter_cache_complete").exists()


def test_write_cache_sentinel_idempotent(tmp_path):
    """Calling twice is a no-op (doesn't raise)."""
    from dataporter import write_cache_sentinel
    write_cache_sentinel(tmp_path)
    write_cache_sentinel(tmp_path)  # second call should not raise
    assert (tmp_path / ".dataporter_cache_complete").exists()


def test_write_cache_sentinel_permission_denied(tmp_path, monkeypatch):
    """Permission errors are logged, not raised (defensive)."""
    from dataporter import write_cache_sentinel

    def boom(self, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", boom)
    # Should not raise — the sentinel is a best-effort hint
    write_cache_sentinel(tmp_path / "some_dir")


def test_preflight_sentinel_round_trip(tmp_path, monkeypatch):
    """Simulated end-to-end: sparkinstance rsyncs data → DataPorter
    check passes → DataPorter writes sentinel."""
    from dataporter import (
        check_hf_cache_populated,
        hf_cache_repo_path,
        write_cache_sentinel,
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    repo_id = "test/dataset"

    # Before rsync: not populated
    populated, _ = check_hf_cache_populated(repo_id)
    assert not populated

    # Simulate rsync: create the snapshot dir + parquets
    cache = hf_cache_repo_path(repo_id)
    snapshots = cache / "snapshots" / "deadbeef"
    (snapshots / "data" / "chunk-000").mkdir(parents=True)
    (snapshots / "data" / "chunk-000" / "episode_000000.parquet").touch()

    # After rsync: populated
    populated, _ = check_hf_cache_populated(repo_id)
    assert populated

    # DataPorter writes sentinel
    write_cache_sentinel(cache)
    assert (cache / ".dataporter_cache_complete").exists()
