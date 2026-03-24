"""LeRobot episode prefetcher — downloads episodes (Parquet + video) incrementally.

Subclasses ParquetPrefetcher for LeRobot-format datasets where each episode
is a Parquet file + one MP4 per video key. Auto-detects video keys from
``meta/info.json`` — zero per-dataset configuration needed.

On-disk layout mirrors HuggingFace Hub:
    output_dir/
    ├── meta/info.json, episodes.jsonl, ...
    ├── data/chunk-000/episode_000000.parquet
    └── videos/chunk-000/observation.images.laptop/episode_000000.mp4
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .prefetcher import CompanionPool, CompanionRef, ParquetPrefetcher, _count_ready_shards

logger = logging.getLogger(__name__)


def _hf_hub_download_file(remote: str, local_path: Path) -> None:
    """Download a single file from HuggingFace Hub.

    ``remote`` is formatted as ``repo_id::file_path[::revision]``.
    """
    parts = remote.split("::")
    repo_id = parts[0]
    file_path = parts[1]
    revision = parts[2] if len(parts) > 2 else None

    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        revision=revision,
        local_dir=local_path.parent.parent,  # download preserves relative path
    )
    # hf_hub_download writes to local_dir/file_path — if that differs from
    # local_path, move it. In practice local_dir is set so they match.


class LeRobotPrefetcher(ParquetPrefetcher):
    """Prefetcher for LeRobot episode datasets.

    Downloads episodes (Parquet + video MP4s) incrementally from HuggingFace
    Hub. Auto-detects video keys from ``meta/info.json``.

    Unlike the generic ParquetPrefetcher which streams rows and writes custom
    shards, this downloads original episode files directly from the Hub so
    that FastLeRobotDataset can read them unchanged.

    Args:
        repo_id: HuggingFace dataset repo (e.g. "lerobot/pusht").
        output_dir: Local directory to mirror the dataset structure.
        revision: Dataset revision/version. Defaults to "v2.1".
        episode_indices: Specific episodes to download. None = all.
        min_shards: Min episodes ready before training starts.
        max_shards: Max episodes on disk (LRU eviction).
        eviction: Eviction strategy.
        companion_workers: Threads for video downloads.
        seed: Random seed.
        _download_fn: Override download function (for testing).
        _meta_loader: Override metadata loading (for testing).
    """

    def __init__(
        self,
        repo_id: str,
        output_dir: str | Path,
        revision: str | None = None,
        episode_indices: list[int] | None = None,
        min_shards: int = 5,
        max_shards: int = 100,
        eviction: str = "stochastic_oldest",
        companion_workers: int = 4,
        seed: int = 42,
        _download_fn: Callable[[str, Path], None] | None = None,
        _meta_loader: Callable[[str, Path], dict] | None = None,
    ):
        self._repo_id = repo_id
        self._revision = revision or "v2.1"
        self._episode_indices = episode_indices
        self._meta_loader = _meta_loader

        output_dir = Path(output_dir)

        # We pass a dummy source — _run_inner is fully overridden
        super().__init__(
            sources=[{"dataset": repo_id}],
            output_dir=output_dir,
            companion_dir=output_dir,  # videos live alongside data
            companion_workers=companion_workers,
            companion_download_fn=_download_fn or _hf_hub_download_file,
            min_shards=min_shards,
            max_shards=max_shards,
            eviction=eviction,
            seed=seed,
            # These aren't used since we override _run_inner
            transform=None,
            stream_shuffle_buffer=0,
            max_rows_per_shard=1,
            row_group_size=1,
        )

        self._meta: dict | None = None

    def _load_meta(self) -> dict:
        """Load dataset metadata (info.json)."""
        if self._meta is not None:
            return self._meta

        if self._meta_loader is not None:
            self._meta = self._meta_loader(self._repo_id, self._output_dir)
            return self._meta

        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(
            self._repo_id,
            root=self._output_dir,
            revision=self._revision,
        )
        self._meta = meta.info
        return self._meta

    def _get_video_keys(self) -> list[str]:
        """Auto-detect video keys from metadata."""
        info = self._load_meta()
        features = info.get("features", {})
        return [key for key, ft in features.items() if ft.get("dtype") == "video"]

    def _get_data_path_template(self) -> str:
        info = self._load_meta()
        return info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )

    def _get_video_path_template(self) -> str | None:
        info = self._load_meta()
        return info.get("video_path")

    def _get_chunks_size(self) -> int:
        info = self._load_meta()
        return info.get("chunks_size", 1000)

    def _get_total_episodes(self) -> int:
        info = self._load_meta()
        return info.get("total_episodes", 0)

    def _episode_parquet_path(self, ep_idx: int) -> str:
        """Relative path for an episode's Parquet file."""
        chunk = ep_idx // self._get_chunks_size()
        return self._get_data_path_template().format(
            episode_chunk=chunk, episode_index=ep_idx
        )

    def _episode_video_paths(self, ep_idx: int) -> list[str]:
        """Relative paths for an episode's video files."""
        template = self._get_video_path_template()
        if not template:
            return []
        chunk = ep_idx // self._get_chunks_size()
        video_keys = self._get_video_keys()
        return [
            template.format(
                episode_chunk=chunk, video_key=vk, episode_index=ep_idx
            )
            for vk in video_keys
        ]

    def _episode_companion_refs(self, ep_idx: int) -> list[CompanionRef]:
        """Build CompanionRef objects for an episode's video files."""
        refs = []
        for vpath in self._episode_video_paths(ep_idx):
            refs.append(
                CompanionRef(
                    remote=f"{self._repo_id}::{vpath}::{self._revision}",
                    local=vpath,
                )
            )
        return refs

    @property
    def shard_count(self) -> int:
        """Number of episode Parquet files on disk."""
        return len(list(self._output_dir.rglob("episode_*.parquet")))

    def _run_inner(self) -> None:
        """Download episodes one by one: Parquet first, then submit video companions."""
        import random as rand_module

        rng = rand_module.Random(self._seed)

        self._load_meta()
        total = self._get_total_episodes()
        if self._episode_indices is not None:
            episode_order = list(self._episode_indices)
        else:
            episode_order = list(range(total))

        rng.shuffle(episode_order)

        download_fn = self._companion_download_fn or _hf_hub_download_file

        for ep_idx in episode_order:
            if self._stop_event.is_set():
                break

            # Download Parquet file
            parquet_rel = self._episode_parquet_path(ep_idx)
            parquet_local = self._output_dir / parquet_rel
            parquet_local.parent.mkdir(parents=True, exist_ok=True)

            remote_parquet = f"{self._repo_id}::{parquet_rel}::{self._revision}"
            try:
                download_fn(remote_parquet, parquet_local)
            except Exception as e:
                logger.warning(f"Failed to download episode {ep_idx} parquet: {e}")
                continue

            # Submit video companion downloads
            companion_refs = self._episode_companion_refs(ep_idx)
            if companion_refs and self._companion_pool is not None:
                self._companion_pool.submit(parquet_local.name, companion_refs)

            # Check readiness and eviction
            self._check_min_ready()
            self._maybe_evict(rng)

        logger.info(f"All {len(episode_order)} episodes processed")
