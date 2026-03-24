"""LeRobot episode prefetcher — downloads episodes (Parquet + video) incrementally.

Subclasses BasePrefetcher for LeRobot-format datasets where each episode
is a Parquet file + one MP4 per video key. Auto-detects video keys from
``meta/info.json`` — zero per-dataset configuration needed.

On-disk layout mirrors HuggingFace Hub:
    output_dir/
    ├── meta/info.json, episodes.jsonl, ...
    ├── data/chunk-000/episode_000000.parquet
    └── videos/chunk-000/observation.images.laptop/episode_000000.mp4
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable

from .prefetcher import (
    BasePrefetcher,
    CompanionPool,
    CompanionRef,
    _write_manifest,
    evict_shard,
)

logger = logging.getLogger(__name__)


def _hf_hub_download_file(remote: str, local_path: Path) -> None:
    """Download a single file from HuggingFace Hub with rate limit retry.

    ``remote`` is formatted as ``repo_id::file_path[::revision]``.
    Retries up to 3 times on 429 rate limit with 310s backoff
    (HF rate limit window is 5 minutes).
    """
    import time

    parts = remote.split("::")
    repo_id = parts[0]
    file_path = parts[1]
    revision = parts[2] if len(parts) > 2 else None

    from huggingface_hub import hf_hub_download

    for attempt in range(3):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                revision=revision,
                local_dir=local_path.parent.parent,
            )
            return
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                logger.warning(
                    f"HF rate limited downloading {file_path}, "
                    f"waiting 310s (attempt {attempt + 1}/3)"
                )
                time.sleep(310)
            else:
                raise


class LeRobotPrefetcher(BasePrefetcher):
    """Prefetcher for LeRobot episode datasets.

    Downloads episodes (Parquet + video MP4s) incrementally from HuggingFace
    Hub. Auto-detects video keys from ``meta/info.json``.

    Args:
        repo_id: HuggingFace dataset repo (e.g. "lerobot/pusht").
        output_dir: Local directory to mirror the dataset structure.
        revision: Dataset revision/version. Defaults to "v2.1".
        episode_indices: Specific episodes to download. None = all.
        min_shards: Min episodes ready before training starts.
        max_shards: Max episodes on disk (eviction).
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
        super().__init__(
            output_dir=output_dir,
            min_shards=min_shards,
            max_shards=max_shards,
            eviction=eviction,
            seed=seed,
            shard_glob="**/episode_*.parquet",
        )
        self._repo_id = repo_id
        self._revision = revision or "v2.1"
        self._episode_indices = episode_indices
        self._companion_workers = companion_workers
        self._download_fn = _download_fn or _hf_hub_download_file
        self._meta_loader = _meta_loader
        self._meta: dict | None = None
        self._companion_pool: CompanionPool | None = None

    def _on_start(self) -> None:
        self._companion_pool = CompanionPool(
            companion_dir=self._output_dir,
            max_workers=self._companion_workers,
            download_fn=self._download_fn,
        )

    def _on_stop(self) -> None:
        if self._companion_pool is not None:
            self._companion_pool.shutdown(wait=False)
            self._companion_pool = None

    def _maybe_evict(self, rng: random.Random) -> None:
        if self._max_shards is None:
            return
        while self.shard_count > self._max_shards:
            evict_shard(
                self._output_dir,
                self._eviction,
                rng,
                companion_pool=self._companion_pool,
                companion_dir=self._output_dir,
                glob_pattern=self._shard_glob,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        if self._meta is not None:
            return self._meta
        if self._meta_loader is not None:
            self._meta = self._meta_loader(self._repo_id, self._output_dir)
            return self._meta

        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(
            self._repo_id, root=self._output_dir, revision=self._revision
        )
        self._meta = meta.info
        return self._meta

    def _get_video_keys(self) -> list[str]:
        info = self._load_meta()
        features = info.get("features", {})
        return [key for key, ft in features.items() if ft.get("dtype") == "video"]

    def _get_data_path_template(self) -> str:
        return self._load_meta().get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )

    def _get_video_path_template(self) -> str | None:
        return self._load_meta().get("video_path")

    def _get_chunks_size(self) -> int:
        return self._load_meta().get("chunks_size", 1000)

    def _get_total_episodes(self) -> int:
        return self._load_meta().get("total_episodes", 0)

    # ------------------------------------------------------------------
    # Episode path helpers
    # ------------------------------------------------------------------

    def _episode_parquet_path(self, ep_idx: int) -> str:
        chunk = ep_idx // self._get_chunks_size()
        return self._get_data_path_template().format(
            episode_chunk=chunk, episode_index=ep_idx
        )

    def _episode_video_paths(self, ep_idx: int) -> list[str]:
        template = self._get_video_path_template()
        if not template:
            return []
        chunk = ep_idx // self._get_chunks_size()
        return [
            template.format(episode_chunk=chunk, video_key=vk, episode_index=ep_idx)
            for vk in self._get_video_keys()
        ]

    def _episode_companion_refs(self, ep_idx: int) -> list[CompanionRef]:
        return [
            CompanionRef(
                remote=f"{self._repo_id}::{vpath}::{self._revision}",
                local=vpath,
            )
            for vpath in self._episode_video_paths(ep_idx)
        ]

    # ------------------------------------------------------------------
    # Download loop
    # ------------------------------------------------------------------

    def _run_inner(self) -> None:
        rng = random.Random(self._seed)

        self._load_meta()
        total = self._get_total_episodes()
        if self._episode_indices is not None:
            episode_order = list(self._episode_indices)
        else:
            episode_order = list(range(total))

        rng.shuffle(episode_order)

        for ep_idx in episode_order:
            if self._stop_event.is_set():
                break

            parquet_rel = self._episode_parquet_path(ep_idx)
            parquet_local = self._output_dir / parquet_rel
            parquet_local.parent.mkdir(parents=True, exist_ok=True)

            remote = f"{self._repo_id}::{parquet_rel}::{self._revision}"
            try:
                self._download_fn(remote, parquet_local)
            except Exception as e:
                logger.warning(f"Failed to download episode {ep_idx} parquet: {e}")
                continue

            companion_refs = self._episode_companion_refs(ep_idx)
            if companion_refs and self._companion_pool is not None:
                self._companion_pool.submit(parquet_local.name, companion_refs)
                _write_manifest(parquet_local, [ref.local for ref in companion_refs])

            self._check_min_ready()
            self._maybe_evict(rng)

        logger.info(f"All {len(episode_order)} episodes processed")
