"""LeRobot episode prefetcher — downloads datasets via snapshot_download.

For small/medium datasets: downloads everything in one batched call via
``snapshot_download`` (fast, no rate limit issues).

For TB-scale datasets: downloads in batches via ``allow_patterns``,
with eviction to stay within disk budget.

On-disk layout mirrors HuggingFace Hub:
    cache_dir/
    ├── meta/info.json, episodes.jsonl, ...
    ├── data/chunk-000/episode_000000.parquet
    └── videos/chunk-000/observation.images.laptop/episode_000000.mp4
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .prefetcher import BasePrefetcher, evict_shard

logger = logging.getLogger(__name__)

# Threshold: datasets with fewer episodes than this are downloaded in full.
# Above this, incremental batch download with eviction.
_BULK_THRESHOLD = 10_000


def _snapshot_with_retry(
    repo_id: str,
    revision: str,
    local_dir: Path,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_retries: int = 3,
    use_hub_cache: bool = False,
) -> None:
    """snapshot_download with retry on 429.

    Two modes controlled by ``use_hub_cache``:

    - **Hub-cache mode** (``use_hub_cache=True``): downloads to HF's
      shared cache dir (no ``local_dir`` kwarg), then symlinks
      ``local_dir`` → snapshot path. Avoids a huggingface_hub issue
      where ``local_dir`` mode issues a HEAD call per file and raises
      ``LocalEntryNotFoundError`` under rate limiting. Safe only when
      the caller does NOT modify files through ``local_dir`` (e.g.
      bulk download without eviction).

    - **Local-dir mode** (``use_hub_cache=False``, default): downloads
      directly into ``local_dir``. Files are owned by the caller and
      safe to evict/delete. Required for incremental download with
      shard eviction.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    for attempt in range(max_retries):
        try:
            if use_hub_cache:
                snapshot_path = Path(
                    snapshot_download(
                        repo_id,
                        repo_type="dataset",
                        revision=revision,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                    )
                )
                # Symlink local_dir → hub snapshot (idempotent).
                local_dir.parent.mkdir(parents=True, exist_ok=True)
                if local_dir.is_symlink():
                    if os.readlink(local_dir) == str(snapshot_path):
                        return
                    local_dir.unlink()
                elif local_dir.exists() and local_dir.is_dir():
                    shutil.rmtree(local_dir)
                local_dir.symlink_to(snapshot_path, target_is_directory=True)
            else:
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(local_dir),
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                )
            return
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 60 * (2 ** attempt)
                logger.warning(
                    f"HF 429 (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise


class LeRobotPrefetcher(BasePrefetcher):
    """Prefetcher for LeRobot episode datasets.

    Downloads episodes (Parquet + video MP4s) from HuggingFace Hub using
    ``snapshot_download`` for efficient batched retrieval.

    Two modes (auto-selected from total_episodes):
    - **Bulk** (< 10k episodes): downloads everything in one call.
    - **Incremental** (>= 10k episodes): downloads in batches with eviction.

    Args:
        repo_id: HuggingFace dataset repo (e.g. "lerobot/pusht").
        cache_dir: Local directory to mirror the dataset structure.
        revision: Dataset revision/version. Defaults to "v2.1".
        episode_indices: Specific episodes to download. None = all.
        min_shards: Min episodes ready before training starts.
        max_shards: Max episodes on disk (eviction, incremental mode only).
        eviction: Eviction strategy.
        batch_size: Episodes per batch in incremental mode.
        seed: Random seed.
        _snapshot_fn: Override snapshot_download (for testing).
        _meta_loader: Override metadata loading (for testing).
    """

    def __init__(
        self,
        repo_id: str,
        cache_dir: str | Path,
        revision: str | None = None,
        episode_indices: list[int] | None = None,
        min_shards: int = 5,
        max_shards: int = 10_000,
        eviction: str = "stochastic_oldest",
        batch_size: int = 500,
        seed: int = 42,
        _snapshot_fn: Callable | None = None,
        _meta_loader: Callable[[str, Path], dict] | None = None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            min_shards=min_shards,
            max_shards=max_shards,
            eviction=eviction,
            seed=seed,
            shard_glob="**/episode_*.parquet",
        )
        self._use_thread = _snapshot_fn is not None or _meta_loader is not None
        self._repo_id = repo_id
        self._revision = revision or "v2.1"
        self._episode_indices = episode_indices
        self._batch_size = batch_size
        self._snapshot_fn = _snapshot_fn
        self._meta_loader = _meta_loader
        self._meta: dict | None = None

    def _get_init_kwargs(self) -> dict[str, Any]:
        return dict(
            repo_id=self._repo_id,
            cache_dir=str(self._cache_dir),
            revision=self._revision,
            episode_indices=self._episode_indices,
            min_shards=self._min_shards,
            max_shards=self._max_shards,
            eviction=self._eviction,
            batch_size=self._batch_size,
            seed=self._seed,
        )

    def _maybe_evict(self, rng: random.Random) -> None:
        if self._max_shards is None:
            return
        while self.shard_count > self._max_shards:
            evict_shard(
                self._cache_dir,
                self._eviction,
                rng,
                companion_dir=self._cache_dir,
                glob_pattern=self._shard_glob,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        if self._meta is not None:
            return self._meta
        if self._meta_loader is not None:
            self._meta = self._meta_loader(self._repo_id, self._cache_dir)
            return self._meta

        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(
            self._repo_id, root=self._cache_dir, revision=self._revision
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

    def _episode_patterns(self, ep_indices: list[int]) -> list[str]:
        """Build allow_patterns for a batch of episodes."""
        patterns = []
        for ep_idx in ep_indices:
            patterns.append(self._episode_parquet_path(ep_idx))
            patterns.extend(self._episode_video_paths(ep_idx))
        return patterns

    # ------------------------------------------------------------------
    # Snapshot download
    # ------------------------------------------------------------------

    def _do_snapshot(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        use_hub_cache: bool = False,
    ) -> None:
        """Run snapshot_download (or test mock).

        Args:
            use_hub_cache: If True, download to HF's shared cache and
                symlink ``cache_dir`` to it. Only safe for bulk downloads
                where the caller won't evict/delete files through the
                symlink. Incremental mode must use False (the default).
        """
        if self._snapshot_fn is not None:
            self._snapshot_fn(
                self._repo_id,
                self._cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        else:
            _snapshot_with_retry(
                self._repo_id,
                self._revision,
                self._cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                use_hub_cache=use_hub_cache,
            )

    # ------------------------------------------------------------------
    # Download loop
    # ------------------------------------------------------------------

    def _run_inner(self) -> None:
        self._load_meta()
        total = self._get_total_episodes()

        if self._episode_indices is not None:
            episodes = list(self._episode_indices)
        else:
            episodes = list(range(total))

        if len(episodes) <= _BULK_THRESHOLD:
            self._download_bulk(episodes)
        else:
            self._download_incremental(episodes)

    def _download_bulk(self, episodes: list[int]) -> None:
        """Download all episodes in one snapshot_download call.

        Uses hub-cache mode (symlink) to avoid per-file HEAD calls that
        trigger HF XET rate limits. Safe here because bulk mode never
        evicts — the entire dataset is downloaded and kept.

        After the initial download, verifies all episode files exist
        and retries missing ones. HF 429 rate limits can interrupt
        snapshot_download mid-flight, leaving an incomplete cache.
        """
        if self._episode_indices is not None:
            patterns = ["meta/*"] + self._episode_patterns(episodes)
            self._do_snapshot(allow_patterns=patterns, use_hub_cache=True)
        else:
            self._do_snapshot(use_hub_cache=True)

        # Verify and retry missing episodes. HF's XET rate limit
        # (500 req/5min per IP) causes partial downloads — each retry
        # picks up ~500 more files. Wait 5 min between retries for the
        # rate limit window to reset.
        missing = self._find_missing_episodes(episodes)
        while missing and not self._stop_event.is_set():
            logger.warning(
                f"Bulk download incomplete: {len(missing)}/{len(episodes)} "
                f"episodes missing. Waiting 5 min for HF rate limit reset..."
            )
            self._stop_event.wait(timeout=310)  # 5 min + 10s buffer
            if self._stop_event.is_set():
                break
            patterns = ["meta/*"] + self._episode_patterns(missing)
            self._do_snapshot(allow_patterns=patterns, use_hub_cache=True)
            missing = self._find_missing_episodes(episodes)

        if missing:
            logger.warning(
                f"Still missing {len(missing)} episodes at shutdown: "
                f"{missing[:10]}..."
            )
        else:
            logger.info(
                f"Bulk download verified: all {len(episodes)} episodes present"
            )

        self._check_min_ready()

    def _find_missing_episodes(self, episodes: list[int]) -> list[int]:
        """Check which episodes are missing from the cache directory.

        Skipped when ``_snapshot_fn`` is set (test mode — mock downloads
        don't create real files on disk).
        """
        if self._snapshot_fn is not None:
            return []
        missing = []
        for ep_idx in episodes:
            parquet = self._cache_dir / self._episode_parquet_path(ep_idx)
            if not parquet.is_file():
                missing.append(ep_idx)
        return missing

    def _download_incremental(self, episodes: list[int]) -> None:
        """Download episodes in batches with eviction."""
        rng = random.Random(self._seed)
        rng.shuffle(episodes)

        for batch_start in range(0, len(episodes), self._batch_size):
            if self._stop_event.is_set():
                break

            batch = episodes[batch_start : batch_start + self._batch_size]
            patterns = self._episode_patterns(batch)

            try:
                self._do_snapshot(allow_patterns=["meta/*"] + patterns)
            except Exception as e:
                logger.warning(f"Failed to download batch at {batch_start}: {e}")
                continue

            self._check_min_ready()
            self._maybe_evict(rng)

        logger.info(f"Incremental download complete: {len(episodes)} episodes")
