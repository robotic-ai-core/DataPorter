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
import random
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
) -> None:
    """snapshot_download into local_dir with retry on 429."""
    from huggingface_hub import snapshot_download

    for attempt in range(max_retries):
        try:
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
    # Episode-level readiness (parquet + video present)
    # ------------------------------------------------------------------

    def ready_episodes(self) -> list[int]:
        """Episodes whose parquet AND all video files exist on disk.

        ``BasePrefetcher.shard_count`` counts parquets only; training
        consumers that decode video need both.  This predicate is the
        authoritative source for "what's safe to load right now" from
        the WRITER side.  :meth:`LeRobotShardSource.list_ready_episodes`
        answers the same question from the READER side (no prefetcher
        state — pure disk scan); the two are kept in sync deliberately
        and either can be the source of truth for a given caller.

        Returns a sorted list of ready episode ids.  Empty until the
        prefetcher has produced a complete episode (important during the
        first few batches of a fresh download).
        """
        cache_dir = self._cache_dir
        # Scan parquets first (cheap rglob) to narrow the set, then
        # verify each candidate has all its video files.
        import re
        parquet_eps = {
            int(m.group(1))
            for p in cache_dir.rglob("episode_*.parquet")
            if (m := re.search(r"episode_(\d+)", p.stem))
        }
        # Only load meta if there are any parquets to check — avoids a
        # spurious HF call during a cold start.
        if not parquet_eps:
            return []
        try:
            self._load_meta()
        except Exception:
            # Metadata not available yet (e.g. meta/* hasn't finished
            # downloading).  Fall back to parquet-only readiness in that
            # window — ffmpeg will still fail loudly downstream, but we
            # don't hide episodes that actually are complete.
            return sorted(parquet_eps)
        # If the dataset has no video features, parquet presence is
        # sufficient.
        if not self._get_video_keys():
            return sorted(parquet_eps)
        video_template = self._get_video_path_template()
        if not video_template:
            return sorted(parquet_eps)
        ready: list[int] = []
        for ep_idx in sorted(parquet_eps):
            if all(
                (cache_dir / video_path).is_file()
                for video_path in self._episode_video_paths(ep_idx)
            ):
                ready.append(ep_idx)
        return ready

    def _check_min_ready(self) -> None:
        # Override BasePrefetcher's parquet-count check: an episode is
        # only usable when BOTH parquet AND its video files are on disk.
        # Without this, a fresh prefetch signals ready as soon as N
        # parquets land but before the MP4s finish, and ProducerPool
        # crashes on the first decode.
        if self._min_ready.is_set():
            return
        if len(self.ready_episodes()) >= self._min_shards:
            self._min_ready.set()
            logger.info(
                f"Min episodes ready ({self._min_shards} with parquet+video), "
                f"training can start"
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

    @property
    def total_episodes(self) -> int:
        """Total episode count reported by the dataset metadata.

        Triggers a metadata load on first access; consumers can call
        this before ``start()`` to compute size-aware setup gates
        (e.g. ``prefetch_min_fraction`` in ``BlendedLeRobotDataModule``).
        """
        return self._get_total_episodes()

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
    ) -> None:
        """Run snapshot_download (or test mock)."""
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

    # HF XET rate limit: 500 requests / 5 min per IP. Each episode
    # has ~2 files (parquet + video), so ~200 episodes per batch stays
    # under the limit with headroom.
    _EPISODES_PER_BATCH = 200
    _BATCH_COOLDOWN_S = 310  # 5 min + 10s buffer

    def _download_bulk(self, episodes: list[int]) -> None:
        """Download episodes in rate-limited batches.

        Uses local-dir mode (direct file copy, no symlinks) to avoid
        pyav hanging on symlinked blob paths. Batches requests to stay
        under HF's XET 500-req/5min per-IP limit.

        Previously used hub-cache mode (symlinks) but this caused:
        - pyav decode hangs on double symlink chains (dir→snapshot→blob)
        - SameFileError when snapshot_download copies onto itself
        """
        # Download metadata first
        self._do_snapshot(allow_patterns=["meta/*"])

        # Batch episodes to stay under rate limit
        total = len(episodes)
        for batch_start in range(0, total, self._EPISODES_PER_BATCH):
            if self._stop_event.is_set():
                break

            batch = episodes[batch_start:batch_start + self._EPISODES_PER_BATCH]
            batch_end = min(batch_start + self._EPISODES_PER_BATCH, total)
            logger.info(
                f"Downloading episodes {batch_start}-{batch_end - 1} "
                f"of {total} ({batch_end * 100 // total}%)"
            )

            patterns = self._episode_patterns(batch)
            self._do_snapshot(allow_patterns=patterns)
            self._check_min_ready()

            # Cooldown between batches (skip after last batch)
            if batch_end < total and not self._stop_event.is_set():
                logger.info(
                    f"Rate limit cooldown: waiting "
                    f"{self._BATCH_COOLDOWN_S}s before next batch..."
                )
                self._stop_event.wait(timeout=self._BATCH_COOLDOWN_S)

        # Verify completeness
        missing = self._find_missing_episodes(episodes)
        if missing:
            logger.warning(
                f"Download incomplete: {len(missing)}/{total} episodes "
                f"missing after batched download: {missing[:10]}..."
            )
        else:
            logger.info(
                f"Bulk download complete: all {total} episodes verified"
            )

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
