"""Rate-limited HuggingFace Hub client.

All HF requests (downloads, metadata, streaming) should go through this
module's shared rate limiter. This prevents 429 rate limit errors when
multiple prefetcher threads and companion pool workers make concurrent
requests.

Usage:
    from dataporter.hf_client import hf_download, hf_snapshot, hf_load_dataset

    # All calls share a single rate limiter (150 req/min default)
    hf_download(repo_id="lerobot/pusht", filename="data/ep_000.parquet", ...)
    hf_snapshot(repo_id="lerobot/pusht", allow_patterns="meta/", ...)
    ds = hf_load_dataset("HuggingFaceTB/smollm-corpus", streaming=True, ...)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Args:
        rate: Tokens added per second.
        capacity: Maximum burst size (bucket capacity).
    """

    def __init__(self, rate: float, capacity: float):
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 600.0) -> bool:
        """Block until a token is available. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                wait = (1.0 - self._tokens) / self._rate

            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 0.5))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


# ---------------------------------------------------------------------------
# Shared global rate limiter
# ---------------------------------------------------------------------------

# HF free tier: 1000 req / 5 min = 200 req/min.
# Target 150 req/min = 2.5 req/sec with burst of 10.
_global_limiter = TokenBucket(rate=2.5, capacity=10)


def get_limiter() -> TokenBucket:
    """Get the shared HF rate limiter."""
    return _global_limiter


def set_limiter(limiter: TokenBucket) -> None:
    """Replace the shared limiter (for testing)."""
    global _global_limiter
    _global_limiter = limiter


# ---------------------------------------------------------------------------
# Rate-limited HF operations
# ---------------------------------------------------------------------------

_MAX_RETRIES = 4
_BASE_BACKOFF = 30.0  # seconds


def _is_rate_limit(e: Exception) -> bool:
    """Check if exception is a 429 rate limit error."""
    return "429" in str(e)


def _retry_with_backoff(fn, *args, **kwargs) -> Any:
    """Call fn with rate limiting and retry on 429."""
    for attempt in range(_MAX_RETRIES):
        limiter = get_limiter()
        if not limiter.acquire(timeout=600):
            raise TimeoutError("Rate limiter timeout waiting for token")

        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit(e) and attempt < _MAX_RETRIES - 1:
                wait = _BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"HF 429 rate limit (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                raise


def hf_download(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    revision: str | None = None,
    local_dir: str | Path | None = None,
) -> Path:
    """Rate-limited wrapper around ``huggingface_hub.hf_hub_download``."""
    from huggingface_hub import hf_hub_download

    return _retry_with_backoff(
        hf_hub_download,
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
    )


def hf_snapshot(
    repo_id: str,
    repo_type: str = "dataset",
    revision: str | None = None,
    local_dir: str | Path | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> Path:
    """Rate-limited wrapper around ``huggingface_hub.snapshot_download``.

    Note: snapshot_download makes multiple requests internally.
    We rate-limit the initial call; individual file fetches within
    the snapshot are handled by HF's own retry logic.
    """
    from huggingface_hub import snapshot_download

    return _retry_with_backoff(
        snapshot_download,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def hf_load_dataset(
    path: str,
    data_dir: str | None = None,
    split: str = "train",
    streaming: bool = True,
    **kwargs: Any,
) -> Any:
    """Rate-limited wrapper around ``datasets.load_dataset``."""
    from datasets import load_dataset

    return _retry_with_backoff(
        load_dataset,
        path,
        data_dir=data_dir,
        split=split,
        streaming=streaming,
        **kwargs,
    )


def make_hf_download_fn(
    repo_id: str,
    revision: str | None = None,
) -> callable:
    """Create a download function for use with CompanionPool.

    Returns a callable with signature (remote: str, local_path: Path) -> None
    that goes through the shared rate limiter.

    ``remote`` is formatted as ``repo_id::file_path[::revision]`` or
    just a plain file path (in which case repo_id/revision from the
    closure are used).
    """

    def download_fn(remote: str, local_path: Path) -> None:
        if "::" in remote:
            parts = remote.split("::")
            rid = parts[0]
            fpath = parts[1]
            rev = parts[2] if len(parts) > 2 else revision
        else:
            rid = repo_id
            fpath = remote
            rev = revision

        hf_download(
            repo_id=rid,
            filename=fpath,
            repo_type="dataset",
            revision=rev,
            local_dir=local_path.parent.parent,
        )

    return download_fn
