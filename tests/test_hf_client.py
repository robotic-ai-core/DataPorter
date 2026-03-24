"""Tests for the HF rate-limited client.

Tests the token bucket rate limiter, retry-on-429, and shared budget
across concurrent callers.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from dataporter.hf_client import (
    TokenBucket,
    _is_rate_limit,
    _retry_with_backoff,
    get_limiter,
    make_hf_download_fn,
    set_limiter,
)


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:

    def test_immediate_acquire_when_tokens_available(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        t0 = time.monotonic()
        assert bucket.acquire(timeout=1.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1  # should be instant

    def test_blocks_when_empty(self):
        bucket = TokenBucket(rate=10.0, capacity=1)
        assert bucket.acquire(timeout=1.0)  # consume the one token

        t0 = time.monotonic()
        assert bucket.acquire(timeout=1.0)  # must wait for refill
        elapsed = time.monotonic() - t0
        assert 0.05 < elapsed < 0.5  # ~0.1s at 10 tokens/sec

    def test_burst_capacity(self):
        bucket = TokenBucket(rate=1.0, capacity=5)
        # Should be able to acquire 5 tokens immediately
        for _ in range(5):
            assert bucket.acquire(timeout=0.1)
        # 6th should block
        assert not bucket.acquire(timeout=0.05)

    def test_timeout_returns_false(self):
        bucket = TokenBucket(rate=0.1, capacity=1)
        assert bucket.acquire(timeout=1.0)  # consume
        assert not bucket.acquire(timeout=0.1)  # timeout

    def test_refill_over_time(self):
        bucket = TokenBucket(rate=100.0, capacity=10)
        # Drain all tokens
        for _ in range(10):
            bucket.acquire(timeout=0.1)
        # Wait for refill
        time.sleep(0.1)  # 100 * 0.1 = 10 tokens
        assert bucket.available_tokens >= 5  # at least some refilled

    def test_concurrent_acquire(self):
        """Multiple threads share the same bucket."""
        bucket = TokenBucket(rate=50.0, capacity=5)
        acquired = []
        lock = threading.Lock()

        def worker():
            for _ in range(3):
                if bucket.acquire(timeout=2.0):
                    with lock:
                        acquired.append(1)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # All 12 acquires should succeed (rate=50/s, 12 tokens needed)
        assert len(acquired) == 12

    def test_available_tokens_property(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        assert bucket.available_tokens == 5.0
        bucket.acquire(timeout=0.1)
        assert bucket.available_tokens == pytest.approx(4.0, abs=0.2)


# ---------------------------------------------------------------------------
# Retry with backoff
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:

    def test_success_on_first_try(self):
        fn = MagicMock(return_value="ok")
        # Use a fast limiter for tests
        set_limiter(TokenBucket(rate=1000, capacity=100))
        result = _retry_with_backoff(fn, "arg1", key="val")
        assert result == "ok"
        fn.assert_called_once_with("arg1", key="val")

    def test_retry_on_429(self):
        fn = MagicMock(side_effect=[
            Exception("HTTP 429 Too Many Requests"),
            "ok",
        ])
        set_limiter(TokenBucket(rate=1000, capacity=100))

        with patch("dataporter.hf_client._BASE_BACKOFF", 0.01):
            result = _retry_with_backoff(fn)

        assert result == "ok"
        assert fn.call_count == 2

    def test_non_429_error_not_retried(self):
        fn = MagicMock(side_effect=ValueError("bad input"))
        set_limiter(TokenBucket(rate=1000, capacity=100))

        with pytest.raises(ValueError, match="bad input"):
            _retry_with_backoff(fn)
        assert fn.call_count == 1

    def test_max_retries_exhausted(self):
        fn = MagicMock(side_effect=Exception("429 rate limited"))
        set_limiter(TokenBucket(rate=1000, capacity=100))

        with patch("dataporter.hf_client._BASE_BACKOFF", 0.01):
            with patch("dataporter.hf_client._MAX_RETRIES", 3):
                with pytest.raises(Exception, match="429"):
                    _retry_with_backoff(fn)

        assert fn.call_count == 3


# ---------------------------------------------------------------------------
# Shared limiter
# ---------------------------------------------------------------------------

class TestSharedLimiter:

    def test_get_set_limiter(self):
        original = get_limiter()
        custom = TokenBucket(rate=1.0, capacity=1)
        set_limiter(custom)
        assert get_limiter() is custom
        set_limiter(original)  # restore

    def test_concurrent_callers_share_budget(self):
        """Two callers sharing a tight limiter can't exceed the rate."""
        # 5 tokens/sec, capacity 2 — very tight
        limiter = TokenBucket(rate=5.0, capacity=2)
        set_limiter(limiter)

        call_times = []
        lock = threading.Lock()

        def caller():
            for _ in range(3):
                limiter.acquire(timeout=5.0)
                with lock:
                    call_times.append(time.monotonic())

        threads = [threading.Thread(target=caller) for _ in range(2)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # 6 total acquires at 5/sec with burst of 2 should take >= 0.8s
        elapsed = call_times[-1] - t0
        assert elapsed >= 0.5

        # Restore default
        set_limiter(TokenBucket(rate=2.5, capacity=10))


# ---------------------------------------------------------------------------
# _is_rate_limit
# ---------------------------------------------------------------------------

class TestIsRateLimit:

    def test_detects_429(self):
        assert _is_rate_limit(Exception("HTTP 429 Too Many Requests"))
        assert _is_rate_limit(Exception("Rate limit exceeded (429)"))

    def test_ignores_other_errors(self):
        assert not _is_rate_limit(Exception("Connection refused"))
        assert not _is_rate_limit(ValueError("bad input"))


# ---------------------------------------------------------------------------
# make_hf_download_fn
# ---------------------------------------------------------------------------

class TestMakeHfDownloadFn:

    def test_parses_remote_format(self, tmp_path):
        """Download function parses repo_id::path::revision format."""
        calls = []

        def mock_download(**kwargs):
            calls.append(kwargs)
            # Create the file so the caller doesn't fail
            local = tmp_path / kwargs["filename"]
            local.parent.mkdir(parents=True, exist_ok=True)
            local.touch()

        set_limiter(TokenBucket(rate=1000, capacity=100))

        with patch("huggingface_hub.hf_hub_download", mock_download):
            fn = make_hf_download_fn("lerobot/pusht", revision="v2.1")
            fn("lerobot/pusht::data/ep_000.parquet::v2.1", tmp_path / "ep_000.parquet")

        assert len(calls) == 1
        assert calls[0]["repo_id"] == "lerobot/pusht"
        assert calls[0]["filename"] == "data/ep_000.parquet"
        assert calls[0]["revision"] == "v2.1"

    def test_plain_path_uses_closure(self, tmp_path):
        """Plain path (no ::) uses repo_id/revision from closure."""
        calls = []

        def mock_download(**kwargs):
            calls.append(kwargs)
            local = tmp_path / kwargs["filename"]
            local.parent.mkdir(parents=True, exist_ok=True)
            local.touch()

        set_limiter(TokenBucket(rate=1000, capacity=100))

        with patch("huggingface_hub.hf_hub_download", mock_download):
            fn = make_hf_download_fn("my/repo", revision="main")
            fn("some/file.mp4", tmp_path / "file.mp4")

        assert calls[0]["repo_id"] == "my/repo"
        assert calls[0]["filename"] == "some/file.mp4"
        assert calls[0]["revision"] == "main"
