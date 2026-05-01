"""Benchmarks for evaluating shuffle/mixing quality across data loading strategies.

Compares:
  - GlobalShuffle: current architecture — shuffle all indices, random access
  - PoolSampler(N): hold N shards in memory, sample across them, replace on exhaust
  - Sequential (N=1): read one shard at a time, shuffle within

All tests use procedural data with embedded shard_id per row so we can
measure mixing quality without any I/O or network access.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Procedural shard data
# ---------------------------------------------------------------------------

@dataclass
class Row:
    shard_id: int
    row_id: int


def make_shards(n_shards: int = 50, rows_per_shard: int = 200) -> list[list[Row]]:
    """Create procedural shards with unique (shard_id, row_id) per row."""
    return [
        [Row(shard_id=s, row_id=r) for r in range(rows_per_shard)]
        for s in range(n_shards)
    ]


# ---------------------------------------------------------------------------
# Sampling strategies (pure Python iterators, no I/O)
# ---------------------------------------------------------------------------

def global_shuffle(shards: list[list[Row]], seed: int = 42) -> list[Row]:
    """Baseline: flatten all rows, shuffle globally."""
    rng = random.Random(seed)
    rows = [row for shard in shards for row in shard]
    rng.shuffle(rows)
    return rows


def pool_sampler(shards: list[list[Row]], pool_size: int, seed: int = 42) -> list[Row]:
    """Hold pool_size shards in memory, sample uniformly, replace on exhaust.

    1. Shuffle shard order
    2. Load first pool_size shards into pool
    3. Pick a random shard from pool, pick a random row from it
    4. When a shard is exhausted, replace with next shard from queue
    """
    rng = random.Random(seed)
    shard_order = list(range(len(shards)))
    rng.shuffle(shard_order)

    # Prepare per-shard shuffled row indices
    shard_rows: dict[int, list[int]] = {}
    for sid in shard_order:
        indices = list(range(len(shards[sid])))
        rng.shuffle(indices)
        shard_rows[sid] = indices

    queue = list(shard_order)
    pool: list[int] = []  # shard ids currently loaded
    for _ in range(min(pool_size, len(queue))):
        pool.append(queue.pop(0))

    output: list[Row] = []
    while pool:
        # Pick random shard from pool
        idx = rng.randrange(len(pool))
        sid = pool[idx]

        # Pick next row from this shard
        row_idx = shard_rows[sid].pop()
        output.append(shards[sid][row_idx])

        # If shard exhausted, replace from queue
        if not shard_rows[sid]:
            pool.pop(idx)
            if queue:
                next_sid = queue.pop(0)
                pool.append(next_sid)

    return output


def sequential_sampler(shards: list[list[Row]], seed: int = 42) -> list[Row]:
    """N=1: shuffle shard order, shuffle within each shard, read sequentially."""
    return pool_sampler(shards, pool_size=1, seed=seed)


# ---------------------------------------------------------------------------
# Mixing quality metrics
# ---------------------------------------------------------------------------

def batch_shard_entropy(rows: list[Row], batch_size: int) -> list[float]:
    """Shannon entropy of shard distribution within each batch.

    Higher entropy = better mixing within a single batch.
    Max entropy = log2(batch_size) when every row is from a different shard.
    """
    entropies = []
    for i in range(0, len(rows) - batch_size + 1, batch_size):
        batch = rows[i:i + batch_size]
        counts = Counter(r.shard_id for r in batch)
        total = len(batch)
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
        entropies.append(entropy)
    return entropies


def window_shard_diversity(rows: list[Row], window_size: int) -> list[int]:
    """Number of unique shards in a sliding window of consecutive rows.

    Higher = better inter-shard mixing over time.
    """
    diversities = []
    for i in range(0, len(rows) - window_size + 1, window_size):
        window = rows[i:i + window_size]
        n_unique = len(set(r.shard_id for r in window))
        diversities.append(n_unique)
    return diversities


def shard_autocorrelation(rows: list[Row], max_lag: int = 50) -> list[float]:
    """Autocorrelation of the shard_id sequence at various lags.

    Values near 0 = low correlation (good mixing).
    Values near 1 = consecutive rows come from the same shard (poor mixing).
    """
    shard_ids = np.array([r.shard_id for r in rows], dtype=np.float64)
    shard_ids -= shard_ids.mean()
    var = np.var(shard_ids)
    if var == 0:
        return [1.0] * max_lag

    correlations = []
    n = len(shard_ids)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            correlations.append(0.0)
            continue
        corr = np.mean(shard_ids[:n - lag] * shard_ids[lag:]) / var
        correlations.append(float(corr))
    return correlations


def coverage_uniformity(rows: list[Row], n_shards: int) -> dict:
    """Chi-squared test for uniform shard coverage over the full sequence.

    Returns dict with per-shard counts, chi2 statistic, and p-value.
    """
    from scipy import stats

    counts = Counter(r.shard_id for r in rows)
    observed = np.array([counts.get(s, 0) for s in range(n_shards)], dtype=np.float64)
    expected = np.full(n_shards, len(rows) / n_shards)
    chi2, p_value = stats.chisquare(observed, expected)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "min_count": int(observed.min()),
        "max_count": int(observed.max()),
        "cv": float(np.std(observed) / np.mean(observed)),  # coefficient of variation
    }


def consecutive_same_shard_runs(rows: list[Row]) -> dict:
    """Measure runs of consecutive rows from the same shard.

    Returns mean, max, and distribution of run lengths.
    Good mixing → short runs. Poor mixing → long runs.
    """
    if not rows:
        return {"mean": 0, "max": 0, "median": 0}
    runs = []
    current_run = 1
    for i in range(1, len(rows)):
        if rows[i].shard_id == rows[i - 1].shard_id:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    return {
        "mean": float(np.mean(runs)),
        "max": int(np.max(runs)),
        "median": float(np.median(runs)),
        "p95": float(np.percentile(runs, 95)),
    }


# ---------------------------------------------------------------------------
# Benchmark fixtures
# ---------------------------------------------------------------------------

N_SHARDS = 50
ROWS_PER_SHARD = 200
BATCH_SIZE = 32
TOTAL_ROWS = N_SHARDS * ROWS_PER_SHARD  # 10,000


@pytest.fixture
def shards():
    return make_shards(N_SHARDS, ROWS_PER_SHARD)


@pytest.fixture
def all_strategies(shards):
    """Generate row sequences for all strategies."""
    return {
        "global_shuffle": global_shuffle(shards),
        "pool_N1": pool_sampler(shards, pool_size=1),
        "pool_N2": pool_sampler(shards, pool_size=2),
        "pool_N3": pool_sampler(shards, pool_size=3),
        "pool_N5": pool_sampler(shards, pool_size=5),
        "pool_N10": pool_sampler(shards, pool_size=10),
    }


# ---------------------------------------------------------------------------
# Tests — each asserts a quality threshold and prints comparison table
# ---------------------------------------------------------------------------

class TestBatchEntropy:
    """Batch-level shard diversity: higher entropy = better intra-batch mixing."""

    def test_all_strategies_produce_correct_count(self, all_strategies):
        for name, rows in all_strategies.items():
            assert len(rows) == TOTAL_ROWS, f"{name}: expected {TOTAL_ROWS}, got {len(rows)}"

    def test_entropy_comparison(self, all_strategies, capsys):
        """Print entropy stats and verify global shuffle is best."""
        results = {}
        for name, rows in all_strategies.items():
            entropies = batch_shard_entropy(rows, BATCH_SIZE)
            results[name] = {
                "mean": np.mean(entropies),
                "min": np.min(entropies),
                "std": np.std(entropies),
            }

        # Print comparison table
        with capsys.disabled():
            print("\n\n=== Batch Shard Entropy (higher = better mixing) ===")
            print(f"  Max possible: {math.log2(BATCH_SIZE):.2f} bits")
            print(f"  {'Strategy':<20} {'Mean':>8} {'Min':>8} {'Std':>8}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
            for name in all_strategies:
                r = results[name]
                print(f"  {name:<20} {r['mean']:>8.3f} {r['min']:>8.3f} {r['std']:>8.3f}")

        # Global shuffle should have highest mean entropy
        global_mean = results["global_shuffle"]["mean"]
        for name, r in results.items():
            if name != "global_shuffle":
                assert r["mean"] <= global_mean + 0.01, (
                    f"{name} entropy {r['mean']:.3f} shouldn't exceed global {global_mean:.3f}"
                )

        # Pool N>=2 should be meaningfully better than N=1
        assert results["pool_N2"]["mean"] > results["pool_N1"]["mean"]


class TestWindowDiversity:
    """Sliding window shard diversity: how many unique shards in W consecutive rows."""

    def test_diversity_comparison(self, all_strategies, capsys):
        window = ROWS_PER_SHARD  # 200-row window = 1 shard's worth
        results = {}
        for name, rows in all_strategies.items():
            diversities = window_shard_diversity(rows, window)
            results[name] = {
                "mean": np.mean(diversities),
                "min": int(np.min(diversities)),
                "max": int(np.max(diversities)),
            }

        with capsys.disabled():
            print(f"\n\n=== Window Shard Diversity (window={window} rows) ===")
            print(f"  {'Strategy':<20} {'Mean':>8} {'Min':>8} {'Max':>8}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
            for name in all_strategies:
                r = results[name]
                print(f"  {name:<20} {r['mean']:>8.1f} {r['min']:>8d} {r['max']:>8d}")

        # N=1 should have diversity of 1-2 (mostly one shard per window)
        assert results["pool_N1"]["mean"] < 3.0

        # Global shuffle should have highest diversity
        assert results["global_shuffle"]["mean"] > results["pool_N3"]["mean"]


class TestAutocorrelation:
    """Shard-id autocorrelation: low = good mixing, high = clustered reads."""

    def test_autocorrelation_comparison(self, all_strategies, capsys):
        results = {}
        for name, rows in all_strategies.items():
            corrs = shard_autocorrelation(rows, max_lag=20)
            results[name] = {
                "lag1": corrs[0],
                "lag5": corrs[4],
                "lag10": corrs[9],
                "mean": np.mean(corrs),
            }

        with capsys.disabled():
            print("\n\n=== Shard Autocorrelation (lower = better mixing) ===")
            print(f"  {'Strategy':<20} {'Lag=1':>8} {'Lag=5':>8} {'Lag=10':>8} {'Mean':>8}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for name in all_strategies:
                r = results[name]
                print(f"  {name:<20} {r['lag1']:>8.3f} {r['lag5']:>8.3f} "
                      f"{r['lag10']:>8.3f} {r['mean']:>8.3f}")

        # Global shuffle should have near-zero autocorrelation
        assert abs(results["global_shuffle"]["lag1"]) < 0.05

        # N=1 should have high lag-1 autocorrelation (same shard runs)
        assert results["pool_N1"]["lag1"] > 0.5

        # N>=3 should be significantly less correlated than N=1
        assert results["pool_N3"]["lag1"] < results["pool_N1"]["lag1"]


class TestCoverageUniformity:
    """Full-epoch coverage: every shard should be sampled equally."""

    def test_all_strategies_cover_all_shards(self, all_strategies):
        for name, rows in all_strategies.items():
            shard_ids = set(r.shard_id for r in rows)
            assert len(shard_ids) == N_SHARDS, (
                f"{name}: only covered {len(shard_ids)}/{N_SHARDS} shards"
            )

    def test_uniformity_comparison(self, all_strategies, capsys):
        results = {}
        for name, rows in all_strategies.items():
            results[name] = coverage_uniformity(rows, N_SHARDS)

        with capsys.disabled():
            print("\n\n=== Coverage Uniformity (lower CV = more uniform) ===")
            print(f"  Expected per shard: {TOTAL_ROWS / N_SHARDS:.0f} rows")
            print(f"  {'Strategy':<20} {'CV':>8} {'Min':>8} {'Max':>8} {'Chi2':>10} {'p-val':>8}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
            for name in all_strategies:
                r = results[name]
                print(f"  {name:<20} {r['cv']:>8.4f} {r['min_count']:>8d} {r['max_count']:>8d} "
                      f"{r['chi2']:>10.1f} {r['p_value']:>8.4f}")

        # All strategies should produce exactly uniform coverage
        # (each row appears exactly once per epoch)
        for name, r in results.items():
            assert r["cv"] < 0.01, f"{name}: CV={r['cv']:.4f}, expected ~0 (exact epoch)"


class TestConsecutiveRuns:
    """Run lengths of consecutive same-shard rows: shorter = better mixing."""

    def test_run_length_comparison(self, all_strategies, capsys):
        results = {}
        for name, rows in all_strategies.items():
            results[name] = consecutive_same_shard_runs(rows)

        with capsys.disabled():
            print("\n\n=== Consecutive Same-Shard Run Lengths (shorter = better) ===")
            print(f"  {'Strategy':<20} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for name in all_strategies:
                r = results[name]
                print(f"  {name:<20} {r['mean']:>8.1f} {r['median']:>8.1f} "
                      f"{r['p95']:>8.1f} {r['max']:>8d}")

        # Global shuffle: mean run ~1 (almost every row switches shard)
        assert results["global_shuffle"]["mean"] < 1.5

        # N=1: mean run = rows_per_shard (entire shard read sequentially)
        assert results["pool_N1"]["mean"] > 50

        # N=2 should be roughly half of N=1
        assert results["pool_N2"]["mean"] < results["pool_N1"]["mean"]

        # N=5 should be significantly shorter
        assert results["pool_N5"]["mean"] < results["pool_N2"]["mean"]


class TestScalingBehavior:
    """How does mixing quality scale with pool size?"""

    def test_monotonic_improvement(self, shards, capsys):
        """Entropy and run length should improve monotonically with pool size."""
        pool_sizes = [1, 2, 3, 5, 10, 20, 50]
        entropies = []
        run_lengths = []
        autocorrs = []

        for n in pool_sizes:
            rows = pool_sampler(shards, pool_size=n)
            ent = np.mean(batch_shard_entropy(rows, BATCH_SIZE))
            runs = consecutive_same_shard_runs(rows)
            corrs = shard_autocorrelation(rows, max_lag=5)
            entropies.append(ent)
            run_lengths.append(runs["mean"])
            autocorrs.append(corrs[0])  # lag-1

        # Compare to global shuffle baseline
        baseline = global_shuffle(shards)
        base_ent = np.mean(batch_shard_entropy(baseline, BATCH_SIZE))
        base_runs = consecutive_same_shard_runs(baseline)["mean"]
        base_corr = shard_autocorrelation(baseline, max_lag=5)[0]

        with capsys.disabled():
            print("\n\n=== Mixing Quality vs Pool Size ===")
            print(f"  {'Pool N':<8} {'Entropy':>10} {'Run Mean':>10} {'Autocorr':>10}")
            print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
            for i, n in enumerate(pool_sizes):
                print(f"  {n:<8} {entropies[i]:>10.3f} {run_lengths[i]:>10.1f} "
                      f"{autocorrs[i]:>10.3f}")
            print(f"  {'global':<8} {base_ent:>10.3f} {base_runs:>10.1f} "
                  f"{base_corr:>10.3f}")

        # Entropy should increase monotonically with pool size
        for i in range(1, len(entropies)):
            assert entropies[i] >= entropies[i - 1] - 0.01, (
                f"Entropy decreased from N={pool_sizes[i-1]} to N={pool_sizes[i]}: "
                f"{entropies[i-1]:.3f} -> {entropies[i]:.3f}"
            )

        # Run length should decrease monotonically with pool size
        for i in range(1, len(run_lengths)):
            assert run_lengths[i] <= run_lengths[i - 1] + 1.0, (
                f"Run length increased from N={pool_sizes[i-1]} to N={pool_sizes[i]}: "
                f"{run_lengths[i-1]:.1f} -> {run_lengths[i]:.1f}"
            )

        # N=50 (all shards) should approach global shuffle quality
        assert abs(entropies[-1] - base_ent) < 0.3, (
            f"Pool N=50 entropy {entropies[-1]:.3f} too far from "
            f"global {base_ent:.3f}"
        )
