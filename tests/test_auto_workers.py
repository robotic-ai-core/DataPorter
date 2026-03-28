"""Tests for auto num_workers resolution."""

from unittest.mock import patch

import pytest

from dataporter.resumable import resolve_num_workers


class TestResolveNumWorkers:

    def test_int_passthrough(self):
        assert resolve_num_workers(0) == 0
        assert resolve_num_workers(4) == 4
        assert resolve_num_workers(8) == 8

    @patch("os.cpu_count", return_value=12)
    def test_auto_12_cores(self, _):
        assert resolve_num_workers("auto") == 2

    @patch("os.cpu_count", return_value=24)
    def test_auto_24_cores(self, _):
        assert resolve_num_workers("auto") == 4

    @patch("os.cpu_count", return_value=64)
    def test_auto_64_cores(self, _):
        assert resolve_num_workers("auto") == 8

    @patch("os.cpu_count", return_value=96)
    def test_auto_96_cores(self, _):
        assert resolve_num_workers("auto") == 12

    @patch("os.cpu_count", return_value=128)
    def test_auto_128_cores(self, _):
        assert resolve_num_workers("auto") == 16

    @patch("os.cpu_count", return_value=None)
    def test_auto_unknown_cores(self, _):
        # Falls back to 4 cores → ceil(4/8) = 1, rounded to 2
        assert resolve_num_workers("auto") == 2

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="must be int or 'auto'"):
            resolve_num_workers("fast")

    def test_auto_always_even(self):
        for cores in [12, 16, 24, 32, 48, 64, 96, 128]:
            with patch("os.cpu_count", return_value=cores):
                n = resolve_num_workers("auto")
                assert n % 2 == 0, f"{cores} cores → {n} workers (not even)"

    def test_auto_at_least_2(self):
        for cores in [1, 2, 4, 6, 8]:
            with patch("os.cpu_count", return_value=cores):
                n = resolve_num_workers("auto")
                assert n >= 2, f"{cores} cores → {n} workers (less than 2)"
