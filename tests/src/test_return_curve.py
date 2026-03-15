"""Tests for src/return_curve.py shared return-curve utilities."""

import numpy as np
import pytest


class TestComputeWeibullReturnPeriods:
    """Tests for compute_weibull_return_periods."""

    def test_basic_descending_scores(self):
        from src.return_curve import compute_weibull_return_periods

        # 4 values sorted descending
        scores = np.array([40.0, 30.0, 20.0, 10.0])
        rp_scores, rp_values = compute_weibull_return_periods(scores)
        # Weibull: RP_i = (N+1)/rank, so (5/1, 5/2, 5/3, 5/4)
        expected_rp = np.array([5.0, 2.5, 5.0 / 3, 1.25])
        np.testing.assert_array_almost_equal(rp_values, expected_rp)
        np.testing.assert_array_equal(rp_scores, scores)

    def test_single_value(self):
        from src.return_curve import compute_weibull_return_periods

        scores = np.array([100.0])
        rp_scores, rp_values = compute_weibull_return_periods(scores)
        np.testing.assert_array_almost_equal(rp_values, [2.0])
        np.testing.assert_array_equal(rp_scores, [100.0])

    def test_empty_array(self):
        from src.return_curve import compute_weibull_return_periods

        scores = np.array([])
        rp_scores, rp_values = compute_weibull_return_periods(scores)
        assert rp_scores.size == 0
        assert rp_values.size == 0


class TestComputeAiresReturnPeriodYears:
    """Tests for compute_aires_return_period_years."""

    def test_basic_probabilities(self):
        from src.return_curve import compute_aires_return_period_years

        proba = np.array([1.0, 0.5, 0.1, 0.0])
        rp = compute_aires_return_period_years(proba)
        assert rp[0] == pytest.approx(1.0)
        assert rp[1] == pytest.approx(2.0)
        assert rp[2] == pytest.approx(10.0)
        assert rp[3] == pytest.approx(1e10)  # zero → cap


class TestEstimateProbaRes:
    """Tests for estimate_proba_res."""

    def test_all_exceed(self):
        from src.return_curve import estimate_proba_res

        scores = np.array([10.0, 20.0, 30.0])
        weights = np.ones(3)
        p = estimate_proba_res(scores, 5.0, weights)
        assert p == pytest.approx(1.0)

    def test_none_exceed(self):
        from src.return_curve import estimate_proba_res

        scores = np.array([1.0, 2.0, 3.0])
        weights = np.ones(3)
        p = estimate_proba_res(scores, 100.0, weights)
        assert p == pytest.approx(0.0)

    def test_partial_exceed_with_weights(self):
        from src.return_curve import estimate_proba_res

        scores = np.array([10.0, 20.0, 30.0])
        weights = np.array([2.0, 2.0, 2.0])
        # scores >= 15 → [False, True, True]
        # inv_weights = [0.5, 0.5, 0.5]
        # mean(inv * indicator) = mean([0, 0.5, 0.5]) = 1/3
        p = estimate_proba_res(scores, 15.0, weights)
        assert p == pytest.approx(1.0 / 3)


class TestLoadDnsScoresFromNpy:
    """Tests for load_dns_scores_from_npy."""

    def test_loads_and_slices(self, tmp_path):
        from src.return_curve import load_dns_scores_from_npy

        data = np.arange(1000, dtype=float)
        path = tmp_path / "scores.npy"
        np.save(path, data)

        full, subset = load_dns_scores_from_npy(path, subset_n=100)
        assert full.shape == (1000,)
        assert subset.shape == (100,)
        np.testing.assert_array_equal(subset, data[:100])

    def test_subset_larger_than_data(self, tmp_path):
        from src.return_curve import load_dns_scores_from_npy

        data = np.arange(50, dtype=float)
        path = tmp_path / "scores.npy"
        np.save(path, data)

        full, subset = load_dns_scores_from_npy(path, subset_n=400)
        assert full.shape == (50,)
        assert subset.shape == (50,)

    def test_missing_file_raises(self, tmp_path):
        from src.return_curve import load_dns_scores_from_npy

        with pytest.raises(FileNotFoundError):
            load_dns_scores_from_npy(tmp_path / "missing.npy")


class TestFormatRegionLabel:
    """Tests for format_region_label."""

    def test_camel_case_split(self):
        from src.return_curve import format_region_label

        assert format_region_label("NorthAtlantic") == "North Atlantic"

    def test_underscore_split(self):
        from src.return_curve import format_region_label

        assert format_region_label("north_atlantic") == "North Atlantic"

    def test_empty(self):
        from src.return_curve import format_region_label

        assert format_region_label("") == "Unknown region"
