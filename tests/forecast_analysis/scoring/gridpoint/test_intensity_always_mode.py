"""Tests for fallback_to_nonblocked='always' mode in GridpointIntensityScorer."""
import numpy as np
import pytest

from forecast_analysis.scoring.gridpoint.intensity_scorer import (
    GridpointIntensityScorer,
)


class TestComputeWindowMaxIntensityAlways:
    """Unit tests for _compute_window_max_intensity with 'always' mode."""

    def _make_data(self, n_time=5, n_lat=3, n_lon=3, fill=1.0):
        z500_anom = np.full((n_time, n_lat, n_lon), fill)
        above_threshold = np.zeros((n_time, n_lat, n_lon), dtype=bool)
        region_mask = np.ones((n_lat, n_lon), dtype=bool)
        return z500_anom, above_threshold, region_mask

    def test_always_returns_max_anomaly_ignoring_threshold(self):
        """With 'always', result should be max regional anomaly even when
        no grid point exceeds the threshold."""
        z500_anom, above_threshold, region_mask = self._make_data(fill=42.0)
        # above_threshold is all False, so with False mode we'd get 0
        result_false = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, above_threshold, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked=False,
        )
        result_always = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, above_threshold, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked="always",
        )
        assert result_false == 0.0
        assert result_always == 42.0

    def test_always_ignores_above_threshold_even_when_some_blocked(self):
        """With 'always', blocked points are irrelevant — uses full region."""
        z500_anom, above_threshold, region_mask = self._make_data(fill=10.0)
        # Make one point have a higher anomaly but not blocked
        z500_anom[:, 0, 0] = 99.0
        # Block a different point with lower anomaly
        above_threshold[:, 1, 1] = True

        result_always = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, above_threshold, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked="always",
        )
        # Should pick up the 99.0 point, not the 10.0 blocked point
        assert result_always == 99.0

    def test_always_with_none_above_threshold(self):
        """With 'always', above_threshold_np can be None."""
        z500_anom = np.full((5, 3, 3), 7.0)
        region_mask = np.ones((3, 3), dtype=bool)

        result = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, None, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked="always",
        )
        assert result == 7.0

    def test_always_respects_region_mask(self):
        """With 'always', region_mask still applies."""
        z500_anom = np.full((5, 3, 3), 5.0)
        z500_anom[:, 0, 0] = 100.0  # high anomaly outside region
        region_mask = np.zeros((3, 3), dtype=bool)
        region_mask[1, 1] = True  # only center point in region

        result = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, None, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked="always",
        )
        assert result == 5.0

    def test_always_empty_region_returns_zero(self):
        """With 'always' and empty region mask, returns 0."""
        z500_anom = np.full((5, 3, 3), 50.0)
        region_mask = np.zeros((3, 3), dtype=bool)

        result = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, None, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked="always",
        )
        assert result == 0.0

    def test_bool_false_unchanged(self):
        """Existing False behavior is unchanged."""
        z500_anom, above_threshold, region_mask = self._make_data(fill=10.0)
        result = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, above_threshold, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked=False,
        )
        assert result == 0.0

    def test_bool_true_unchanged(self):
        """Existing True (fallback) behavior is unchanged."""
        z500_anom, above_threshold, region_mask = self._make_data(fill=10.0)
        result = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, above_threshold, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked=True,
        )
        assert result == 10.0

    def test_invalid_string_raises_error(self):
        """Typos like 'alwyas' should raise ValueError, not silently behave
        like True."""
        z500_anom, above_threshold, region_mask = self._make_data(fill=10.0)
        with pytest.raises(ValueError, match="Invalid.*fallback_to_nonblocked"):
            GridpointIntensityScorer._compute_window_max_intensity(
                z500_anom, above_threshold, region_mask,
                start_idx=0, end_idx=5, window_days=5,
                fallback_to_nonblocked="alwyas",
            )

    @pytest.mark.parametrize("value", ["Always", "ALWAYS", " always "])
    def test_case_insensitive_always(self, value):
        """Case-insensitive and whitespace-trimmed 'always' should work."""
        z500_anom, above_threshold, region_mask = self._make_data(fill=42.0)
        result = GridpointIntensityScorer._compute_window_max_intensity(
            z500_anom, above_threshold, region_mask,
            start_idx=0, end_idx=5, window_days=5,
            fallback_to_nonblocked=value,
        )
        assert result == 42.0


class TestApplyMonthlyThresholdSkipped:
    """Verify that apply_monthly_threshold is NOT called in 'always' mode."""

    def test_from_anomalies_skips_threshold(self, monkeypatch):
        """compute_intensity_score_from_anomalies must not call
        apply_monthly_threshold when fallback_to_nonblocked='always'."""
        import forecast_analysis.scoring.gridpoint.intensity_scorer as mod

        call_count = 0
        original = mod.apply_monthly_threshold

        def spy(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(mod, "apply_monthly_threshold", spy)

        scorer = GridpointIntensityScorer(min_persistence=5)

        # Build minimal xarray inputs
        import xarray as xr
        times = np.arange(
            np.datetime64("2000-01-01"), np.datetime64("2000-01-11"),
            np.timedelta64(1, "D"),
        )
        lats = np.array([50.0, 55.0, 60.0])
        lons = np.array([0.0, 5.0, 10.0])
        data = np.random.rand(len(times), len(lats), len(lons))
        z500_anom = xr.DataArray(
            data, dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lats, "lon": lons},
        )
        threshold_90 = {1: 0.5}  # January only

        scorer.compute_intensity_score_from_anomalies(
            z500_anom=z500_anom,
            threshold_90=threshold_90,
            onset_time_idx=0,
            duration_days=10,
            region_lon_min=-10.0, region_lon_max=20.0,
            region_lat_min=40.0, region_lat_max=70.0,
            fallback_to_nonblocked="always",
        )
        assert call_count == 0, (
            "apply_monthly_threshold should not be called in 'always' mode"
        )
