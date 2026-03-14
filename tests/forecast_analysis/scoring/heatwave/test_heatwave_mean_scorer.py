"""Tests for HeatwaveMeanScorer."""

import numpy as np
import pytest
import xarray as xr

from forecast_analysis.scoring.base import BlockingScorer
from forecast_analysis.scoring.heatwave.mean_scorer import HeatwaveMeanScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_uniform_tas(value: float, n_time=10, n_lat=5, n_lon=8):
    """Create a uniform temperature DataArray."""
    data = np.full((n_time, n_lat, n_lon), value)
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(n_time),
            "lat": np.linspace(30, 50, n_lat),
            "lon": np.linspace(-100, -80, n_lon),
        },
    )


def _make_gradient_tas(n_time=10, n_lat=5, n_lon=8):
    """Create temperature DataArray with a spatial gradient.

    Values equal lon index so region subsetting can be verified.
    """
    data = np.zeros((n_time, n_lat, n_lon))
    lons = np.linspace(-100, -80, n_lon)
    for j in range(n_lon):
        data[:, :, j] = lons[j]
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(n_time),
            "lat": np.linspace(30, 50, n_lat),
            "lon": lons,
        },
    )


REGION_FULL = {
    "lon_min": -100,
    "lon_max": -80,
    "lat_min": 30,
    "lat_max": 50,
}


# ---------------------------------------------------------------------------
# Class attribute tests
# ---------------------------------------------------------------------------

class TestHeatwaveMeanScorerAttributes:
    def test_is_subclass_of_blocking_scorer(self):
        assert issubclass(HeatwaveMeanScorer, BlockingScorer)

    def test_name(self):
        assert HeatwaveMeanScorer.name == "HeatwaveMeanScorer"

    def test_requires_blocking_detection_false(self):
        assert HeatwaveMeanScorer.requires_blocking_detection is False

    def test_requires_anomaly_false(self):
        assert HeatwaveMeanScorer.requires_anomaly is False

    def test_required_variable_tas(self):
        assert HeatwaveMeanScorer.required_variable == "tas"

    def test_allowed_regions(self):
        assert HeatwaveMeanScorer.allowed_regions == ("Chicago", "France")

    def test_default_n_days(self):
        scorer = HeatwaveMeanScorer()
        assert scorer.n_days == 7

    def test_custom_n_days(self):
        scorer = HeatwaveMeanScorer(n_days=14)
        assert scorer.n_days == 14


# ---------------------------------------------------------------------------
# ABC compliance
# ---------------------------------------------------------------------------

class TestABCCompliance:
    def test_compute_event_scores_raises_not_implemented(self):
        scorer = HeatwaveMeanScorer()
        with pytest.raises(NotImplementedError, match="event-based scoring"):
            scorer.compute_event_scores(z500=None, event_info={})

    def test_has_compute_score_from_field(self):
        scorer = HeatwaveMeanScorer()
        assert callable(getattr(scorer, "compute_score_from_field", None))


# ---------------------------------------------------------------------------
# Uniform data tests
# ---------------------------------------------------------------------------

class TestUniformData:
    def test_uniform_300K(self):
        tas = _make_uniform_tas(300.0)
        scorer = HeatwaveMeanScorer(n_days=5)
        score = scorer.compute_score_from_field(tas, onset_time_idx=0, region_bounds=REGION_FULL)
        assert score == pytest.approx(300.0)

    def test_uniform_different_value(self):
        tas = _make_uniform_tas(280.0)
        scorer = HeatwaveMeanScorer(n_days=3)
        score = scorer.compute_score_from_field(tas, onset_time_idx=2, region_bounds=REGION_FULL)
        assert score == pytest.approx(280.0)


# ---------------------------------------------------------------------------
# Time windowing tests
# ---------------------------------------------------------------------------

class TestTimeWindowing:
    def test_onset_offset(self):
        """Window starting at onset_time_idx=3, n_days=4 should only use times 3-6."""
        n_time = 10
        data = np.zeros((n_time, 3, 3))
        # Put distinct values in time so we can verify which times are used
        for t in range(n_time):
            data[t, :, :] = float(t)
        tas = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(n_time),
                "lat": np.array([40.0, 42.0, 44.0]),
                "lon": np.array([-90.0, -88.0, -86.0]),
            },
        )
        region = {"lon_min": -90, "lon_max": -86, "lat_min": 40, "lat_max": 44}
        scorer = HeatwaveMeanScorer(n_days=4)
        score = scorer.compute_score_from_field(tas, onset_time_idx=3, region_bounds=region)
        # Mean of times 3, 4, 5, 6 = (3+4+5+6)/4 = 4.5
        assert score == pytest.approx(4.5)

    def test_n_days_1(self):
        """Single-day window."""
        data = np.zeros((5, 2, 2))
        for t in range(5):
            data[t, :, :] = 100.0 + t * 10.0
        tas = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(5),
                "lat": np.array([35.0, 40.0]),
                "lon": np.array([-95.0, -90.0]),
            },
        )
        region = {"lon_min": -95, "lon_max": -90, "lat_min": 35, "lat_max": 40}
        scorer = HeatwaveMeanScorer(n_days=1)
        score = scorer.compute_score_from_field(tas, onset_time_idx=2, region_bounds=region)
        assert score == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Region subsetting tests
# ---------------------------------------------------------------------------

class TestRegionSubsetting:
    def test_subset_region(self):
        """Only grid points inside region_bounds contribute to the mean."""
        tas = _make_gradient_tas(n_time=5, n_lat=5, n_lon=8)
        lons = tas.lon.values  # linspace(-100, -80, 8)

        # Restrict to middle lons only
        region = {
            "lon_min": -97,
            "lon_max": -83,
            "lat_min": 30,
            "lat_max": 50,
        }
        scorer = HeatwaveMeanScorer(n_days=5)
        score = scorer.compute_score_from_field(tas, onset_time_idx=0, region_bounds=region)

        # Expected: mean of lons in [-97, -83]
        selected_lons = lons[(lons >= -97) & (lons <= -83)]
        expected = float(np.mean(selected_lons))
        assert score == pytest.approx(expected, abs=1e-6)

    def test_full_region_matches_global_mean(self):
        """Full region bounds should give same result as global mean."""
        tas = _make_gradient_tas(n_time=5, n_lat=5, n_lon=8)
        scorer = HeatwaveMeanScorer(n_days=5)
        score = scorer.compute_score_from_field(tas, onset_time_idx=0, region_bounds=REGION_FULL)
        expected = float(tas.isel(time=slice(0, 5)).mean().values)
        assert score == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# repr test
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self):
        scorer = HeatwaveMeanScorer()
        assert "HeatwaveMeanScorer" in repr(scorer)
