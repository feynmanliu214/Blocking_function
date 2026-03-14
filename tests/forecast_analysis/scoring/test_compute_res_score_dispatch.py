"""Tests for compute_res_score dispatch — heatwave vs blocking paths."""

import numpy as np
import pytest
import xarray as xr

from forecast_analysis.scoring import compute_res_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_uniform_tas(value, n_time=10, n_lat=4, n_lon=6):
    return xr.DataArray(
        np.full((n_time, n_lat, n_lon), value),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(n_time),
            "lat": np.linspace(35, 50, n_lat),
            "lon": np.linspace(-95, -85, n_lon),
        },
    )


REGION = {"lon_min": -95, "lon_max": -85, "lat_min": 35, "lat_max": 50}


# ---------------------------------------------------------------------------
# Heatwave dispatch
# ---------------------------------------------------------------------------

class TestHeatwaveDispatch:
    def test_heatwave_uniform_300K(self):
        tas = _make_uniform_tas(300.0)
        score = compute_res_score(
            scorer_name="HeatwaveMeanScorer",
            scorer_params={"n_days": 5},
            region_bounds=REGION,
            onset_time_idx=0,
            field_data=tas,
        )
        assert score == pytest.approx(300.0)

    def test_heatwave_custom_n_days(self):
        """Time-varying data, check that correct window is used."""
        n_time = 10
        data = np.zeros((n_time, 3, 3))
        for t in range(n_time):
            data[t, :, :] = 280.0 + t
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
        score = compute_res_score(
            scorer_name="HeatwaveMeanScorer",
            scorer_params={"n_days": 3},
            region_bounds=region,
            onset_time_idx=2,
            field_data=tas,
        )
        # times 2, 3, 4 -> values 282, 283, 284 -> mean 283
        assert score == pytest.approx(283.0)

    def test_heatwave_missing_field_data_raises(self):
        with pytest.raises(ValueError, match="requires field_data"):
            compute_res_score(
                scorer_name="HeatwaveMeanScorer",
                scorer_params={},
                region_bounds=REGION,
                onset_time_idx=0,
                # field_data omitted
            )


# ---------------------------------------------------------------------------
# Blocking scorers should still work (smoke test via unknown scorer error)
# ---------------------------------------------------------------------------

class TestBlockingScorerDispatchDoesNotRegress:
    def test_unknown_scorer_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            compute_res_score(
                scorer_name="TotallyFakeScorer",
                scorer_params={},
                z500_anom=None,
                event_info={},
                region_bounds=REGION,
            )

    def test_heatwave_does_not_reach_anomaly_path(self):
        """HeatwaveMeanScorer should dispatch via field_data, never touching z500_anom."""
        tas = _make_uniform_tas(290.0)
        # z500_anom is None — if anomaly path is reached this would fail
        score = compute_res_score(
            scorer_name="HeatwaveMeanScorer",
            scorer_params={"n_days": 5},
            z500_anom=None,
            event_info=None,
            region_bounds=REGION,
            onset_time_idx=0,
            field_data=tas,
        )
        assert score == pytest.approx(290.0)
