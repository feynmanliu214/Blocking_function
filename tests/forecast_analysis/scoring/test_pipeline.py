"""Tests for the shared scoring pipeline (forecast_analysis.scoring.pipeline).

Covers:
- extract_variable() for z500 and tas
- extract_variable() with unknown variable
- score_single_member() end-to-end for both blocking and heatwave paths
- score_single_member() with mismatched variable raises ValueError
"""

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from forecast_analysis.scoring.pipeline import (
    extract_variable,
    prepare_daily_field,
    score_single_member,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_z500_dataset(var_name="zg", with_plev=True):
    """Build a minimal Dataset containing a z500-like variable."""
    n_time, n_lat, n_lon = 8, 5, 10
    rng = np.random.RandomState(42)
    data = rng.randn(n_time, n_lat, n_lon) * 50 + 5500

    coords = {
        "time": np.arange(n_time),
        "lat": np.linspace(20, 80, n_lat),
        "lon": np.linspace(0, 350, n_lon),
    }
    dims = ["time", "lat", "lon"]

    if with_plev:
        # Add a pressure-level dimension with 50000 Pa present
        plev = np.array([85000.0, 50000.0, 25000.0])
        data = np.stack([data, data + 100, data + 200], axis=1)  # (time, plev, lat, lon)
        coords["plev"] = plev
        dims = ["time", "plev", "lat", "lon"]

    da = xr.DataArray(data, dims=dims, coords=coords, name=var_name)
    return xr.Dataset({var_name: da})


def _make_tas_dataset(var_name="tas"):
    """Build a minimal Dataset containing a temperature variable."""
    n_time, n_lat, n_lon = 10, 4, 6
    data = np.full((n_time, n_lat, n_lon), 290.0)
    ds = xr.Dataset(
        {
            var_name: xr.DataArray(
                data,
                dims=["time", "lat", "lon"],
                coords={
                    "time": np.arange(n_time),
                    "lat": np.linspace(35, 50, n_lat),
                    "lon": np.linspace(-95, -85, n_lon),
                },
            )
        }
    )
    return ds


# ===========================================================================
# extract_variable tests
# ===========================================================================


class TestExtractVariableZ500:
    def test_extracts_zg_with_plev_selection(self):
        ds = _make_z500_dataset(var_name="zg", with_plev=True)
        result = extract_variable(ds, "z500")

        assert isinstance(result, xr.DataArray)
        assert "plev" not in result.dims
        assert result.name == "z500"

    def test_extracts_z500_name_variant(self):
        ds = _make_z500_dataset(var_name="z500", with_plev=False)
        result = extract_variable(ds, "z500")

        assert isinstance(result, xr.DataArray)
        assert result.name == "z500"

    def test_plev_in_hPa(self):
        """When plev values are in hPa (e.g. 500), target_p should be 500."""
        ds = _make_z500_dataset(var_name="zg", with_plev=True)
        # Override plev to be in hPa
        ds = ds.assign_coords(plev=np.array([850.0, 500.0, 250.0]))
        result = extract_variable(ds, "z500")

        assert "plev" not in result.dims

    def test_missing_z500_variable_raises(self):
        ds = xr.Dataset({"temperature": xr.DataArray([1, 2, 3])})
        with pytest.raises(KeyError, match="Neither 'zg' nor 'z500'"):
            extract_variable(ds, "z500")


class TestExtractVariableTas:
    def test_extracts_tas(self):
        ds = _make_tas_dataset(var_name="tas")
        result = extract_variable(ds, "tas")

        assert isinstance(result, xr.DataArray)

    def test_extracts_t2m_variant(self):
        ds = _make_tas_dataset(var_name="t2m")
        result = extract_variable(ds, "tas")

        assert isinstance(result, xr.DataArray)

    def test_missing_tas_variable_raises(self):
        ds = xr.Dataset({"wind_speed": xr.DataArray([1, 2, 3])})
        with pytest.raises(KeyError, match="Neither 'tas' nor 't2m'"):
            extract_variable(ds, "tas")


class TestExtractVariableUnknown:
    def test_unsupported_variable_raises(self):
        ds = _make_tas_dataset()
        with pytest.raises(ValueError, match="Unsupported variable"):
            extract_variable(ds, "wind_speed")

    def test_unsupported_variable_message(self):
        ds = _make_tas_dataset()
        with pytest.raises(ValueError, match="'wind_speed'"):
            extract_variable(ds, "wind_speed")


# ===========================================================================
# score_single_member tests
# ===========================================================================

REGION = {"lon_min": -10, "lon_max": 30, "lat_min": 55, "lat_max": 75}


class TestScoreSingleMemberBlocking:
    """End-to-end test for the anomaly/blocking path (z500-based scorer)."""

    def test_blocking_path_calls_pipeline(self):
        """score_single_member with a z500-based scorer exercises the full
        anomaly -> blocking detection -> compute_res_score chain.

        We mock the heavy ANO_PlaSim functions and compute_anomalies to keep
        the test fast and self-contained.
        """
        n_time, n_lat, n_lon = 15, 10, 20
        rng = np.random.RandomState(0)
        field_daily = xr.DataArray(
            rng.randn(n_time, n_lat, n_lon) * 50 + 5500,
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(n_time),
                "lat": np.linspace(30, 80, n_lat),
                "lon": np.linspace(0, 350, n_lon),
            },
            name="z500",
        )

        fake_anom = field_daily.copy()
        fake_anom.name = "z500_anomaly"

        fake_blocked = xr.DataArray(
            np.zeros((n_time, n_lat, n_lon)),
            dims=["time", "lat", "lon"],
            coords=field_daily.coords,
        )
        fake_event_info = {"blocked_mask": fake_blocked}

        expected_score = 42.0

        with (
            mock.patch(
                "forecast_analysis.data_loading.compute_anomalies_with_climatology",
                return_value=fake_anom,
            ) as mock_anom,
            mock.patch.dict("sys.modules", {"ANO_PlaSim": mock.MagicMock()}),
            mock.patch(
                "forecast_analysis.scoring.compute_res_score",
                return_value=expected_score,
            ) as mock_score,
        ):
            # Configure the ANO_PlaSim mock functions
            import sys as _sys
            ano_mock = _sys.modules["ANO_PlaSim"]
            ano_mock.create_blocking_mask_fast.return_value = fake_blocked
            ano_mock.identify_blocking_events.return_value = (None, fake_event_info)
            score = score_single_member(
                field_daily=field_daily,
                scorer_name="GridpointPersistenceScorer",
                scorer_params={"min_persistence": 5},
                variable="z500",
                region_bounds=REGION,
                onset_time_idx=3,
                z500_clim=xr.DataArray([0]),  # dummy
                threshold_90={1: 100.0},       # dummy
            )

        assert score == expected_score
        mock_anom.assert_called_once()
        mock_score.assert_called_once()

    def test_anomaly_scorer_no_blocking_detection(self):
        """ANOScorer with scorer_requires_blocking_detection=False should
        skip the blocking detection step.
        """
        n_time, n_lat, n_lon = 10, 5, 8
        field_daily = xr.DataArray(
            np.ones((n_time, n_lat, n_lon)) * 5500,
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(n_time),
                "lat": np.linspace(30, 80, n_lat),
                "lon": np.linspace(0, 350, n_lon),
            },
            name="z500",
        )

        fake_anom = field_daily.copy()
        expected_score = 7.5

        with (
            mock.patch(
                "forecast_analysis.data_loading.compute_anomalies_with_climatology",
                return_value=fake_anom,
            ),
            mock.patch(
                "forecast_analysis.scoring.scorer_requires_blocking_detection",
                return_value=False,
            ),
            mock.patch(
                "forecast_analysis.scoring.compute_res_score",
                return_value=expected_score,
            ) as mock_score,
        ):
            score = score_single_member(
                field_daily=field_daily,
                scorer_name="ANOScorer",
                scorer_params={"mode": "onset", "n_days": 5},
                variable="z500",
                region_bounds=REGION,
                onset_time_idx=2,
                z500_clim=xr.DataArray([0]),
                threshold_90={1: 100.0},
            )

        assert score == expected_score
        # Verify that blocking detection was NOT called — the event_info
        # should be an empty dict.
        call_kwargs = mock_score.call_args.kwargs
        assert call_kwargs["event_info"] == {}


class TestScoreSingleMemberHeatwave:
    """End-to-end test for the raw-field / heatwave path."""

    def test_heatwave_path(self):
        """HeatwaveMeanScorer should dispatch to the raw-field path."""
        n_time, n_lat, n_lon = 10, 4, 6
        field_daily = xr.DataArray(
            np.full((n_time, n_lat, n_lon), 300.0),
            dims=["time", "lat", "lon"],
            coords={
                "time": np.arange(n_time),
                "lat": np.linspace(35, 50, n_lat),
                "lon": np.linspace(-95, -85, n_lon),
            },
            name="tas",
        )

        region = {"lon_min": -95, "lon_max": -85, "lat_min": 35, "lat_max": 50}

        expected_score = 300.0

        with mock.patch(
            "forecast_analysis.scoring.compute_res_score",
            return_value=expected_score,
        ) as mock_score:
            score = score_single_member(
                field_daily=field_daily,
                scorer_name="HeatwaveMeanScorer",
                scorer_params={"n_days": 5},
                variable="tas",
                region_bounds=region,
                onset_time_idx=0,
                # z500_clim and threshold_90 not needed
            )

        assert score == expected_score
        mock_score.assert_called_once()
        call_kwargs = mock_score.call_args.kwargs
        assert "field_data" in call_kwargs
        # The anomaly path should NOT be invoked — no z500_anom in kwargs
        assert call_kwargs.get("z500_anom") is None or "z500_anom" not in call_kwargs


class TestScoreSingleMemberVariableMismatch:
    """score_single_member with mismatched variable/scorer raises ValueError."""

    def test_z500_scorer_with_tas_variable(self):
        field_daily = xr.DataArray(
            np.ones((5, 3, 3)),
            dims=["time", "lat", "lon"],
        )
        with pytest.raises(ValueError, match="does not match"):
            score_single_member(
                field_daily=field_daily,
                scorer_name="GridpointPersistenceScorer",
                scorer_params={},
                variable="tas",  # mismatch: scorer expects z500
                region_bounds=REGION,
                onset_time_idx=0,
            )

    def test_tas_scorer_with_z500_variable(self):
        field_daily = xr.DataArray(
            np.ones((5, 3, 3)),
            dims=["time", "lat", "lon"],
        )
        with pytest.raises(ValueError, match="does not match"):
            score_single_member(
                field_daily=field_daily,
                scorer_name="HeatwaveMeanScorer",
                scorer_params={},
                variable="z500",  # mismatch: scorer expects tas
                region_bounds=REGION,
                onset_time_idx=0,
            )
