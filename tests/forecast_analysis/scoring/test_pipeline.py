"""Tests for the shared scoring pipeline (forecast_analysis.scoring.pipeline).

Covers:
- extract_variable() for z500 and tas
- extract_variable() with unknown variable
- prepare_daily_field() resampling, NH filtering, and standardize_coordinates
- score_single_member() end-to-end for both blocking and heatwave paths
- score_single_member() with mismatched variable raises ValueError
- score_single_member() null guards for z500_clim and threshold_90
"""

from unittest import mock

import numpy as np
import pandas as pd
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

def make_ctx(scorer_name, scorer_params, variable, region_bounds, onset_time_idx):
    from forecast_analysis.scoring.context import ScorerContext, _instantiate_scorer
    scorer = _instantiate_scorer(scorer_name, scorer_params, onset_time_idx, region_bounds)
    return ScorerContext(scorer=scorer, variable=variable, region_bounds=region_bounds,
                         onset_time_idx=onset_time_idx, scorer_params=scorer_params)


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
        assert result.name == "tas"

    def test_extracts_t2m_variant(self):
        """When source variable is 't2m', result.name is normalized to 'tas'."""
        ds = _make_tas_dataset(var_name="t2m")
        result = extract_variable(ds, "tas")

        assert isinstance(result, xr.DataArray)
        assert result.name == "tas"

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
# prepare_daily_field tests
# ===========================================================================


def _make_subdaily_field(n_days=5, steps_per_day=4, include_sh=True):
    """Build a sub-daily DataArray spanning both hemispheres.

    Parameters
    ----------
    n_days : int
        Number of days.
    steps_per_day : int
        Sub-daily time steps per day.
    include_sh : bool
        If True, latitude range spans from -90 to 90 (both hemispheres).
        If False, only Northern Hemisphere latitudes (0 to 90).
    """
    n_time = n_days * steps_per_day
    lat_vals = np.linspace(-90, 90, 37) if include_sh else np.linspace(0, 90, 19)
    lon_vals = np.linspace(0, 350, 36)
    times = pd.date_range("2000-01-01", periods=n_time, freq="6h")

    rng = np.random.RandomState(99)
    data = rng.randn(n_time, len(lat_vals), len(lon_vals)) * 50 + 5500

    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat_vals, "lon": lon_vals},
        name="z500",
    )


class TestPrepareDailyField:
    """Tests for prepare_daily_field()."""

    @staticmethod
    def _mock_process():
        """Return a context manager that mocks Process.standardize_coordinates.

        The function under test does ``from Process import standardize_coordinates``
        at call time.  We inject a fake ``Process`` module into sys.modules so the
        import resolves without needing the real module.

        Returns a (context_manager, mock_fn) tuple so callers can inspect the mock.
        """
        mock_std = mock.MagicMock(side_effect=lambda x: x)
        process_mod = mock.MagicMock()
        process_mod.standardize_coordinates = mock_std
        return mock.patch.dict("sys.modules", {"Process": process_mod}), mock_std

    def test_subdaily_resampled_to_daily(self):
        """Sub-daily data resampled to daily produces expected number of time steps."""
        n_days = 5
        field = _make_subdaily_field(n_days=n_days, steps_per_day=4)

        ctx, _ = self._mock_process()
        with ctx:
            result = prepare_daily_field(field, filter_nh=False)

        assert result.sizes["time"] == n_days

    def test_filter_nh_true_removes_southern_hemisphere(self):
        """filter_nh=True removes Southern Hemisphere latitudes."""
        field = _make_subdaily_field(include_sh=True)

        ctx, _ = self._mock_process()
        with ctx:
            result = prepare_daily_field(field, filter_nh=True)

        # All remaining latitudes should be >= 0
        assert (result.lat.values >= 0).all()
        # We should have fewer latitudes than the input (which includes SH)
        assert result.sizes["lat"] < field.sizes["lat"]

    def test_filter_nh_false_preserves_all_latitudes(self):
        """filter_nh=False preserves all latitudes including Southern Hemisphere."""
        field = _make_subdaily_field(include_sh=True)

        ctx, _ = self._mock_process()
        with ctx:
            result = prepare_daily_field(field, filter_nh=False)

        # All original latitudes should be preserved
        assert result.sizes["lat"] == field.sizes["lat"]
        np.testing.assert_array_equal(result.lat.values, field.lat.values)

    def test_calls_standardize_coordinates(self):
        """prepare_daily_field calls standardize_coordinates (verified via mock)."""
        field = _make_subdaily_field(n_days=3, steps_per_day=4)

        ctx, mock_std = self._mock_process()
        with ctx:
            prepare_daily_field(field, filter_nh=False)

        mock_std.assert_called_once()


# ===========================================================================
# score_single_member tests
# ===========================================================================

REGION = {"lon_min": -10, "lon_max": 30, "lat_min": 55, "lat_max": 75}


class TestScoreSingleMemberBlocking:
    """End-to-end test for the anomaly/blocking path (z500-based scorer)."""

    def test_blocking_path_calls_pipeline(self):
        """score_single_member with a z500-based scorer exercises the full
        anomaly -> blocking detection -> scorer.score_from_anomaly() chain.

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

        ctx = make_ctx(
            scorer_name="GridpointPersistenceScorer",
            scorer_params={"min_persistence": 5},
            variable="z500",
            region_bounds=REGION,
            onset_time_idx=3,
        )

        with (
            mock.patch(
                "forecast_analysis.data_loading.compute_anomalies_with_climatology",
                return_value=fake_anom,
            ) as mock_anom,
            mock.patch.dict("sys.modules", {"ANO_PlaSim": mock.MagicMock()}),
            mock.patch.object(ctx.scorer, "score_from_anomaly", return_value=expected_score) as mock_score,
        ):
            # Configure the ANO_PlaSim mock functions
            import sys as _sys
            ano_mock = _sys.modules["ANO_PlaSim"]
            ano_mock.create_blocking_mask_fast.return_value = fake_blocked
            ano_mock.identify_blocking_events.return_value = (None, fake_event_info)
            score = score_single_member(
                field_daily=field_daily,
                ctx=ctx,
                z500_clim=xr.DataArray([0]),  # dummy
                threshold_90={1: 100.0},       # dummy
            )

        assert score == expected_score
        mock_anom.assert_called_once()
        mock_score.assert_called_once()

    def test_anomaly_scorer_no_blocking_detection(self):
        """ANOScorer with requires_blocking_detection=False should
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

        ctx = make_ctx(
            scorer_name="ANOScorer",
            scorer_params={"mode": "onset", "n_days": 5},
            variable="z500",
            region_bounds=REGION,
            onset_time_idx=2,
        )

        with (
            mock.patch(
                "forecast_analysis.data_loading.compute_anomalies_with_climatology",
                return_value=fake_anom,
            ),
            mock.patch.object(
                type(ctx.scorer), "requires_blocking_detection",
                new_callable=lambda: property(lambda self: False),
            ),
            mock.patch.object(ctx.scorer, "score_from_anomaly", return_value=expected_score) as mock_score,
        ):
            score = score_single_member(
                field_daily=field_daily,
                ctx=ctx,
                z500_clim=xr.DataArray([0]),
                threshold_90={1: 100.0},
            )

        assert score == expected_score
        # Verify that score_from_anomaly was called with event_info={}
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

        region = {"lon": [-95.0, -90.0, -85.0], "lat": [35.0, 42.5, 50.0]}

        expected_score = 300.0

        ctx = make_ctx(
            scorer_name="HeatwaveMeanScorer",
            scorer_params={"n_days": 5},
            variable="tas",
            region_bounds=region,
            onset_time_idx=0,
        )

        with mock.patch.object(
            ctx.scorer, "compute_score_from_field", return_value=expected_score
        ) as mock_score:
            score = score_single_member(
                field_daily=field_daily,
                ctx=ctx,
                # z500_clim and threshold_90 not needed
            )

        assert score == expected_score
        mock_score.assert_called_once()
        call_kwargs = mock_score.call_args.kwargs
        assert "field_data" in call_kwargs
        assert call_kwargs["region_bounds"] == region
        assert call_kwargs["onset_time_idx"] == 0


class TestScoreSingleMemberVariableMismatch:
    """score_single_member with mismatched variable/scorer raises ValueError."""

    def test_z500_scorer_with_tas_variable(self):
        field_daily = xr.DataArray(
            np.ones((5, 3, 3)),
            dims=["time", "lat", "lon"],
        )
        ctx = make_ctx(
            scorer_name="GridpointPersistenceScorer",
            scorer_params={},
            variable="tas",  # mismatch: scorer expects z500
            region_bounds=REGION,
            onset_time_idx=0,
        )
        with pytest.raises(ValueError, match="does not match"):
            score_single_member(
                field_daily=field_daily,
                ctx=ctx,
            )

    def test_tas_scorer_with_z500_variable(self):
        field_daily = xr.DataArray(
            np.ones((5, 3, 3)),
            dims=["time", "lat", "lon"],
        )
        ctx = make_ctx(
            scorer_name="HeatwaveMeanScorer",
            scorer_params={},
            variable="z500",  # mismatch: scorer expects tas
            region_bounds=REGION,
            onset_time_idx=0,
        )
        with pytest.raises(ValueError, match="does not match"):
            score_single_member(
                field_daily=field_daily,
                ctx=ctx,
            )


# ===========================================================================
# Null guard tests (I-4)
# ===========================================================================


class TestScoreSingleMemberNullGuards:
    """score_single_member raises ValueError when required inputs are None."""

    def test_z500_clim_none_raises(self):
        """Anomaly-path scorer with z500_clim=None raises ValueError."""
        field_daily = xr.DataArray(
            np.ones((5, 3, 3)),
            dims=["time", "lat", "lon"],
        )
        ctx = make_ctx(
            scorer_name="GridpointPersistenceScorer",
            scorer_params={"min_persistence": 5},
            variable="z500",
            region_bounds=REGION,
            onset_time_idx=0,
        )
        with pytest.raises(ValueError, match="requires z500_clim"):
            score_single_member(
                field_daily=field_daily,
                ctx=ctx,
                z500_clim=None,
                threshold_90={1: 100.0},
            )

    def test_threshold_90_none_raises_when_blocking_detection_required(self):
        """Blocking-detection scorer with threshold_90=None raises ValueError."""
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

        ctx = make_ctx(
            scorer_name="ANOScorer",
            scorer_params={"mode": "onset", "n_days": 5},
            variable="z500",
            region_bounds=REGION,
            onset_time_idx=2,
        )

        with (
            mock.patch(
                "forecast_analysis.data_loading.compute_anomalies_with_climatology",
                return_value=fake_anom,
            ),
            mock.patch.object(
                type(ctx.scorer), "requires_blocking_detection",
                new_callable=lambda: property(lambda self: True),
            ),
        ):
            with pytest.raises(ValueError, match="requires threshold_90"):
                score_single_member(
                    field_daily=field_daily,
                    ctx=ctx,
                    z500_clim=xr.DataArray([0]),  # non-None so we pass the first guard
                    threshold_90=None,
                )
