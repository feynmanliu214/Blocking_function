#!/usr/bin/env python3
"""
Tests for score_from_anomaly() on anomaly-path scorer classes.

Covers:
- BlockingScorer base raises NotImplementedError
- ANOScorer.score_from_anomaly returns float
- GridpointPersistenceScorer.score_from_anomaly delegates correctly
- GridpointIntensityScorer.score_from_anomaly delegates correctly
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forecast_analysis.scoring.base import BlockingScorer
from forecast_analysis.scoring.ano.scorer import ANOScorer
from forecast_analysis.scoring.gridpoint.persistence_scorer import GridpointPersistenceScorer
from forecast_analysis.scoring.gridpoint.intensity_scorer import GridpointIntensityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anom(n_time=10, nlat=8, nlon=8, value=500.0):
    """Create a minimal synthetic Z500 anomaly DataArray."""
    times = pd.date_range("2000-12-01", periods=n_time, freq="D")
    lats = np.linspace(55.0, 75.0, nlat)
    lons = np.linspace(-60.0, 0.0, nlon)
    data = np.full((n_time, nlat, nlon), value, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _make_region_bounds():
    return {
        "lon_min": -60.0,
        "lon_max": 0.0,
        "lat_min": 55.0,
        "lat_max": 75.0,
    }


# Module-level helpers used by multiple tests
BOUNDS = _make_region_bounds()
_make_z500_anom = _make_anom


def _make_threshold_90(value=400.0):
    """Monthly threshold dict: all months set to value."""
    return {m: value for m in range(1, 13)}


# ---------------------------------------------------------------------------
# Test 1: base class raises NotImplementedError
# ---------------------------------------------------------------------------

class _ConcreteScorer(BlockingScorer):
    """Minimal concrete subclass for testing the base."""
    name = "concrete"
    description = "test"

    def compute_event_scores(self, z500, event_info, **kwargs):
        return pd.DataFrame()


def test_base_score_from_anomaly_raises():
    scorer = _ConcreteScorer()
    z500_anom = _make_anom()
    with pytest.raises(NotImplementedError, match="does not implement score_from_anomaly"):
        scorer.score_from_anomaly(
            z500_anom=z500_anom,
            event_info={},
            region_bounds=_make_region_bounds(),
            onset_time_idx=0,
        )


# ---------------------------------------------------------------------------
# Test 2: ANOScorer.score_from_anomaly returns float
# ---------------------------------------------------------------------------

def test_ano_scorer_score_from_anomaly_returns_float():
    scorer = ANOScorer(mode="auto")
    z500_anom = _make_anom(n_time=10, value=500.0)

    # Provide a blocked_mask so that event_info is complete
    blocked_data = np.ones((10, 8, 8), dtype=float)
    times = z500_anom.time
    lats = z500_anom.lat
    lons = z500_anom.lon
    blocked_mask = xr.DataArray(
        blocked_data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
    )
    event_info = {"blocked_mask": blocked_mask}

    result = scorer.score_from_anomaly(
        z500_anom=z500_anom,
        event_info=event_info,
        region_bounds=_make_region_bounds(),
        onset_time_idx=0,
    )
    assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# Test 3: GridpointPersistenceScorer.score_from_anomaly delegates
# ---------------------------------------------------------------------------

def test_gridpoint_persistence_score_from_anomaly():
    scorer = GridpointPersistenceScorer(min_persistence=5)

    z500_anom = _make_anom(n_time=10, value=500.0)
    threshold_90 = _make_threshold_90(value=400.0)  # all points above threshold

    result = scorer.score_from_anomaly(
        z500_anom=z500_anom,
        event_info={},
        region_bounds=_make_region_bounds(),
        onset_time_idx=0,
        threshold_90=threshold_90,
        scorer_params={"n_days": 5},
    )
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert result >= 0.0


def test_gridpoint_persistence_score_from_anomaly_requires_threshold():
    scorer = GridpointPersistenceScorer(min_persistence=5)

    z500_anom = _make_anom(n_time=10, value=500.0)

    with pytest.raises(ValueError, match="requires threshold_90"):
        scorer.score_from_anomaly(
            z500_anom=z500_anom,
            event_info={},
            region_bounds=_make_region_bounds(),
            onset_time_idx=0,
            threshold_90=None,
        )


# ---------------------------------------------------------------------------
# Test 4: GridpointIntensityScorer.score_from_anomaly delegates
# ---------------------------------------------------------------------------

def test_gridpoint_intensity_score_from_anomaly():
    scorer = GridpointIntensityScorer(min_persistence=5)

    z500_anom = _make_anom(n_time=10, value=500.0)
    threshold_90 = _make_threshold_90(value=400.0)  # all points above threshold

    result = scorer.score_from_anomaly(
        z500_anom=z500_anom,
        event_info={},
        region_bounds=_make_region_bounds(),
        onset_time_idx=0,
        threshold_90=threshold_90,
        scorer_params={"n_days": 5},
    )
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert result >= 0.0


def test_heatwave_scorer_raises_not_implemented():
    """HeatwaveMeanScorer inherits NotImplementedError default from base."""
    from forecast_analysis.scoring import HeatwaveMeanScorer
    scorer = HeatwaveMeanScorer(n_days=7)
    z500_anom = _make_z500_anom()
    with pytest.raises(NotImplementedError):
        scorer.score_from_anomaly(
            z500_anom=z500_anom,
            event_info={},
            region_bounds=BOUNDS,
            onset_time_idx=0,
        )
