#!/usr/bin/env python3
"""
Shared Scoring Pipeline for RES Forecast Analysis.

Extracts the duplicated scoring pipeline from PFS.py and PanguPlasimFS.py
into reusable functions.  Used by PFS.py, PanguPlasimFS.py, and
compute_plasim_scores.py.

The pipeline stages are:

1. ``extract_variable``  -- pull z500 (with plev selection) or tas from a dataset
2. ``prepare_daily_field`` -- resample to daily, optionally filter NH, standardize
3. ``score_single_member`` -- full scoring pipeline for one ensemble member

Author: AI-RES Project
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Ensure src/ is importable (mirrors the pattern in data_loading.py)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---- public API -----------------------------------------------------------

def extract_variable(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """Extract z500 (with plev selection) or tas from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        An opened xarray Dataset.
    variable : str
        ``"z500"`` or ``"tas"``.

    Returns
    -------
    xr.DataArray
        The extracted field.  For z500 the 500 hPa level is already
        selected (if a ``plev`` dimension exists).

    Raises
    ------
    KeyError
        If the expected variable names are not found in *ds*.
    ValueError
        If *variable* is not ``"z500"`` or ``"tas"``.
    """
    if variable == "z500":
        return _extract_z500(ds)
    elif variable == "tas":
        return _extract_tas(ds)
    else:
        raise ValueError(
            f"Unsupported variable '{variable}'. Expected 'z500' or 'tas'."
        )


def prepare_daily_field(
    field: xr.DataArray,
    filter_nh: bool = True,
) -> xr.DataArray:
    """Resample to daily mean, optionally filter NH, standardize coords.

    Parameters
    ----------
    field : xr.DataArray
        Sub-daily field with at least dimensions ``(time, lat, lon)``.
    filter_nh : bool
        If ``True`` (the default), keep only Northern-Hemisphere latitudes
        (lat >= 0).  Set ``False`` for tas (heatwave scoring).

    Returns
    -------
    xr.DataArray
        Daily-mean field with standardized coordinates.
    """
    from Process import standardize_coordinates

    daily = field.resample(time="1D").mean()
    daily.name = field.name or "field"

    if filter_nh:
        daily = daily.sel(lat=(daily.lat >= 0))

    daily = standardize_coordinates(daily)
    return daily


def score_single_member(
    field_daily: xr.DataArray,
    scorer_name: str,
    scorer_params: dict,
    variable: str,
    region_bounds: dict,
    onset_time_idx: int,
    z500_clim: Optional[xr.DataArray] = None,
    threshold_90: Optional[Dict[int, float]] = None,
) -> float:
    """Full scoring pipeline for one ensemble member.

    1. Validates the scorer/variable combination.
    2. Checks whether the scorer requires anomalies:

       * **Anomaly path** (z500-based scorers):
         compute anomalies with external climatology, optionally run
         blocking detection, then call ``compute_res_score(z500_anom=...)``.
       * **Raw-field path** (e.g. ``HeatwaveMeanScorer``):
         call ``compute_res_score(field_data=...)``.

    Parameters
    ----------
    field_daily : xr.DataArray
        Daily-averaged field produced by :func:`prepare_daily_field`.
    scorer_name : str
        Scorer name accepted by the registry (e.g. ``"ANOScorer"``,
        ``"GridpointIntensityScorer"``, ``"HeatwaveMeanScorer"``).
    scorer_params : dict
        Parameters forwarded to the scorer constructor.
    variable : str
        ``"z500"`` or ``"tas"``.
    region_bounds : dict
        Bounding box ``{lon_min, lon_max, lat_min, lat_max}``.
    onset_time_idx : int
        Daily time index for blocking/event onset.
    z500_clim : xr.DataArray, optional
        Day-of-year climatology for anomaly computation.
        Required when the scorer needs anomalies.
    threshold_90 : dict, optional
        Monthly 90th-percentile thresholds for blocking detection.
        Required when the scorer needs blocking detection.

    Returns
    -------
    float
        Scalar score for this member.

    Raises
    ------
    ValueError
        On scorer/variable mismatch.
    """
    from forecast_analysis.scoring import (
        compute_res_score,
        validate_scorer_variable,
        scorer_requires_anomaly,
        scorer_requires_blocking_detection,
    )

    # ---- 1. Fail-fast validation ------------------------------------------
    validate_scorer_variable(scorer_name, variable)

    # ---- 2. Dispatch -------------------------------------------------------
    if not scorer_requires_anomaly(scorer_name):
        # Raw-field path (e.g. HeatwaveMeanScorer)
        return compute_res_score(
            scorer_name=scorer_name,
            scorer_params=scorer_params,
            region_bounds=region_bounds,
            onset_time_idx=onset_time_idx,
            field_data=field_daily,
        )

    # ---- Anomaly path ------------------------------------------------------
    if z500_clim is None:
        raise ValueError(
            f"Scorer '{scorer_name}' requires z500_clim for anomaly computation, "
            "but z500_clim=None was provided."
        )

    from forecast_analysis.data_loading import compute_anomalies_with_climatology

    z500_anom = compute_anomalies_with_climatology(field_daily, z500_clim)

    # Blocking detection (if required by scorer)
    if scorer_requires_blocking_detection(scorer_name):
        if threshold_90 is None:
            raise ValueError(
                f"Scorer '{scorer_name}' requires threshold_90 for blocking detection, "
                "but threshold_90=None was provided."
            )
        from ANO_PlaSim import create_blocking_mask_fast, identify_blocking_events

        blocked_mask = create_blocking_mask_fast(z500_anom, threshold_90)
        _, event_info = identify_blocking_events(blocked_mask)
    else:
        event_info = {}

    return compute_res_score(
        scorer_name=scorer_name,
        scorer_params=scorer_params,
        z500_anom=z500_anom,
        event_info=event_info,
        region_bounds=region_bounds,
        onset_time_idx=onset_time_idx,
        threshold_90=threshold_90,
    )


# ---- private helpers -------------------------------------------------------

def _extract_z500(ds: xr.Dataset) -> xr.DataArray:
    """Extract z500 with plev selection, matching PFS.py logic."""
    if "zg" in ds:
        zg = ds["zg"]
    elif "z500" in ds:
        zg = ds["z500"]
    else:
        raise KeyError(
            "Neither 'zg' nor 'z500' found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    # Select 500 hPa level if plev dimension exists
    if "plev" in zg.dims:
        p = np.asarray(ds["plev"].values, dtype=float)
        target_p = 50000.0 if p.max() > 10000 else 500.0
        if np.isclose(p, target_p).any():
            zg = zg.sel(plev=target_p)
        else:
            nearest_p = p[np.argmin(np.abs(p - target_p))]
            zg = zg.sel(plev=nearest_p)

    zg.name = "z500"
    return zg


def _extract_tas(ds: xr.Dataset) -> xr.DataArray:
    """Extract near-surface air temperature from a dataset."""
    if "tas" in ds:
        result = ds["tas"]
    elif "t2m" in ds:
        result = ds["t2m"]
    else:
        raise KeyError(
            "Neither 'tas' nor 't2m' found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )
    result.name = "tas"
    return result


__all__ = [
    "extract_variable",
    "prepare_daily_field",
    "score_single_member",
]
