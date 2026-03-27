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
    ctx: "ScorerContext",
    z500_clim: Optional[xr.DataArray] = None,
    threshold_90: Optional[Dict[int, float]] = None,
) -> float:
    """Full scoring pipeline for one ensemble member.

    Uses ctx.scorer object directly for both anomaly and raw-field paths.
    Scorer dispatch is encapsulated in the scorer class via score_from_anomaly()
    or compute_score_from_field().

    Parameters
    ----------
    field_daily : xr.DataArray
        Daily-averaged field from prepare_daily_field().
    ctx : ScorerContext
        Pre-built context from build_scorer_context().
    z500_clim : xr.DataArray, optional
        Climatology for anomaly computation; required if ctx.scorer.requires_anomaly.
    threshold_90 : dict, optional
        Monthly blocking thresholds; required if ctx.scorer.requires_blocking_detection.

    Returns
    -------
    float
        Scalar score for this member.

    Raises
    ------
    ValueError
        On scorer/variable mismatch or missing required inputs.
    """
    # ---- 1. Fail-fast validation ------------------------------------------
    if ctx.scorer.required_variable != ctx.variable:
        raise ValueError(
            f"Variable '{ctx.variable}' does not match scorer's required variable "
            f"'{ctx.scorer.required_variable}'."
        )

    # ---- 2. Non-anomaly path: call scorer object directly ------------------
    if not ctx.scorer.requires_anomaly:
        return ctx.scorer.compute_score_from_field(
            field_data=field_daily,
            onset_time_idx=ctx.onset_time_idx,
            region_bounds=ctx.region_bounds,
        )

    # ---- 3. Anomaly path ---------------------------------------------------
    if z500_clim is None:
        raise ValueError(
            f"Scorer '{type(ctx.scorer).__name__}' requires z500_clim, "
            "but None was provided."
        )

    from forecast_analysis.data_loading import compute_anomalies_with_climatology

    z500_anom = compute_anomalies_with_climatology(field_daily, z500_clim)

    # Blocking detection (if required by scorer)
    if ctx.scorer.requires_blocking_detection:
        if threshold_90 is None:
            raise ValueError(
                f"Scorer '{type(ctx.scorer).__name__}' requires threshold_90, "
                "but None was provided."
            )
        from ANO_PlaSim import create_blocking_mask_fast, identify_blocking_events

        blocked_mask = create_blocking_mask_fast(z500_anom, threshold_90)
        _, event_info = identify_blocking_events(blocked_mask)
    else:
        event_info = {}

    return ctx.scorer.score_from_anomaly(
        z500_anom=z500_anom,
        event_info=event_info,
        region_bounds=ctx.region_bounds,
        onset_time_idx=ctx.onset_time_idx,
        threshold_90=threshold_90,
        scorer_params=ctx.scorer_params,  # carries n_days, fallback_to_nonblocked, etc.
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

    # Select 500 hPa level for datasets that carry a vertical pressure dimension.
    vertical_dim = next((name for name in ("plev", "lev") if name in zg.dims), None)
    if vertical_dim is not None:
        p = np.asarray(ds[vertical_dim].values, dtype=float)
        target_p = 50000.0 if p.max() > 10000 else 500.0
        if np.isclose(p, target_p).any():
            zg = zg.sel({vertical_dim: target_p})
        else:
            nearest_p = p[np.argmin(np.abs(p - target_p))]
            zg = zg.sel({vertical_dim: nearest_p})

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
