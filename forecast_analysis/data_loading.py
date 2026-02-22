#!/usr/bin/env python3
"""
Data Loading and Preprocessing for Forecast Blocking Analysis.

This module provides functions for:
- Loading pre-computed climatology and thresholds
- Extracting z500 from forecast files
- Computing anomalies using external climatology

Author: AI-RES Project
"""

import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import xarray as xr

# Add src/ directory to path for importing core modules
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from Process import z500_seasonal_to_daily, standardize_coordinates


def load_climatology_and_thresholds(
    clim_file: str,
    threshold_json_file: str
) -> Tuple[xr.DataArray, Dict[int, float]]:
    """
    Load pre-computed climatology and 90th percentile thresholds.

    The ANO blocking method requires:
    1. Day-of-year climatology (for computing anomalies)
    2. 90th percentile thresholds (for detecting blocking)

    Both must come from long-term data (e.g., 100-year simulation), NOT from
    short ensemble/forecast runs.

    Parameters
    ----------
    clim_file : str
        Path to NetCDF file containing day-of-year climatology (z500_climatology).
        Should have dimensions (dayofyear, lat, lon).
    threshold_json_file : str
        Path to JSON file containing monthly 90th percentile thresholds.
        Expected structure: {"monthly_thresholds": {"1": 147.2, "2": 148.1, ...}}

    Returns
    -------
    z500_clim : xr.DataArray
        Day-of-year climatology for z500.
    threshold_90 : dict
        Dictionary mapping month number (int) to 90th percentile threshold (float).

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    """
    if not os.path.exists(clim_file):
        raise FileNotFoundError(f"Climatology file not found: {clim_file}")
    if not os.path.exists(threshold_json_file):
        raise FileNotFoundError(f"Threshold file not found: {threshold_json_file}")

    # Load climatology
    z500_clim = xr.open_dataarray(clim_file)

    # Load thresholds
    with open(threshold_json_file, 'r') as f:
        threshold_data = json.load(f)

    # Convert string keys to integers
    threshold_90 = {int(k): v for k, v in threshold_data['monthly_thresholds'].items()}

    return z500_clim, threshold_90


def extract_z500_daily(
    file_path: str,
    season: str = 'DJF',
    extend: bool = True,
    northern_hemisphere_only: bool = True,
    full_simulation: bool = False
) -> xr.DataArray:
    """
    Extract z500 from a forecast file and convert from 6-hourly to daily.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF forecast file.
    season : str, optional
        Season to extract ('DJF' or 'JJA'). Default: 'DJF'.
        Ignored if full_simulation=True.
    extend : bool, optional
        If True, extend season by one month on each side.
        DJF becomes NDJFM (Nov-Dec-Jan-Feb-Mar). Default: True.
        Ignored if full_simulation=True.
    northern_hemisphere_only : bool, optional
        If True, filter to Northern Hemisphere (lat >= 0). Default: True.
    full_simulation : bool, optional
        If True, extract the ENTIRE simulation without seasonal filtering.
        This is appropriate for short forecasts. Default: False.

    Returns
    -------
    z500_daily : xr.DataArray
        Daily-averaged z500 with standardized coordinates.
        Shape: (days, lat, lon)
    """
    if full_simulation:
        # Load entire simulation without seasonal filtering
        z500_daily = _load_full_simulation_daily(file_path)
    else:
        # Use existing function from Process.py (filters to season)
        z500_daily = z500_seasonal_to_daily(file_path, season=season, extend=extend)

    # Filter to Northern Hemisphere if requested
    if northern_hemisphere_only:
        z500_daily = z500_daily.sel(lat=(z500_daily.lat >= 0))

    # Standardize coordinates (lat orientation, lon 0-360, cyclic point)
    z500_daily = standardize_coordinates(z500_daily)

    return z500_daily


def _load_full_simulation_daily(file_path: str) -> xr.DataArray:
    """
    Load z500 from a NetCDF file and convert 6-hourly to daily averages.

    This function loads the ENTIRE simulation without any seasonal filtering,
    which is appropriate for short forecast runs.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.

    Returns
    -------
    z500_daily : xr.DataArray
        Daily-averaged z500 with shape (days, lat, lon).
    """
    # Open dataset with cftime-safe decoding
    try:
        from xarray.coders import CFDatetimeCoder
        ds = xr.open_dataset(file_path, decode_times=CFDatetimeCoder(use_cftime=True))
    except Exception:
        try:
            ds = xr.open_dataset(file_path, decode_times=True, use_cftime=True)
        except Exception:
            ds = xr.open_dataset(file_path, decode_times=False)

    # Find z500 variable
    if "zg" in ds:
        zg = ds["zg"]
    elif "z500" in ds:
        zg = ds["z500"]
    else:
        raise KeyError("Neither 'zg' nor 'z500' found in dataset")

    # Handle ensemble_idx dimension (for Pangu-PlaSim emulator output)
    if "ensemble_idx" in zg.dims:
        zg = zg.isel(ensemble_idx=0)

    # Select 500 hPa level if present
    if "plev" in zg.dims:
        p = np.asarray(ds["plev"].values, dtype=float)

        # Detect if pressure is in Pa or hPa
        if p.max() > 10000:
            target_p = 50000.0  # 500 hPa in Pa
        else:
            target_p = 500.0  # 500 hPa

        if np.isclose(p, target_p).any():
            zg500 = zg.sel(plev=target_p)
        else:
            nearest_p = p[np.argmin(np.abs(p - target_p))]
            zg500 = zg.sel(plev=nearest_p)
    else:
        zg500 = zg

    # Convert 6-hourly to daily averages
    n_times = len(zg500.time)

    if n_times >= 4:
        n_days = n_times // 4
        n_times_to_use = n_days * 4

        z500_trimmed = zg500.isel(time=slice(0, n_times_to_use))

        # Reshape and average
        data = z500_trimmed.values
        lat = z500_trimmed.lat.values
        lon = z500_trimmed.lon.values

        # Reshape: (n_times, lat, lon) -> (n_days, 4, lat, lon)
        data_reshaped = data.reshape(n_days, 4, len(lat), len(lon))
        data_daily = data_reshaped.mean(axis=1)

        # Create daily time coordinate (use first timestep of each day)
        time_daily = z500_trimmed.time.values[::4]

        # Create output DataArray
        z500_daily = xr.DataArray(
            data_daily,
            dims=('time', 'lat', 'lon'),
            coords={'time': time_daily, 'lat': lat, 'lon': lon},
            name='z500',
            attrs={
                'units': 'm',
                'long_name': 'Geopotential Height at 500 hPa (daily average)',
                'description': 'Daily averages from 6-hourly data (full simulation)'
            }
        )
    else:
        z500_daily = zg500.rename('z500')

    return z500_daily


def compute_anomalies_with_climatology(
    z500_daily: xr.DataArray,
    z500_clim: xr.DataArray
) -> xr.DataArray:
    """
    Compute anomalies using pre-computed climatology.

    This is the correct approach for short forecasts, where the climatology
    must come from long-term reference data rather than the forecast itself.

    Parameters
    ----------
    z500_daily : xr.DataArray
        Daily z500 from forecast with dimensions (time, lat, lon).
    z500_clim : xr.DataArray
        Pre-computed day-of-year climatology with dimensions (dayofyear, lat, lon).

    Returns
    -------
    z500_anom : xr.DataArray
        Daily anomalies (z500 - climatology) with same shape as input.

    Notes
    -----
    The climatology is interpolated to match the forecast grid if needed.
    Uses explicit indexing to avoid xarray groupby alignment issues.
    """
    # Make a copy to avoid modifying the original
    z500_clim_work = z500_clim.copy()

    # Get the dayofyear coordinate name (might be 'dayofyear' or the dim name)
    doy_dim = z500_clim_work.dims[0]  # First dimension should be dayofyear

    # Interpolate climatology to match forecast grid (lat/lon only)
    z500_clim_interp = z500_clim_work.interp(
        lat=z500_daily.lat,
        lon=z500_daily.lon,
        method='nearest'
    )

    # Get day-of-year for each timestep in the forecast
    forecast_doy = z500_daily.time.dt.dayofyear.values

    # Select climatology for forecast days
    clim_for_forecast = z500_clim_interp.sel({doy_dim: forecast_doy})

    # Align dimensions
    clim_for_forecast = clim_for_forecast.rename({doy_dim: 'time'})
    clim_for_forecast = clim_for_forecast.assign_coords(time=z500_daily.time.values)

    # Compute anomalies
    z500_anom = z500_daily - clim_for_forecast

    # Add metadata
    z500_anom.name = 'z500_anomaly'
    z500_anom.attrs['description'] = 'Daily anomalies using pre-computed climatology'
    z500_anom.attrs['units'] = 'm'

    return z500_anom
