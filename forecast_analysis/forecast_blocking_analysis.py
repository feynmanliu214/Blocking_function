#!/usr/bin/env python3
"""
Forecast Blocking Analysis Module.

This module provides reusable functions to analyze blocking events in climate
model forecast outputs, including:
- Ensemble forecast analysis
- Deterministic forecast analysis across multiple lead times
- Blocking event animation generation
- Blocking score computation and ranking

The module uses pre-computed climatology and thresholds from long-term simulations
to properly detect ANO (Anomaly-based) blocking events, as short forecast runs
cannot provide statistically meaningful climatology.

Usage Examples
--------------
# Analyze ensemble forecast
>>> from forecast_blocking_analysis import analyze_ensemble_forecast, rank_ensemble_scores
>>> results = analyze_ensemble_forecast(
...     ensemble_dir='/path/to/ensemble/',
...     clim_threshold_file='/path/to/climatology.nc',
...     threshold_json_file='/path/to/thresholds.json',
...     output_dir='/path/to/output/',
...     create_animations=True
... )
>>> df_ranked = rank_ensemble_scores(results, threshold_score=300000)

# Analyze deterministic forecast
>>> from forecast_blocking_analysis import analyze_deterministic_forecast
>>> results = analyze_deterministic_forecast(
...     forecast_files=['/path/to/forecast_lead1.nc', ...],
...     clim_threshold_file='/path/to/climatology.nc',
...     threshold_json_file='/path/to/thresholds.json',
...     output_dir='/path/to/output/'
... )

Author: AI-RES Project
"""

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

# Add src/ directory to path for importing core modules
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Import from core modules in src/
from Process import z500_seasonal_to_daily, standardize_coordinates
from ANO_PlaSim import create_blocking_mask, identify_blocking_events
from compute_blocking_scores import compute_blocking_scores


# =============================================================================
# Climatology and Threshold Loading
# =============================================================================

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
    
    Examples
    --------
    >>> z500_clim, threshold_90 = load_climatology_and_thresholds(
    ...     '/path/to/ano_climatology_thresholds.nc',
    ...     '/path/to/ano_thresholds.json'
    ... )
    >>> print(f"Climatology shape: {z500_clim.shape}")
    >>> print(f"December threshold: {threshold_90[12]:.2f} m")
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


# =============================================================================
# Forecast Data Extraction
# =============================================================================

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
    
    Examples
    --------
    >>> # For long climatology runs (filter to season)
    >>> z500 = extract_z500_daily('/path/to/forecast.nc', season='DJF', extend=True)
    >>> 
    >>> # For short forecasts (use entire simulation)
    >>> z500 = extract_z500_daily('/path/to/forecast.nc', full_simulation=True)
    """
    print(f"  [DEBUG] extract_z500_daily: Loading {file_path}")
    print(f"  [DEBUG] full_simulation={full_simulation}, season={season}, extend={extend}")
    
    if full_simulation:
        # Load entire simulation without seasonal filtering
        z500_daily = _load_full_simulation_daily(file_path)
        print(f"  [DEBUG] After _load_full_simulation_daily: shape={z500_daily.shape}")
    else:
        # Use existing function from Process.py (filters to season)
        z500_daily = z500_seasonal_to_daily(file_path, season=season, extend=extend)
        print(f"  [DEBUG] After z500_seasonal_to_daily: shape={z500_daily.shape}")
    
    print(f"  [DEBUG] lon range: {float(z500_daily.lon.min()):.1f} to {float(z500_daily.lon.max()):.1f}")
    
    # Filter to Northern Hemisphere if requested
    if northern_hemisphere_only:
        z500_daily = z500_daily.sel(lat=(z500_daily.lat >= 0))
        print(f"  [DEBUG] After NH filter: shape={z500_daily.shape}")
    
    # Check for duplicates before standardize
    lon_before = z500_daily.lon.values
    print(f"  [DEBUG] Before standardize: lon unique={len(np.unique(lon_before))}, total={len(lon_before)}")
    
    # Standardize coordinates (lat orientation, lon 0-360, cyclic point)
    z500_daily = standardize_coordinates(z500_daily)
    
    # Check for duplicates after standardize
    lon_after = z500_daily.lon.values
    print(f"  [DEBUG] After standardize: shape={z500_daily.shape}, lon unique={len(np.unique(lon_after))}, total={len(lon_after)}")
    print(f"  [DEBUG] lon values: {lon_after[:3]}...{lon_after[-3:]}")
    
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
        # For deterministic runs, ensemble_idx=0; for ensemble, select first member
        zg = zg.isel(ensemble_idx=0)
        print(f"  [DEBUG] Selected ensemble_idx=0, new shape: {zg.shape}")

    # Select 500 hPa level if present
    if "plev" in zg.dims:
        p = np.asarray(ds["plev"].values, dtype=float)

        # Detect if pressure is in Pa or hPa
        # If max plev > 10000, it's likely in Pa (100000 Pa = 1000 hPa)
        # If max plev < 2000, it's likely in hPa (1000 hPa max)
        if p.max() > 10000:
            # Pressure in Pascals - 500 hPa = 50000 Pa
            target_p = 50000.0
            print(f"  [DEBUG] Pressure appears to be in Pa (max={p.max():.0f})")
        else:
            # Pressure in hPa
            target_p = 500.0
            print(f"  [DEBUG] Pressure appears to be in hPa (max={p.max():.0f})")

        if np.isclose(p, target_p).any():
            zg500 = zg.sel(plev=target_p)
            print(f"  [DEBUG] Selected plev={target_p} (500 hPa)")
        else:
            nearest_p = p[np.argmin(np.abs(p - target_p))]
            zg500 = zg.sel(plev=nearest_p)
            print(f"  [DEBUG] Target plev={target_p} not found, using nearest: {nearest_p}")
    else:
        zg500 = zg  # Already 500 hPa or no pressure dimension
    
    # Convert 6-hourly to daily averages
    n_times = len(zg500.time)
    print(f"  [DEBUG] Total timesteps in file: {n_times}")
    
    # Check if data is 6-hourly (4 timesteps per day)
    if n_times >= 4:
        n_days = n_times // 4
        n_times_to_use = n_days * 4
        
        print(f"  [DEBUG] Converting {n_times_to_use} timesteps to {n_days} daily averages")
        
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
        # Data might already be daily or has fewer timesteps
        print(f"  [DEBUG] Data has {n_times} timesteps, assuming already daily")
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
    print("  [DEBUG] compute_anomalies_with_climatology starting...")
    print(f"  [DEBUG] z500_daily shape: {z500_daily.shape}, dims: {z500_daily.dims}")
    print(f"  [DEBUG] z500_daily lat: {z500_daily.lat.values[:3]}...{z500_daily.lat.values[-3:]}")
    print(f"  [DEBUG] z500_daily lon: {z500_daily.lon.values[:3]}...{z500_daily.lon.values[-3:]}")
    print(f"  [DEBUG] z500_daily lon unique: {len(np.unique(z500_daily.lon.values))}, total: {len(z500_daily.lon.values)}")
    
    print(f"  [DEBUG] z500_clim shape: {z500_clim.shape}, dims: {z500_clim.dims}")
    print(f"  [DEBUG] z500_clim coords: {list(z500_clim.coords.keys())}")
    
    # Make a copy to avoid modifying the original
    print("  [DEBUG] Making copy of climatology...")
    z500_clim_work = z500_clim.copy()
    
    # Get the dayofyear coordinate name (might be 'dayofyear' or the dim name)
    doy_dim = z500_clim_work.dims[0]  # First dimension should be dayofyear
    print(f"  [DEBUG] Day-of-year dimension name: {doy_dim}")
    
    # Check for duplicate coordinates in z500_daily
    print(f"  [DEBUG] Checking for duplicate lon in z500_daily...")
    lon_unique = len(np.unique(z500_daily.lon.values))
    lon_total = len(z500_daily.lon.values)
    if lon_unique != lon_total:
        print(f"  [DEBUG] WARNING: z500_daily has duplicate lon! unique={lon_unique}, total={lon_total}")
    
    # Interpolate climatology to match forecast grid (lat/lon only)
    print("  [DEBUG] Interpolating climatology to forecast grid...")
    try:
        z500_clim_interp = z500_clim_work.interp(
            lat=z500_daily.lat, 
            lon=z500_daily.lon, 
            method='nearest'
        )
        print(f"  [DEBUG] z500_clim_interp shape: {z500_clim_interp.shape}")
    except Exception as e:
        print(f"  [DEBUG] ERROR during interp: {e}")
        raise
    
    # Get day-of-year for each timestep in the forecast
    print("  [DEBUG] Getting forecast day-of-year values...")
    forecast_doy = z500_daily.time.dt.dayofyear.values
    print(f"  [DEBUG] forecast_doy: {forecast_doy[:5]}... (len={len(forecast_doy)})")
    
    # Create anomaly array by explicit indexing (avoids groupby alignment issues)
    # Get climatology values for each day's dayofyear
    print(f"  [DEBUG] Selecting climatology for forecast days using dim '{doy_dim}'...")
    try:
        clim_for_forecast = z500_clim_interp.sel({doy_dim: forecast_doy})
        print(f"  [DEBUG] clim_for_forecast shape: {clim_for_forecast.shape}, dims: {clim_for_forecast.dims}")
    except Exception as e:
        print(f"  [DEBUG] ERROR during sel: {e}")
        raise
    
    # Align dimensions - climatology selection gives (time, lat, lon) where 
    # the 'time' dim corresponds to the selected dayofyear values
    # We need to rename and assign the proper time coordinate
    print("  [DEBUG] Renaming dimension and assigning time coords...")
    try:
        clim_for_forecast = clim_for_forecast.rename({doy_dim: 'time'})
        clim_for_forecast = clim_for_forecast.assign_coords(time=z500_daily.time.values)
        print(f"  [DEBUG] After rename/assign: shape={clim_for_forecast.shape}, dims={clim_for_forecast.dims}")
    except Exception as e:
        print(f"  [DEBUG] ERROR during rename/assign_coords: {e}")
        raise
    
    # Compute anomalies
    print("  [DEBUG] Computing anomalies (subtraction)...")
    try:
        z500_anom = z500_daily - clim_for_forecast
        print(f"  [DEBUG] z500_anom shape: {z500_anom.shape}")
    except Exception as e:
        print(f"  [DEBUG] ERROR during subtraction: {e}")
        raise
    
    # Add metadata
    z500_anom.name = 'z500_anomaly'
    z500_anom.attrs['description'] = 'Daily anomalies using pre-computed climatology'
    z500_anom.attrs['units'] = 'm'
    
    print("  [DEBUG] compute_anomalies_with_climatology completed successfully")
    return z500_anom


# =============================================================================
# Single Member/Forecast Processing
# =============================================================================

def process_single_forecast(
    nc_file: str,
    z500_clim: xr.DataArray,
    threshold_90: Dict[int, float],
    season: str = 'DJF',
    extend: bool = True,
    min_area: float = 2e6,
    min_duration: int = 5,
    min_overlap: float = 0.5,
    full_simulation: bool = True,
    restrict_to_core_season: bool = True
) -> Dict:
    """
    Process a single forecast file for blocking detection.
    
    Parameters
    ----------
    nc_file : str
        Path to the forecast NetCDF file.
    z500_clim : xr.DataArray
        Pre-computed day-of-year climatology.
    threshold_90 : dict
        Monthly 90th percentile thresholds {month: threshold}.
    season : str, optional
        Season to extract ('DJF' or 'JJA'). Default: 'DJF'.
        Ignored if full_simulation=True.
    extend : bool, optional
        Extend season by one month on each side. Default: True.
        Ignored if full_simulation=True.
    min_area : float, optional
        Minimum blocking area in km². Default: 2e6.
    min_duration : int, optional
        Minimum blocking duration in days. Default: 5.
    min_overlap : float, optional
        Minimum spatial overlap between consecutive days. Default: 0.5.
    full_simulation : bool, optional
        If True, use entire simulation without seasonal filtering.
        This is appropriate for short forecasts. Default: True.
    restrict_to_core_season : bool, optional
        If True (default), restrict blocking event detection to core season months
        only, even when data includes extended shoulder months for threshold
        computation. This follows the Woollings et al. (2018) methodology:
        - NDJFM data → events detected only in DJF (Dec, Jan, Feb)
        - MJJAS data → events detected only in JJA (Jun, Jul, Aug)
        If False, events are detected in all months present in the data.
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'z500': Daily z500 data
        - 'z500_anom': Anomalies
        - 'blocked_mask': Binary blocking mask
        - 'blocking_freq': Blocking frequency map
        - 'event_info': Event tracking information
        - 'num_events': Number of detected events
        - 'file': Source file path
        
    Raises
    ------
    Exception
        If processing fails, returns dict with 'error' key instead.
    """
    print(f"[DEBUG] process_single_forecast: Starting for {nc_file}")
    try:
        # Extract z500 daily data
        print("[DEBUG] Step 1: Extracting z500 daily data...")
        z500_daily = extract_z500_daily(
            nc_file, season=season, extend=extend, full_simulation=full_simulation
        )
        print(f"[DEBUG] Step 1 complete: z500_daily shape={z500_daily.shape}")
        
        # Compute anomalies using pre-computed climatology
        print("[DEBUG] Step 2: Computing anomalies...")
        z500_anom = compute_anomalies_with_climatology(z500_daily, z500_clim)
        print(f"[DEBUG] Step 2 complete: z500_anom shape={z500_anom.shape}")
        
        # Apply pre-computed thresholds to create blocking mask
        print("[DEBUG] Step 3: Creating blocking mask...")
        blocked_mask = create_blocking_mask(z500_anom, threshold_90)
        print(f"[DEBUG] Step 3 complete: blocked_mask shape={blocked_mask.shape}")
        
        # Step 3.5: Restrict to core season if requested
        # This masks out shoulder months (Nov, Mar for DJF; May, Sep for JJA)
        detection_season_info = "all months in data"
        if restrict_to_core_season:
            months_in_data = set(blocked_mask.time.dt.month.values)
            
            # Determine core season based on months present
            # NDJFM (11,12,1,2,3) -> DJF (12,1,2)
            # MJJAS (5,6,7,8,9) -> JJA (6,7,8)
            if months_in_data == {11, 12, 1, 2, 3} or months_in_data == {12, 1, 2}:
                core_months = {12, 1, 2}
                detection_season_info = "DJF only (Dec, Jan, Feb)"
            elif months_in_data == {5, 6, 7, 8, 9} or months_in_data == {6, 7, 8}:
                core_months = {6, 7, 8}
                detection_season_info = "JJA only (Jun, Jul, Aug)"
            elif 12 in months_in_data or 1 in months_in_data or 2 in months_in_data:
                # Winter-like data, restrict to DJF
                core_months = {12, 1, 2}
                detection_season_info = "DJF only (Dec, Jan, Feb)"
            elif 6 in months_in_data or 7 in months_in_data or 8 in months_in_data:
                # Summer-like data, restrict to JJA
                core_months = {6, 7, 8}
                detection_season_info = "JJA only (Jun, Jul, Aug)"
            else:
                # Unknown season pattern, don't restrict
                core_months = months_in_data
                detection_season_info = "all months (unknown season pattern)"
            
            # Create mask for core season months
            import numpy as np
            core_season_mask = np.isin(blocked_mask.time.dt.month.values, list(core_months))
            
            # Zero out blocking in shoulder months
            blocked_mask_for_events = blocked_mask.copy()
            blocked_mask_for_events.values[~core_season_mask, :, :] = 0
            
            print(f"[DEBUG] Step 3.5: Restricting to core season: {detection_season_info}")
        else:
            blocked_mask_for_events = blocked_mask
        
        # Identify blocking events with spatial/temporal tracking
        print("[DEBUG] Step 4: Identifying blocking events...")
        blocking_freq, event_info = identify_blocking_events(
            blocked_mask_for_events,  # Use filtered mask (core season only if restrict_to_core_season=True)
            min_area=min_area,
            min_duration=min_duration,
            min_overlap=min_overlap
        )
        print(f"[DEBUG] Step 4 complete: num_events={event_info.get('num_events', 0)}")
        
        # Add additional info to event_info for downstream use
        event_info['z500_anom'] = z500_anom
        event_info['blocked_mask'] = blocked_mask_for_events  # Use filtered mask (core season only)
        event_info['blocked_mask_full'] = blocked_mask  # Keep original for reference if needed
        event_info['threshold_90'] = threshold_90
        event_info['restrict_to_core_season'] = restrict_to_core_season
        event_info['detection_season'] = detection_season_info
        
        print("[DEBUG] process_single_forecast: Completed successfully")
        return {
            'z500': z500_daily,
            'z500_anom': z500_anom,
            'blocked_mask': blocked_mask_for_events,  # Use filtered mask (core season only)
            'blocked_mask_full': blocked_mask,  # Keep original for reference if needed
            'blocking_freq': blocking_freq,
            'event_info': event_info,
            'num_events': event_info['num_events'],
            'file': nc_file
        }
        
    except Exception as e:
        import traceback
        print(f"[DEBUG] process_single_forecast: FAILED with error: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'file': nc_file
        }


# =============================================================================
# Animation Generation
# =============================================================================

def create_forecast_animation(
    event_info: Dict,
    output_path: str,
    title_prefix: str = "Forecast",
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    fps: float = 0.5,
    show_diagnostics: bool = True
) -> str:
    """
    Create an animation GIF for forecast blocking analysis.
    
    Parameters
    ----------
    event_info : dict
        Event information dictionary from process_single_forecast().
        Must contain 'z500_anom', 'blocked_mask', 'threshold_90'.
    output_path : str
        Path to save the output GIF file.
    title_prefix : str, optional
        Prefix for the animation title. Default: "Forecast".
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for diagnostic region. Default: 30-100°E.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for diagnostic region. Default: 55-75°N.
    fps : float, optional
        Frames per second (0.5 = 2 seconds per frame). Default: 0.5.
    show_diagnostics : bool, optional
        Whether to show diagnostic panels. Default: True.
    
    Returns
    -------
    output_path : str
        Path to the saved GIF file.
    
    Notes
    -----
    Requires the event_animation module to be available.
    """
    try:
        from event_animation import create_event_animation_gif_fast
    except ImportError:
        print("Warning: event_animation not available. Skipping animation.")
        return None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create full simulation animation
    create_event_animation_gif_fast(
        event_id=320,  # Placeholder event ID for title
        ano_stats=event_info,
        save_path=output_path,
        title_prefix=title_prefix,
        event_start_times=None,  # Not needed for full simulation
        show_diagnostics=show_diagnostics,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        fps=fps,
        animate_full_simulation=True,
    )
    
    return output_path


# =============================================================================
# Ensemble Forecast Analysis
# =============================================================================

def analyze_ensemble_forecast(
    ensemble_dir: str,
    clim_threshold_file: str,
    threshold_json_file: str,
    output_dir: Optional[str] = None,
    file_pattern: str = '*_postproc.nc',
    min_area: float = 2e6,
    min_duration: int = 5,
    min_overlap: float = 0.5,
    create_animations: bool = True,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Analyze all ensemble members in a directory for blocking events.
    
    This function processes the ENTIRE simulation from each ensemble member
    (no seasonal filtering). The climatology and thresholds should be pre-computed
    from a long reference run and provided via the clim_threshold_file and
    threshold_json_file parameters.
    
    Parameters
    ----------
    ensemble_dir : str
        Directory containing ensemble forecast NetCDF files.
    clim_threshold_file : str
        Path to pre-computed climatology NetCDF file.
    threshold_json_file : str
        Path to JSON file with 90th percentile thresholds.
    output_dir : str, optional
        Directory for output animations. If None, no animations are created.
    file_pattern : str, optional
        Glob pattern for finding ensemble files. Default: '*_postproc.nc'.
    min_area : float, optional
        Minimum blocking area in km². Default: 2e6.
    min_duration : int, optional
        Minimum blocking duration in days. Default: 5.
    min_overlap : float, optional
        Minimum spatial overlap fraction. Default: 0.5.
    create_animations : bool, optional
        Whether to create GIF animations. Default: True.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for diagnostic region. Default: 30-100°E.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for diagnostic region. Default: 55-75°N.
    verbose : bool, optional
        Print progress information. Default: True.
    
    Returns
    -------
    ensemble_results : dict
        Dictionary mapping member names to their analysis results.
        Each result contains z500, anomalies, blocking_freq, event_info, etc.
    
    Examples
    --------
    >>> results = analyze_ensemble_forecast(
    ...     ensemble_dir='/path/to/ensemble/',
    ...     clim_threshold_file='/path/to/climatology.nc',
    ...     threshold_json_file='/path/to/thresholds.json',
    ...     output_dir='/path/to/output/',
    ...     create_animations=True
    ... )
    >>> for name, result in results.items():
    ...     if 'error' not in result:
    ...         print(f"{name}: {result['num_events']} events")
    """
    # Load pre-computed climatology and thresholds
    if verbose:
        print("=" * 70)
        print("Loading pre-computed climatology and thresholds")
        print("=" * 70)
    
    z500_clim, threshold_90 = load_climatology_and_thresholds(
        clim_threshold_file, threshold_json_file
    )
    
    if verbose:
        print(f"   ✓ Climatology loaded: shape={z500_clim.shape}")
        print(f"   ✓ Thresholds loaded: {threshold_90}")
    
    # Find ensemble files
    nc_files = sorted(glob.glob(os.path.join(ensemble_dir, file_pattern)))
    if verbose:
        print(f"\nFound {len(nc_files)} ensemble member files")
    
    if len(nc_files) == 0:
        raise ValueError(f"No files found matching pattern '{file_pattern}' in {ensemble_dir}")
    
    # Create output directory if needed
    if output_dir and create_animations:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each ensemble member
    ensemble_results = {}
    
    for i, nc_file in enumerate(nc_files, 1):
        # Extract member name from filename
        member_name = os.path.basename(nc_file).split('_plasim')[0]
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Processing ensemble member {i}/{len(nc_files)}: {member_name}")
            print(f"{'=' * 60}")
        
        # Process this member (always use full simulation - no seasonal filtering)
        result = process_single_forecast(
            nc_file,
            z500_clim,
            threshold_90,
            min_area=min_area,
            min_duration=min_duration,
            min_overlap=min_overlap,
            full_simulation=True  # Always load entire simulation for ensemble forecasts
        )
        
        if 'error' in result:
            if verbose:
                print(f"  ERROR: {result['error']}")
            ensemble_results[member_name] = result
            continue
        
        if verbose:
            print(f"  z500 shape: {result['z500'].shape} (days, lat, lon)")
            print(f"  Anomaly mean: {float(result['z500_anom'].mean()):.2f} m")
            print(f"  Blocked grid points: {int(result['blocked_mask'].sum())}")
            print(f"  Blocking events detected: {result['num_events']}")
        
        # Store results
        ensemble_results[member_name] = result
        
        # Create animation if requested
        if create_animations and output_dir:
            if verbose:
                print(f"\n  Creating animation...")
            
            gif_path = os.path.join(output_dir, f"{member_name}_full_simulation.gif")
            create_forecast_animation(
                result['event_info'],
                gif_path,
                title_prefix=f"Ensemble Member {i}",
                region_lon_min=region_lon_min,
                region_lon_max=region_lon_max,
                region_lat_min=region_lat_min,
                region_lat_max=region_lat_max,
                fps=0.5,
                show_diagnostics=True
            )
            
            if verbose:
                print(f"  ✓ Animation saved: {gif_path}")
    
    # Print summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("ENSEMBLE SUMMARY")
        print(f"{'=' * 70}")
        total_events = 0
        for name, result in ensemble_results.items():
            if 'error' in result:
                print(f"  {name}: ERROR - {result['error'][:50]}...")
            else:
                print(f"  {name}: {result['num_events']} blocking events")
                total_events += result['num_events']
        print(f"\nTotal blocking events across all members: {total_events}")
        if output_dir:
            print(f"Animations saved to: {output_dir}")
    
    return ensemble_results


# =============================================================================
# Deterministic Forecast Analysis (Multiple Lead Times)
# =============================================================================

def analyze_deterministic_forecast(
    forecast_files: List[str],
    clim_threshold_file: str,
    threshold_json_file: str,
    output_dir: Optional[str] = None,
    lead_time_labels: Optional[List[str]] = None,
    season: str = 'DJF',
    extend: bool = True,
    min_area: float = 2e6,
    min_duration: int = 5,
    min_overlap: float = 0.5,
    create_animations: bool = True,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    restrict_to_core_season: bool = True,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Analyze deterministic forecasts across multiple lead times.
    
    Parameters
    ----------
    forecast_files : list of str
        List of paths to forecast NetCDF files (one per lead time).
    clim_threshold_file : str
        Path to pre-computed climatology NetCDF file.
    threshold_json_file : str
        Path to JSON file with 90th percentile thresholds.
    output_dir : str, optional
        Directory for output animations. If None, no animations are created.
    lead_time_labels : list of str, optional
        Labels for each lead time (e.g., ['Day 1-5', 'Day 6-10', ...]).
        If None, uses 'Lead_1', 'Lead_2', etc.
    season : str, optional
        Season to analyze ('DJF' or 'JJA'). Default: 'DJF'.
    extend : bool, optional
        Extend season by one month on each side. Default: True.
    min_area : float, optional
        Minimum blocking area in km². Default: 2e6.
    min_duration : int, optional
        Minimum blocking duration in days. Default: 5.
    min_overlap : float, optional
        Minimum spatial overlap fraction. Default: 0.5.
    create_animations : bool, optional
        Whether to create GIF animations. Default: True.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for diagnostic region. Default: 30-100°E.
    restrict_to_core_season : bool, optional
        If True (default), restrict blocking event detection to core season months
        only (DJF for winter, JJA for summer). Default: True.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for diagnostic region. Default: 55-75°N.
    verbose : bool, optional
        Print progress information. Default: True.
    
    Returns
    -------
    lead_time_results : dict
        Dictionary mapping lead time labels to their analysis results.
    
    Examples
    --------
    >>> files = ['/path/to/lead1.nc', '/path/to/lead2.nc', '/path/to/lead3.nc']
    >>> labels = ['Day 1-5', 'Day 6-10', 'Day 11-15']
    >>> results = analyze_deterministic_forecast(
    ...     forecast_files=files,
    ...     clim_threshold_file='/path/to/climatology.nc',
    ...     threshold_json_file='/path/to/thresholds.json',
    ...     lead_time_labels=labels,
    ...     output_dir='/path/to/output/'
    ... )
    """
    # Generate default labels if not provided
    if lead_time_labels is None:
        lead_time_labels = [f"Lead_{i+1}" for i in range(len(forecast_files))]
    
    if len(lead_time_labels) != len(forecast_files):
        raise ValueError("Number of labels must match number of files")
    
    # Load pre-computed climatology and thresholds
    if verbose:
        print("=" * 70)
        print("Loading pre-computed climatology and thresholds")
        print("=" * 70)
    
    z500_clim, threshold_90 = load_climatology_and_thresholds(
        clim_threshold_file, threshold_json_file
    )
    
    if verbose:
        print(f"   ✓ Climatology loaded: shape={z500_clim.shape}")
        print(f"   ✓ Thresholds loaded: {threshold_90}")
        print(f"\nProcessing {len(forecast_files)} lead times")
    
    # Create output directory if needed
    if output_dir and create_animations:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each lead time
    lead_time_results = {}
    
    for i, (nc_file, label) in enumerate(zip(forecast_files, lead_time_labels), 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Processing {label} ({i}/{len(forecast_files)})")
            print(f"{'=' * 60}")
        
        # Process this lead time
        result = process_single_forecast(
            nc_file,
            z500_clim,
            threshold_90,
            season=season,
            extend=extend,
            min_area=min_area,
            min_duration=min_duration,
            min_overlap=min_overlap,
            restrict_to_core_season=restrict_to_core_season
        )
        
        if 'error' in result:
            if verbose:
                print(f"  ERROR: {result['error']}")
            lead_time_results[label] = result
            continue
        
        if verbose:
            print(f"  z500 shape: {result['z500'].shape}")
            print(f"  Blocking events detected: {result['num_events']}")
        
        # Store results
        lead_time_results[label] = result
        
        # Create animation if requested
        if create_animations and output_dir:
            gif_path = os.path.join(output_dir, f"{label.replace(' ', '_')}_simulation.gif")
            create_forecast_animation(
                result['event_info'],
                gif_path,
                title_prefix=label,
                region_lon_min=region_lon_min,
                region_lon_max=region_lon_max,
                region_lat_min=region_lat_min,
                region_lat_max=region_lat_max,
                fps=0.5,
                show_diagnostics=True
            )
            if verbose:
                print(f"  ✓ Animation saved: {gif_path}")
    
    return lead_time_results


# =============================================================================
# Blocking Score Computation and Ranking
# =============================================================================

def compute_member_scores(
    results: Dict[str, Dict],
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    gamma: float = 5.0
) -> pd.DataFrame:
    """
    Compute blocking scores for all forecast members/lead times.
    
    Parameters
    ----------
    results : dict
        Results dictionary from analyze_ensemble_forecast() or 
        analyze_deterministic_forecast().
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for scoring region. Default: 30-100°E (Eurasia).
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for scoring region. Default: 55-75°N.
    gamma : float, optional
        Drift penalty constant. Default: 5.0.
    
    Returns
    -------
    df_scores : pd.DataFrame
        DataFrame with columns:
        - member: Member/lead time name
        - n_events: Number of regional blocking events
        - sum_P_total: Sum of all event scores
        - sum_P_block: Sum of blocking components
        - sum_P_drift: Sum of drift penalties
        - max_P_total: Maximum single event score
    
    Examples
    --------
    >>> results = analyze_ensemble_forecast(...)
    >>> df = compute_member_scores(results)
    >>> print(df.sort_values('sum_P_total', ascending=False))
    """
    member_scores = []
    
    for member_name, result in results.items():
        if 'error' in result:
            continue
        
        # Check for required keys - support both 'event_info' and 'stats' keys
        event_info_key = 'event_info' if 'event_info' in result else 'stats'
        if event_info_key not in result or 'z500' not in result:
            print(f"Warning: Skipping {member_name} - missing required keys")
            continue
        
        try:
            # Compute regional blocking scores
            df_events = compute_blocking_scores(
                z500=result['z500'],
                event_info=result[event_info_key],
                region_lon_min=region_lon_min,
                region_lon_max=region_lon_max,
                region_lat_min=region_lat_min,
                region_lat_max=region_lat_max,
                gamma=gamma
            )
            
            member_scores.append({
                'member': member_name,
                'n_events': len(df_events),
                'sum_P_total': df_events['P_total'].sum() if len(df_events) > 0 else 0,
                'sum_P_block': df_events['P_block'].sum() if len(df_events) > 0 else 0,
                'sum_P_drift': df_events['P_drift'].sum() if len(df_events) > 0 else 0,
                'max_P_total': df_events['P_total'].max() if len(df_events) > 0 else 0,
            })
        except Exception as e:
            print(f"Warning: Error computing scores for {member_name}: {e}")
            continue
    
    # Return DataFrame with proper columns even if empty
    if not member_scores:
        return pd.DataFrame(columns=['member', 'n_events', 'sum_P_total', 
                                      'sum_P_block', 'sum_P_drift', 'max_P_total'])
    
    return pd.DataFrame(member_scores)


def rank_ensemble_scores(
    results: Dict[str, Dict],
    threshold_score: Optional[float] = None,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    gamma: float = 5.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute, rank, and report blocking scores for ensemble members.
    
    Parameters
    ----------
    results : dict
        Results dictionary from analyze_ensemble_forecast().
    threshold_score : float, optional
        If provided, highlight members with sum_P_total above this threshold.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for scoring region. Default: 30-100°E (Eurasia).
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for scoring region. Default: 55-75°N.
    gamma : float, optional
        Drift penalty constant. Default: 5.0.
    verbose : bool, optional
        Print ranking tables. Default: True.
    
    Returns
    -------
    df_ranked : pd.DataFrame
        DataFrame with ranked members including:
        - rank: Ranking (1 = highest score)
        - member: Member name
        - n_events: Number of events
        - sum_P_total, sum_P_block, sum_P_drift, max_P_total: Score components
    
    Examples
    --------
    >>> results = analyze_ensemble_forecast(...)
    >>> df = rank_ensemble_scores(results, threshold_score=300000)
    >>> # Get top 3 performers
    >>> top_3 = df.head(3)
    """
    if verbose:
        print("=" * 70)
        print("COMPUTING BLOCKING SCORES FOR ALL MEMBERS")
        print(f"Region: {region_lon_min}°-{region_lon_max}°E, {region_lat_min}°-{region_lat_max}°N")
        print("=" * 70)
    
    # Compute scores for all members
    df_all = compute_member_scores(
        results,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        gamma=gamma
    )
    
    # Handle empty DataFrame
    if len(df_all) == 0:
        if verbose:
            print("\n⚠ No valid members to rank. All members may have errors.")
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=['rank', 'member', 'n_events', 'sum_P_total', 
                                      'sum_P_block', 'sum_P_drift', 'max_P_total'])
    
    # Sort by total score (descending)
    df_all = df_all.sort_values('sum_P_total', ascending=False).reset_index(drop=True)
    df_all['rank'] = df_all.index + 1
    
    # Reorder columns
    df_all = df_all[['rank', 'member', 'n_events', 'sum_P_total', 
                      'sum_P_block', 'sum_P_drift', 'max_P_total']]
    
    if verbose:
        # Display full ranking
        print("\n" + "=" * 90)
        print("ALL MEMBERS - FULL RANKING (Highest → Lowest)")
        print("=" * 90)
        print(df_all.to_string(index=False))
        print("=" * 90)
        
        # Display selected members if threshold provided
        if threshold_score is not None:
            df_selected = df_all[df_all['sum_P_total'] > threshold_score]
            print("\n" + "=" * 90)
            print(f"SELECTED MEMBERS (sum_P_total > {threshold_score:.2f})")
            print("=" * 90)
            if len(df_selected) == 0:
                print(f"\n⚠ No members above threshold.\n")
            else:
                print(df_selected.to_string(index=False))
                print(f"\n✓ Selected {len(df_selected)} out of {len(df_all)} members")
        
        # Summary statistics
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)
        print(f"  Max score:   {df_all['sum_P_total'].max():.2f}  ({df_all.iloc[0]['member']})")
        print(f"  Min score:   {df_all['sum_P_total'].min():.2f}  ({df_all.iloc[-1]['member']})")
        print(f"  Mean:        {df_all['sum_P_total'].mean():.2f}")
        if threshold_score is not None:
            print(f"  Threshold:   {threshold_score:.2f}")
        print("=" * 90)
    
    return df_all


# =============================================================================
# Convenience Wrapper Functions
# =============================================================================

def analyze_and_rank_ensemble(
    ensemble_dir: str,
    clim_threshold_file: str,
    threshold_json_file: str,
    output_dir: Optional[str] = None,
    file_pattern: str = '*_postproc.nc',
    threshold_score: Optional[float] = None,
    min_area: float = 2e6,
    min_duration: int = 5,
    min_overlap: float = 0.5,
    create_animations: bool = True,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    gamma: float = 5.0,
    verbose: bool = True
) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    """
    One-step analysis: process ensemble, create animations, and rank by score.
    
    This convenience function combines analyze_ensemble_forecast() and 
    rank_ensemble_scores() into a single call.
    
    The ENTIRE simulation from each ensemble member is analyzed (no seasonal
    filtering). Climatology and thresholds should be pre-computed from a long
    reference run.
    
    Parameters
    ----------
    ensemble_dir : str
        Directory containing ensemble forecast NetCDF files.
    clim_threshold_file : str
        Path to pre-computed climatology NetCDF file.
    threshold_json_file : str
        Path to JSON file with 90th percentile thresholds.
    output_dir : str, optional
        Directory for output animations.
    file_pattern : str, optional
        Glob pattern for finding ensemble files. Default: '*_postproc.nc'.
    threshold_score : float, optional
        Score threshold for selecting high-performing members.
    min_area : float, optional
        Minimum blocking area in km². Default: 2e6.
    min_duration : int, optional
        Minimum blocking duration in days. Default: 5.
    min_overlap : float, optional
        Minimum spatial overlap fraction. Default: 0.5.
    create_animations : bool, optional
        Whether to create GIF animations. Default: True.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for scoring region. Default: 30-100°E.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for scoring region. Default: 55-75°N.
    gamma : float, optional
        Drift penalty constant. Default: 5.0.
    verbose : bool, optional
        Print progress information. Default: True.
    
    Returns
    -------
    results : dict
        Dictionary mapping member names to their analysis results.
    df_ranked : pd.DataFrame
        Ranked DataFrame of member scores.
    
    Examples
    --------
    >>> results, df = analyze_and_rank_ensemble(
    ...     ensemble_dir='/path/to/ensemble/',
    ...     clim_threshold_file='/path/to/climatology.nc',
    ...     threshold_json_file='/path/to/thresholds.json',
    ...     output_dir='/path/to/output/',
    ...     threshold_score=300000
    ... )
    >>> # Get selected high performers
    >>> selected = df[df['sum_P_total'] > 300000]
    """
    # Step 1: Analyze ensemble (full simulation - no seasonal filtering)
    results = analyze_ensemble_forecast(
        ensemble_dir=ensemble_dir,
        clim_threshold_file=clim_threshold_file,
        threshold_json_file=threshold_json_file,
        output_dir=output_dir,
        file_pattern=file_pattern,
        min_area=min_area,
        min_duration=min_duration,
        min_overlap=min_overlap,
        create_animations=create_animations,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        verbose=verbose
    )
    
    # Step 2: Compute and rank scores
    df_ranked = rank_ensemble_scores(
        results,
        threshold_score=threshold_score,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        gamma=gamma,
        verbose=verbose
    )
    
    return results, df_ranked


def analyze_and_rank_deterministic(
    forecast_files: List[str],
    clim_threshold_file: str,
    threshold_json_file: str,
    output_dir: Optional[str] = None,
    lead_time_labels: Optional[List[str]] = None,
    threshold_score: Optional[float] = None,
    season: str = 'DJF',
    extend: bool = True,
    min_area: float = 2e6,
    min_duration: int = 5,
    min_overlap: float = 0.5,
    create_animations: bool = True,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    gamma: float = 5.0,
    restrict_to_core_season: bool = True,
    verbose: bool = True
) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    """
    One-step analysis: process deterministic forecast and rank by score.
    
    This convenience function combines analyze_deterministic_forecast() and 
    rank_ensemble_scores() into a single call.
    
    Parameters
    ----------
    forecast_files : list of str
        List of paths to forecast NetCDF files (one per lead time).
    clim_threshold_file : str
        Path to pre-computed climatology NetCDF file.
    threshold_json_file : str
        Path to JSON file with 90th percentile thresholds.
    output_dir : str, optional
        Directory for output animations.
    lead_time_labels : list of str, optional
        Labels for each lead time (e.g., ['Day 1-5', 'Day 6-10', ...]).
    threshold_score : float, optional
        Score threshold for highlighting high-scoring lead times.
    season : str, optional
        Season to analyze. Default: 'DJF'.
    extend : bool, optional
        Extend season by one month on each side. Default: True.
    min_area : float, optional
        Minimum blocking area in km². Default: 2e6.
    min_duration : int, optional
        Minimum blocking duration in days. Default: 5.
    min_overlap : float, optional
        Minimum spatial overlap fraction. Default: 0.5.
    restrict_to_core_season : bool, optional
        If True (default), restrict blocking event detection to core season months
        only (DJF for winter, JJA for summer). Default: True.
    create_animations : bool, optional
        Whether to create GIF animations. Default: True.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for scoring region. Default: 30-100°E.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for scoring region. Default: 55-75°N.
    gamma : float, optional
        Drift penalty constant. Default: 5.0.
    verbose : bool, optional
        Print progress information. Default: True.
    
    Returns
    -------
    results : dict
        Dictionary mapping lead time labels to their analysis results.
    df_ranked : pd.DataFrame
        Ranked DataFrame of lead time scores.
    
    Examples
    --------
    >>> files = ['/path/to/lead1.nc', '/path/to/lead2.nc', '/path/to/lead3.nc']
    >>> labels = ['Day 1-5', 'Day 6-10', 'Day 11-15']
    >>> results, df = analyze_and_rank_deterministic(
    ...     forecast_files=files,
    ...     clim_threshold_file='/path/to/climatology.nc',
    ...     threshold_json_file='/path/to/thresholds.json',
    ...     lead_time_labels=labels,
    ...     output_dir='/path/to/output/'
    ... )
    """
    # Step 1: Analyze deterministic forecast
    results = analyze_deterministic_forecast(
        forecast_files=forecast_files,
        clim_threshold_file=clim_threshold_file,
        threshold_json_file=threshold_json_file,
        output_dir=output_dir,
        lead_time_labels=lead_time_labels,
        season=season,
        extend=extend,
        min_area=min_area,
        min_duration=min_duration,
        min_overlap=min_overlap,
        create_animations=create_animations,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        restrict_to_core_season=restrict_to_core_season,
        verbose=verbose
    )
    
    # Step 2: Compute and rank scores
    df_ranked = rank_ensemble_scores(
        results,
        threshold_score=threshold_score,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        gamma=gamma,
        verbose=verbose
    )
    
    return results, df_ranked


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    print("Forecast Blocking Analysis Module")
    print("=" * 50)
    print("Available functions:")
    print("  - load_climatology_and_thresholds()")
    print("  - extract_z500_daily()")
    print("  - compute_anomalies_with_climatology()")
    print("  - process_single_forecast()")
    print("  - create_forecast_animation()")
    print("  - analyze_ensemble_forecast()")
    print("  - analyze_deterministic_forecast()")
    print("  - compute_member_scores()")
    print("  - rank_ensemble_scores()")
    print("  - analyze_and_rank_ensemble()")
    print("  - analyze_and_rank_deterministic()")
    print("\nSee docstrings for usage examples.")

