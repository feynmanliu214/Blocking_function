#!/usr/bin/env python3
"""
Blocking Detection for Single Forecast Files.

This module provides functions for processing forecast files:
- process_single_forecast(): Full blocking detection with ANO method
- load_forecast_anomalies_only(): Lightweight loading of Z500 anomalies only

Author: AI-RES Project
"""

import os
import sys
import traceback
from typing import Dict

import numpy as np
import xarray as xr

# Add src/ directory to path for importing core modules
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ANO_PlaSim import create_blocking_mask, identify_blocking_events
from .data_loading import extract_z500_daily, compute_anomalies_with_climatology


def load_forecast_anomalies_only(
    nc_file: str,
    z500_clim: xr.DataArray,
    season: str = 'DJF',
    extend: bool = True,
    full_simulation: bool = True
) -> Dict:
    """
    Load Z500 anomalies from a forecast file WITHOUT running blocking detection.

    This is a lightweight alternative to process_single_forecast() for scorers
    that don't need blocking event identification (e.g., RMSEScorer).

    Parameters
    ----------
    nc_file : str
        Path to the forecast NetCDF file.
    z500_clim : xr.DataArray
        Pre-computed day-of-year climatology.
    season : str, optional
        Season to extract ('DJF' or 'JJA'). Default: 'DJF'.
        Ignored if full_simulation=True.
    extend : bool, optional
        Extend season by one month on each side. Default: True.
        Ignored if full_simulation=True.
    full_simulation : bool, optional
        If True, use entire simulation without seasonal filtering.
        Default: True.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'z500': Daily z500 data
        - 'z500_anom': Anomalies
        - 'file': Source file path
        - 'blocking_detection_skipped': True (indicates no blocking detection)

    Raises
    ------
    Exception
        If loading fails, returns dict with 'error' key instead.
    """
    try:
        # Extract z500 daily data
        z500_daily = extract_z500_daily(
            nc_file, season=season, extend=extend, full_simulation=full_simulation
        )

        # Compute anomalies using pre-computed climatology
        z500_anom = compute_anomalies_with_climatology(z500_daily, z500_clim)

        return {
            'z500': z500_daily,
            'z500_anom': z500_anom,
            'file': nc_file,
            'blocking_detection_skipped': True
        }

    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'file': nc_file
        }


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
    try:
        # Extract z500 daily data
        z500_daily = extract_z500_daily(
            nc_file, season=season, extend=extend, full_simulation=full_simulation
        )

        # Compute anomalies using pre-computed climatology
        z500_anom = compute_anomalies_with_climatology(z500_daily, z500_clim)

        # Apply pre-computed thresholds to create blocking mask
        blocked_mask = create_blocking_mask(z500_anom, threshold_90)

        # Restrict to core season if requested
        detection_season_info = "all months in data"
        if restrict_to_core_season:
            months_in_data = set(blocked_mask.time.dt.month.values)

            # Determine core season based on months present
            if months_in_data == {11, 12, 1, 2, 3} or months_in_data == {12, 1, 2}:
                core_months = {12, 1, 2}
                detection_season_info = "DJF only (Dec, Jan, Feb)"
            elif months_in_data == {5, 6, 7, 8, 9} or months_in_data == {6, 7, 8}:
                core_months = {6, 7, 8}
                detection_season_info = "JJA only (Jun, Jul, Aug)"
            elif 12 in months_in_data or 1 in months_in_data or 2 in months_in_data:
                core_months = {12, 1, 2}
                detection_season_info = "DJF only (Dec, Jan, Feb)"
            elif 6 in months_in_data or 7 in months_in_data or 8 in months_in_data:
                core_months = {6, 7, 8}
                detection_season_info = "JJA only (Jun, Jul, Aug)"
            else:
                core_months = months_in_data
                detection_season_info = "all months (unknown season pattern)"

            # Create mask for core season months
            core_season_mask = np.isin(blocked_mask.time.dt.month.values, list(core_months))

            # Zero out blocking in shoulder months
            blocked_mask_for_events = blocked_mask.copy()
            blocked_mask_for_events.values[~core_season_mask, :, :] = 0
        else:
            blocked_mask_for_events = blocked_mask

        # Identify blocking events with spatial/temporal tracking
        blocking_freq, event_info = identify_blocking_events(
            blocked_mask_for_events,
            min_area=min_area,
            min_duration=min_duration,
            min_overlap=min_overlap
        )

        # Add additional info to event_info for downstream use
        event_info['z500_anom'] = z500_anom
        event_info['blocked_mask'] = blocked_mask_for_events
        event_info['blocked_mask_full'] = blocked_mask
        event_info['threshold_90'] = threshold_90
        event_info['restrict_to_core_season'] = restrict_to_core_season
        event_info['detection_season'] = detection_season_info

        return {
            'z500': z500_daily,
            'z500_anom': z500_anom,
            'blocked_mask': blocked_mask_for_events,
            'blocked_mask_full': blocked_mask,
            'blocking_freq': blocking_freq,
            'event_info': event_info,
            'num_events': event_info['num_events'],
            'file': nc_file
        }

    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'file': nc_file
        }
