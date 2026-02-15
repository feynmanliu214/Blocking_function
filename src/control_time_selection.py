#!/usr/bin/env python3
"""
Control Time Selection for ANO Blocking Events.

This module provides functionality to select non-blocking control verification
times for blocking events identified by ano_blocking_complete. Each control time
is season-matched (same day-of-year) and verified to have no blocking activity
within a specified exclusion region during a buffer window.

Usage:
    from ANO_PlaSim import ano_blocking_complete
    from compute_blocking_scores import compute_blocking_scores
    from control_time_selection import select_control_times

    # Step 1: Run blocking detection
    blocking_freq, event_info = ano_blocking_complete(z500, ...)

    # Step 2: Compute regional scores and get top 50 events
    df_scores = compute_blocking_scores(z500, event_info, region="Atlantic")
    top_50 = df_scores.head(50)

    # Step 3: Select control times
    control_df = select_control_times(
        blocked_mask=event_info['blocked_mask'],
        top_events_df=top_50,
        lat_min=55.0, lat_max=75.0,
        lon_min=-60.0, lon_max=0.0,
        buffer_days=7,
        seed=42
    )

Author: Generated with Claude Code assistance
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Optional, Tuple, List, Union
from datetime import datetime

# Try to import cftime for non-standard calendar support
try:
    import cftime
    HAS_CFTIME = True
except ImportError:
    HAS_CFTIME = False


# Calendar year lengths
CALENDAR_YEAR_LENGTHS = {
    'noleap': 365,
    '365_day': 365,
    'all_leap': 366,
    '366_day': 366,
    '360_day': 360,
    'standard': 365,  # Use 365 as default for standard/proleptic_gregorian
    'gregorian': 365,
    'proleptic_gregorian': 365,
    None: 365,  # Default fallback
}


def detect_calendar(time_coord: xr.DataArray) -> str:
    """
    Detect the calendar type from an xarray time coordinate.

    Parameters
    ----------
    time_coord : xr.DataArray
        The time coordinate from an xarray Dataset/DataArray.

    Returns
    -------
    str
        Calendar name (e.g., 'noleap', '360_day', 'standard').
    """
    # Check if there's an encoding with calendar info
    if hasattr(time_coord, 'encoding') and 'calendar' in time_coord.encoding:
        return time_coord.encoding['calendar']

    # Check the dtype of the first value
    first_val = time_coord.values[0]

    if HAS_CFTIME and isinstance(first_val, cftime.datetime):
        # cftime objects have a calendar attribute
        return first_val.calendar

    # Default to standard calendar for numpy datetime64
    if isinstance(first_val, np.datetime64):
        return 'standard'

    # Try to get calendar from attributes
    if hasattr(time_coord, 'attrs') and 'calendar' in time_coord.attrs:
        return time_coord.attrs['calendar']

    return 'standard'


def get_year_length(calendar: str) -> int:
    """
    Get the number of days in a year for a given calendar.

    Parameters
    ----------
    calendar : str
        Calendar name.

    Returns
    -------
    int
        Number of days in a year (360, 365, or 366).
    """
    return CALENDAR_YEAR_LENGTHS.get(calendar, 365)


def get_day_of_year(time_val, calendar: str = None) -> int:
    """
    Extract day-of-year from a time value, respecting the calendar type.

    Parameters
    ----------
    time_val : numpy.datetime64, pandas.Timestamp, or cftime.datetime
        The time value to extract DOY from.
    calendar : str, optional
        Calendar type hint. If None, will try to detect from time_val.

    Returns
    -------
    int
        Day of year (1-360/365/366 depending on calendar).

    Notes
    -----
    For cftime objects, uses the native dayofyr attribute.
    For numpy/pandas, converts to Timestamp and uses dayofyear.
    """
    # Handle cftime objects
    if HAS_CFTIME and isinstance(time_val, cftime.datetime):
        return time_val.dayofyr

    # Handle numpy datetime64
    if isinstance(time_val, np.datetime64):
        ts = pd.Timestamp(time_val)
        return ts.dayofyear

    # Handle pandas Timestamp
    if isinstance(time_val, pd.Timestamp):
        return time_val.dayofyear

    # Fallback: try to convert to Timestamp
    try:
        ts = pd.Timestamp(time_val)
        return ts.dayofyear
    except Exception:
        raise TypeError(f"Cannot extract day-of-year from type {type(time_val)}")


def get_year(time_val) -> int:
    """
    Extract year from a time value.

    Parameters
    ----------
    time_val : numpy.datetime64, pandas.Timestamp, or cftime.datetime
        The time value to extract year from.

    Returns
    -------
    int
        Year.
    """
    # Handle cftime objects
    if HAS_CFTIME and isinstance(time_val, cftime.datetime):
        return time_val.year

    # Handle numpy datetime64
    if isinstance(time_val, np.datetime64):
        ts = pd.Timestamp(time_val)
        return ts.year

    # Handle pandas Timestamp
    if isinstance(time_val, pd.Timestamp):
        return time_val.year

    # Fallback: try to convert to Timestamp
    try:
        ts = pd.Timestamp(time_val)
        return ts.year
    except Exception:
        raise TypeError(f"Cannot extract year from type {type(time_val)}")


def doy_distance(doy1: int, doy2: int, days_in_year: int = 365) -> int:
    """
    Compute circular distance between two days-of-year, handling wrap-around.

    Parameters
    ----------
    doy1 : int
        First day of year (1-366).
    doy2 : int
        Second day of year (1-366).
    days_in_year : int
        Number of days in year (default 365, use 366 for leap years).

    Returns
    -------
    int
        Minimum circular distance between the two DOYs.

    Examples
    --------
    >>> doy_distance(1, 365, 365)
    1
    >>> doy_distance(10, 20, 365)
    10
    >>> doy_distance(360, 5, 365)
    10
    """
    direct = abs(doy1 - doy2)
    wrap = days_in_year - direct
    return min(direct, wrap)


def find_candidate_control_times(
    blocked_mask: xr.DataArray,
    event_time,
    event_year: int,
    doy_tolerance: int = 5,
    calendar: str = 'standard',
) -> List[Tuple[int, int]]:
    """
    Find candidate control times that are season-matched to the event.

    Candidates are times with the same day-of-year (within tolerance) but
    from different years than the blocking event.

    Parameters
    ----------
    blocked_mask : xr.DataArray
        The 3D blocking mask (time, lat, lon) from event_info.
    event_time : numpy.datetime64, pandas.Timestamp, or cftime.datetime
        The verification time of the blocking event.
    event_year : int
        The year of the blocking event (controls must be from different years).
    doy_tolerance : int
        Maximum allowed difference in day-of-year for season matching.
    calendar : str
        Calendar type (e.g., 'noleap', '360_day', 'standard').
        Used for correct DOY wrap-around calculation.

    Returns
    -------
    List[Tuple[int, int]]
        List of (time_index, doy_distance) tuples for valid candidates,
        sorted by doy_distance (closest match first).
    """
    event_doy = get_day_of_year(event_time)
    times = blocked_mask.time.values
    days_in_year = get_year_length(calendar)

    candidates = []

    for idx, t in enumerate(times):
        t_year = get_year(t)
        t_doy = get_day_of_year(t)

        # Must be from a different year
        if t_year == event_year:
            continue

        # Check DOY distance with wrap-around handling
        dist = doy_distance(event_doy, t_doy, days_in_year=days_in_year)

        if dist <= doy_tolerance:
            candidates.append((idx, dist))

    # Sort by DOY distance (closest first)
    candidates.sort(key=lambda x: x[1])

    return candidates


def check_no_blocking_in_region(
    blocked_mask: xr.DataArray,
    center_idx: int,
    buffer_days: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> bool:
    """
    Check if there is no blocking within the exclusion region during the buffer window.

    Parameters
    ----------
    blocked_mask : xr.DataArray
        The 3D blocking mask (time, lat, lon) from event_info.
    center_idx : int
        Time index of the candidate control time.
    buffer_days : int
        Half-width of the buffer window (e.g., 7 means ±7 days).
    lat_min, lat_max : float
        Latitude bounds of the exclusion region.
    lon_min, lon_max : float
        Longitude bounds of the exclusion region.

    Returns
    -------
    bool
        True if no blocking exists in the region during the window, False otherwise.
    """
    n_times = blocked_mask.sizes['time']

    # Define window bounds (clipped to valid range)
    start_idx = max(0, center_idx - buffer_days)
    end_idx = min(n_times, center_idx + buffer_days + 1)  # +1 for inclusive end

    # Extract the window
    window = blocked_mask.isel(time=slice(start_idx, end_idx))

    # Apply regional mask
    # Handle longitude wrap-around if needed (lon_min > lon_max means crossing 180)
    if lon_min <= lon_max:
        lon_mask = (blocked_mask.lon >= lon_min) & (blocked_mask.lon <= lon_max)
    else:
        # Crossing the date line (e.g., lon_min=170, lon_max=-170)
        lon_mask = (blocked_mask.lon >= lon_min) | (blocked_mask.lon <= lon_max)

    lat_mask = (blocked_mask.lat >= lat_min) & (blocked_mask.lat <= lat_max)

    # Extract region
    regional_window = window.where(lon_mask & lat_mask, drop=True)

    # Check if ANY blocking exists in this region during the window
    has_blocking = regional_window.sum() > 0

    return not has_blocking


def select_control_times(
    blocked_mask: xr.DataArray,
    top_events_df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    doy_tolerance: int = 5,
    buffer_days: int = 7,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Select non-blocking control times for each blocking event.

    For each event in top_events_df, this function finds a control time that:
    1. Has the same day-of-year (within doy_tolerance) as the event's start time
    2. Is from a different year than the event
    3. Has no blocking detected within the exclusion region during a
       ±buffer_days window centered on the control time

    Parameters
    ----------
    blocked_mask : xr.DataArray
        The 3D blocking mask (time, lat, lon) from event_info['blocked_mask'].
        Values should be 0 (no blocking) or 1 (blocking).
    top_events_df : pd.DataFrame
        DataFrame of top blocking events from compute_blocking_scores.
        Must contain columns: 'region_event_id', 'start_time', 'end_time'.
    lat_min, lat_max : float
        Latitude bounds of the exclusion region where no blocking should occur.
    lon_min, lon_max : float
        Longitude bounds of the exclusion region where no blocking should occur.
        For regions crossing the date line, use lon_min > lon_max
        (e.g., lon_min=170, lon_max=-170).
    doy_tolerance : int, default=5
        Maximum allowed difference in day-of-year between event and control.
        Handles year wrap-around (e.g., DOY 1 is close to DOY 365).
    buffer_days : int, default=7
        Half-width of the no-blocking window. A value of 7 means the window
        spans [control_time - 7 days, control_time + 7 days] (15 days total).
    seed : int, default=42
        Random seed for reproducible selection when multiple valid controls exist.
    verbose : bool, default=True
        If True, print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per input event, containing:
        - event_id: The region_event_id from input
        - event_start_time: The start (verification) time of the blocking event
        - event_doy: Day-of-year of the event start time
        - event_year: Year of the blocking event
        - control_time: Selected control time (NaT if no valid control found)
        - control_doy: Day-of-year of control time
        - control_year: Year of control time
        - doy_offset: Actual DOY difference between event and control
        - n_valid_candidates: Number of valid control candidates found
        - status: 'matched', 'no_valid_control', or 'no_candidates'

    Examples
    --------
    >>> control_df = select_control_times(
    ...     blocked_mask=event_info['blocked_mask'],
    ...     top_events_df=top_50,
    ...     lat_min=55.0, lat_max=75.0,
    ...     lon_min=-60.0, lon_max=0.0,
    ...     buffer_days=7,
    ...     seed=42
    ... )
    >>> print(control_df[['event_id', 'event_start_time', 'control_time', 'status']])
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Detect calendar type from the time coordinate
    calendar = detect_calendar(blocked_mask.time)
    days_in_year = get_year_length(calendar)

    if verbose:
        print(f"Selecting control times for {len(top_events_df)} events")
        print(f"  Calendar: {calendar} ({days_in_year} days/year)")
        print(f"  Exclusion region: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
        print(f"  DOY tolerance: ±{doy_tolerance} days")
        print(f"  Buffer window: ±{buffer_days} days")
        print(f"  Random seed: {seed}")
        print()

    results = []

    for idx, row in top_events_df.iterrows():
        event_id = row['region_event_id']
        event_start_time = row['start_time']
        event_year = get_year(event_start_time)
        event_doy = get_day_of_year(event_start_time)

        if verbose:
            print(f"Event {event_id}: start_time={event_start_time}, DOY={event_doy}, year={event_year}")

        # Step 1: Find season-matched candidates from different years
        candidates = find_candidate_control_times(
            blocked_mask=blocked_mask,
            event_time=event_start_time,
            event_year=event_year,
            doy_tolerance=doy_tolerance,
            calendar=calendar,
        )

        if not candidates:
            if verbose:
                print(f"  -> No season-matched candidates found (different year required)")
            results.append({
                'event_id': event_id,
                'event_start_time': event_start_time,
                'event_doy': event_doy,
                'event_year': event_year,
                'control_time': pd.NaT,
                'control_doy': np.nan,
                'control_year': np.nan,
                'doy_offset': np.nan,
                'n_valid_candidates': 0,
                'status': 'no_candidates',
            })
            continue

        # Step 2: Filter candidates by no-blocking constraint
        valid_candidates = []

        for time_idx, doy_dist in candidates:
            is_clear = check_no_blocking_in_region(
                blocked_mask=blocked_mask,
                center_idx=time_idx,
                buffer_days=buffer_days,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )

            if is_clear:
                valid_candidates.append((time_idx, doy_dist))

        if not valid_candidates:
            if verbose:
                print(f"  -> {len(candidates)} candidates checked, none satisfy no-blocking constraint")
            results.append({
                'event_id': event_id,
                'event_start_time': event_start_time,
                'event_doy': event_doy,
                'event_year': event_year,
                'control_time': pd.NaT,
                'control_doy': np.nan,
                'control_year': np.nan,
                'doy_offset': np.nan,
                'n_valid_candidates': 0,
                'status': 'no_valid_control',
            })
            continue

        # Step 3: Randomly select from valid candidates
        selected_idx = rng.integers(0, len(valid_candidates))
        chosen_time_idx, chosen_doy_dist = valid_candidates[selected_idx]

        control_time = blocked_mask.time.values[chosen_time_idx]
        control_doy = get_day_of_year(control_time)
        control_year = get_year(control_time)

        if verbose:
            print(f"  -> {len(valid_candidates)}/{len(candidates)} valid candidates, "
                  f"selected: {control_time} (DOY={control_doy}, year={control_year})")

        results.append({
            'event_id': event_id,
            'event_start_time': event_start_time,
            'event_doy': event_doy,
            'event_year': event_year,
            'control_time': control_time,
            'control_doy': control_doy,
            'control_year': control_year,
            'doy_offset': chosen_doy_dist,
            'n_valid_candidates': len(valid_candidates),
            'status': 'matched',
        })

    df = pd.DataFrame(results)

    if verbose:
        print()
        print("=" * 60)
        n_matched = (df['status'] == 'matched').sum()
        n_no_control = (df['status'] == 'no_valid_control').sum()
        n_no_candidates = (df['status'] == 'no_candidates').sum()
        print(f"Summary: {n_matched} matched, {n_no_control} no valid control, "
              f"{n_no_candidates} no candidates")

        if n_matched > 0:
            print(f"Mean DOY offset for matched events: {df.loc[df['status']=='matched', 'doy_offset'].mean():.2f} days")

    return df


def summarize_control_selection(control_df: pd.DataFrame) -> None:
    """
    Print a summary of control time selection results.

    Parameters
    ----------
    control_df : pd.DataFrame
        Output from select_control_times().
    """
    print("Control Time Selection Summary")
    print("=" * 60)

    n_total = len(control_df)
    status_counts = control_df['status'].value_counts()

    print(f"Total events: {n_total}")
    print()

    for status, count in status_counts.items():
        pct = 100 * count / n_total
        print(f"  {status}: {count} ({pct:.1f}%)")

    print()

    matched = control_df[control_df['status'] == 'matched']
    if len(matched) > 0:
        print("Matched events statistics:")
        print(f"  DOY offset - mean: {matched['doy_offset'].mean():.2f}, "
              f"max: {matched['doy_offset'].max():.0f}")
        print(f"  Valid candidates - mean: {matched['n_valid_candidates'].mean():.1f}, "
              f"min: {matched['n_valid_candidates'].min():.0f}, "
              f"max: {matched['n_valid_candidates'].max():.0f}")

        # Year distribution of controls
        year_counts = matched['control_year'].value_counts().sort_index()
        print(f"  Control years: {year_counts.to_dict()}")


def export_control_times(
    control_df: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
) -> None:
    """
    Export control time selection results to a file.

    Parameters
    ----------
    control_df : pd.DataFrame
        Output from select_control_times().
    output_path : str
        Path to output file.
    format : str, default='csv'
        Output format: 'csv' or 'parquet'.
    """
    if format == 'csv':
        control_df.to_csv(output_path, index=False)
    elif format == 'parquet':
        control_df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'parquet'.")

    print(f"Exported control times to {output_path}")
