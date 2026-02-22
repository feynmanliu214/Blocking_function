#!/usr/bin/env python3
"""
Deterministic Forecast Analysis for Blocking Events.

This module provides functions for analyzing blocking events in deterministic
forecasts across multiple lead times.

Author: AI-RES Project
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .data_loading import load_climatology_and_thresholds
from .blocking_detection import process_single_forecast, load_forecast_anomalies_only
from .animation import create_forecast_animation
from .scoring import (
    rank_ensemble_scores,
    BlockingScorer,
    compute_rmse_score,
    compute_onset_centroid_from_event_info,
)


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
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for diagnostic region. Default: 55-75°N.
    restrict_to_core_season : bool, optional
        If True (default), restrict blocking event detection to core season
        months only (DJF for winter, JJA for summer).
    verbose : bool, optional
        Print progress information. Default: True.

    Returns
    -------
    lead_time_results : dict
        Dictionary mapping lead time labels to their analysis results.
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
        print(f"   Climatology loaded: shape={z500_clim.shape}")
        print(f"   Thresholds loaded: {threshold_90}")
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
                print(f"  Animation saved: {gif_path}")

    return lead_time_results


def analyze_and_rank_deterministic(
    forecast_files: List[str],
    clim_threshold_file: str,
    threshold_json_file: str,
    output_dir: Optional[str] = None,
    lead_time_labels: Optional[List[str]] = None,
    threshold_score: Optional[float] = None,
    scorer: Optional[BlockingScorer] = None,
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
    scorer : BlockingScorer, optional
        Scorer instance to use. If None, uses DriftPenalizedScorer.
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
    create_animations : bool, optional
        Whether to create GIF animations. Default: True.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for scoring region. Default: 30-100°E.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for scoring region. Default: 55-75°N.
    gamma : float, optional
        Drift penalty constant. Default: 5.0.
    restrict_to_core_season : bool, optional
        If True (default), restrict blocking event detection to core season
        months only (DJF for winter, JJA for summer).
    verbose : bool, optional
        Print progress information. Default: True.

    Returns
    -------
    results : dict
        Dictionary mapping lead time labels to their analysis results.
    df_ranked : pd.DataFrame
        Ranked DataFrame of lead time scores.
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
        scorer=scorer,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        gamma=gamma,
        verbose=verbose
    )

    return results, df_ranked


# Default path pattern for PlaSim truth data
PLASIM_TRUTH_PATH_PATTERN = (
    '/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/plev_data/{year}_gaussian.nc'
)

# Predefined blocking regions of interest
BLOCKING_REGIONS = {
    'Eurasia': {'lon_min': 20.0, 'lon_max': 150.0, 'lat_min': 45.0, 'lat_max': 80.0},
    'Siberia': {'lon_min': 60.0, 'lon_max': 120.0, 'lat_min': 50.0, 'lat_max': 75.0},
    'Ural': {'lon_min': 40.0, 'lon_max': 80.0, 'lat_min': 50.0, 'lat_max': 70.0},
    'North_Atlantic': {'lon_min': -60.0, 'lon_max': 0.0, 'lat_min': 45.0, 'lat_max': 75.0},
    'North_Pacific': {'lon_min': 150.0, 'lon_max': 240.0, 'lat_min': 45.0, 'lat_max': 75.0},
    'Europe': {'lon_min': -10.0, 'lon_max': 40.0, 'lat_min': 45.0, 'lat_max': 70.0},
    'NH': {'lon_min': 0.0, 'lon_max': 360.0, 'lat_min': 30.0, 'lat_max': 90.0},  # Northern Hemisphere
}


def compute_rmse_by_leadtime(
    forecast_files: List[str],
    clim_threshold_file: str,
    threshold_json_file: str,
    truth_file: Optional[str] = None,
    truth_path_pattern: str = PLASIM_TRUTH_PATH_PATTERN,
    region: Optional[Union[str, Dict[str, float]]] = 'Eurasia',
    lead_time_labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    season: str = 'DJF',
    extend: bool = True,
    min_area: float = 2e6,
    min_duration: int = 5,
    min_overlap: float = 0.5,
    restrict_to_core_season: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute RMSE between emulator forecasts and truth at the blocking onset centroid.

    This function:
    1. Extracts target onset date from forecast folder name (e.g., 'deterministic_0104-12-05')
    2. Auto-constructs truth file path from the year if not provided
    3. Computes blocking centroid at the target onset date
    4. For each forecast lead time, computes RMSE at a 3x3 patch centered on the centroid

    Parameters
    ----------
    forecast_files : list of str
        List of paths to emulator forecast NetCDF files (one per lead time).
        The folder name must contain the target date (e.g., 'deterministic_0104-12-05').
    clim_threshold_file : str
        Path to pre-computed climatology NetCDF file.
    threshold_json_file : str
        Path to JSON file with 90th percentile thresholds.
    truth_file : str, optional
        Path to the PlaSim truth NetCDF file. If None, auto-constructed from
        the target year using truth_path_pattern.
    truth_path_pattern : str, optional
        Path pattern for truth files with {year} placeholder.
        Default: '/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/plev_data/{year}_gaussian.nc'
    region : str or dict, optional
        Region of interest for centroid calculation. Can be:
        - A string key from BLOCKING_REGIONS: 'Eurasia', 'Siberia', 'Ural',
          'North_Atlantic', 'North_Pacific', 'Europe', 'NH'
        - A dict with keys: 'lon_min', 'lon_max', 'lat_min', 'lat_max'
        - None to use global domain (not recommended if multiple blocks exist)
        Default: 'Eurasia'.
        If multiple disconnected blocks exist within the region, uses the largest.
    lead_time_labels : list of str, optional
        Labels for each lead time (e.g., ['1d', '2d', ...]).
        If None, extracts from filenames or uses 'Lead_1', 'Lead_2', etc.
    output_dir : str, optional
        Directory for output CSV. If None, doesn't save to file.
    season : str, optional
        Season to analyze ('DJF' or 'JJA'). Default: 'DJF'.
    extend : bool, optional
        Extend season by one month on each side. Default: True.
    min_area : float, optional
        Minimum blocking area in km² for truth detection. Default: 2e6.
    min_duration : int, optional
        Minimum blocking duration in days for truth detection. Default: 5.
    min_overlap : float, optional
        Minimum spatial overlap fraction for truth detection. Default: 0.5.
    restrict_to_core_season : bool, optional
        If True (default), restrict blocking detection to core season months.
    verbose : bool, optional
        Print progress information. Default: True.

    Returns
    -------
    df_rmse : pd.DataFrame
        DataFrame with columns:
        - lead_time: Lead time label
        - rmse_m: RMSE in meters at the onset centroid
        - block_lon: Longitude of blocking centroid at verification date
        - block_lat: Latitude of blocking centroid at verification date
        - matched_date: Verification date (target onset date)
        - truth_time_idx: Time index in truth data
        - emulator_time_idx: Time index in emulator data
    metadata : dict
        Dictionary containing:
        - truth_file: Path to truth file
        - truth_num_events: Number of blocking events in truth
        - target_onset_date: Target onset date from folder name
        - verification_lon: Blocking centroid longitude at target date
        - verification_lat: Blocking centroid latitude at target date

    Raises
    ------
    ValueError
        If no blocking events are detected in truth data, or if target date
        cannot be extracted from folder name.

    Examples
    --------
    >>> # Auto-construct truth file from folder name
    >>> df_rmse, metadata = compute_rmse_by_leadtime(
    ...     forecast_files=glob.glob('/path/to/deterministic_0104-12-05/*.nc'),
    ...     clim_threshold_file='climatology.nc',
    ...     threshold_json_file='thresholds.json',
    ... )
    >>> print(df_rmse)

    >>> # Or provide truth file explicitly
    >>> df_rmse, metadata = compute_rmse_by_leadtime(
    ...     forecast_files=glob.glob('/path/to/deterministic_0104-12-05/*.nc'),
    ...     clim_threshold_file='climatology.nc',
    ...     threshold_json_file='thresholds.json',
    ...     truth_file='/path/to/104_gaussian.nc',
    ... )
    """
    # Generate default labels if not provided
    if lead_time_labels is None:
        lead_time_labels = []
        for f in forecast_files:
            # Try to extract lead time from filename (e.g., '1d', '2d')
            basename = os.path.basename(f)
            parts = basename.split('_')
            label = None
            for part in parts:
                if part.endswith('d') and part[:-1].isdigit():
                    label = part
                    break
            if label is None:
                label = f"Lead_{len(lead_time_labels) + 1}"
            lead_time_labels.append(label)

    if len(lead_time_labels) != len(forecast_files):
        raise ValueError("Number of labels must match number of files")

    # Extract target onset date from folder name FIRST
    # Expected format: .../deterministic_YYYY-MM-DD/... or .../deterministic_0104-12-05/...
    import re
    folder_path = os.path.dirname(forecast_files[0])
    folder_name = os.path.basename(folder_path)

    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', folder_name)
    if date_match:
        target_year = int(date_match.group(1))
        target_month = int(date_match.group(2))
        target_day = int(date_match.group(3))
        verification_date = (target_year, target_month, target_day)
    else:
        raise ValueError(
            f"Could not extract target date from folder name '{folder_name}'. "
            f"Expected format like 'deterministic_0104-12-05'"
        )

    # Auto-construct truth file path if not provided
    if truth_file is None:
        truth_file = truth_path_pattern.format(year=target_year)

    # Load climatology and thresholds
    if verbose:
        print("=" * 70)
        print("RMSE Analysis: Emulator vs Truth at Blocking Onset")
        print("=" * 70)
        print(f"\nTarget onset date from folder: {verification_date}")
        print("\nLoading climatology and thresholds...")

    z500_clim, threshold_90 = load_climatology_and_thresholds(
        clim_threshold_file, threshold_json_file
    )

    # Process truth data to detect blocking and get onset centroid
    if verbose:
        print(f"\nProcessing truth file for blocking detection...")
        print(f"  File: {truth_file}")

    truth_result = process_single_forecast(
        truth_file,
        z500_clim,
        threshold_90,
        season=season,
        extend=extend,
        min_area=min_area,
        min_duration=min_duration,
        min_overlap=min_overlap,
        full_simulation=False,
        restrict_to_core_season=restrict_to_core_season
    )

    if 'error' in truth_result:
        raise ValueError(f"Error processing truth file: {truth_result['error']}")

    if truth_result['num_events'] == 0:
        raise ValueError("No blocking events detected in truth data")

    if verbose:
        print(f"  Blocking events detected: {truth_result['num_events']}")

    # Get truth anomalies and onset centroid
    truth_z500_anom = truth_result['z500_anom']
    event_info = truth_result['event_info']

    truth_onset_lon, truth_onset_lat, truth_onset_time_idx = \
        compute_onset_centroid_from_event_info(event_info)

    truth_onset_date = str(truth_z500_anom.time.values[truth_onset_time_idx])

    if verbose:
        print(f"\nTruth blocking onset (first event):")
        print(f"  Location: ({truth_onset_lon:.1f}°E, {truth_onset_lat:.1f}°N)")
        print(f"  Time index: {truth_onset_time_idx}")
        print(f"  Date: {truth_onset_date}")

    # Get blocked mask for centroid computation
    blocked_mask = event_info['blocked_mask']

    # Find the truth time index for the verification date (already extracted from folder name)
    verification_truth_idx = None
    for truth_idx in range(len(truth_z500_anom.time)):
        truth_time = truth_z500_anom.time.values[truth_idx]
        truth_tuple = (truth_time.year, truth_time.month, truth_time.day)
        if truth_tuple == verification_date:
            verification_truth_idx = truth_idx
            break

    if verification_truth_idx is None:
        raise ValueError(
            f"Target date {verification_date} not found in truth data. "
            f"Truth time range: {truth_z500_anom.time.values[0]} to {truth_z500_anom.time.values[-1]}"
        )

    # Parse region bounds
    if region is None:
        region_bounds = {}
    elif isinstance(region, str):
        if region not in BLOCKING_REGIONS:
            raise ValueError(
                f"Unknown region '{region}'. Available: {list(BLOCKING_REGIONS.keys())}"
            )
        region_bounds = BLOCKING_REGIONS[region]
    elif isinstance(region, dict):
        required_keys = {'lon_min', 'lon_max', 'lat_min', 'lat_max'}
        if not required_keys.issubset(region.keys()):
            raise ValueError(f"Region dict must have keys: {required_keys}")
        region_bounds = region
    else:
        raise ValueError(f"region must be str, dict, or None, got {type(region)}")

    if verbose and region_bounds:
        print(f"\nRegion of interest: {region if isinstance(region, str) else 'custom'}")
        print(f"  Bounds: {region_bounds['lon_min']:.1f}-{region_bounds['lon_max']:.1f}°E, "
              f"{region_bounds['lat_min']:.1f}-{region_bounds['lat_max']:.1f}°N")

    # Compute blocking centroid at the verification date (within region, largest component)
    from .scoring import compute_blocking_centroid
    try:
        verification_lon, verification_lat = compute_blocking_centroid(
            blocked_mask, verification_truth_idx,
            region_lon_min=region_bounds.get('lon_min'),
            region_lon_max=region_bounds.get('lon_max'),
            region_lat_min=region_bounds.get('lat_min'),
            region_lat_max=region_bounds.get('lat_max'),
        )
    except ValueError as e:
        if verbose:
            print(f"  Warning: {e}")
            print(f"  Falling back to first blocking onset centroid")
        verification_lon, verification_lat = truth_onset_lon, truth_onset_lat

    if verbose:
        print(f"\nVerification date: {verification_date} (truth_idx={verification_truth_idx})")
        print(f"Blocking centroid at verification: ({verification_lon:.1f}°E, {verification_lat:.1f}°N)")

    # Compute RMSE for each lead time
    if verbose:
        print(f"\nComputing RMSE for {len(forecast_files)} lead times...")

    results = []

    for i, (fpath, label) in enumerate(zip(forecast_files, lead_time_labels), 1):
        if verbose:
            print(f"\n  [{i}/{len(forecast_files)}] {label}...", end=" ")

        # Load emulator anomalies
        emulator_result = load_forecast_anomalies_only(
            fpath,
            z500_clim,
            full_simulation=True
        )

        if 'error' in emulator_result:
            if verbose:
                print(f"ERROR: {emulator_result['error']}")
            results.append({
                'lead_time': label,
                'rmse_m': np.nan,
                'truth_onset_lon': truth_onset_lon,
                'truth_onset_lat': truth_onset_lat,
                'truth_onset_time_idx': truth_onset_time_idx,
                'error': emulator_result['error']
            })
            continue

        emulator_z500_anom = emulator_result['z500_anom']

        # Find the emulator time index for the common verification date
        emu_verification_idx = None
        for idx, emu_time in enumerate(emulator_z500_anom.time.values):
            emu_tuple = (emu_time.year, emu_time.month, emu_time.day)
            if emu_tuple == verification_date:
                emu_verification_idx = idx
                break

        if emu_verification_idx is None:
            if verbose:
                print(f"Verification date {verification_date} not in emulator range")
            results.append({
                'lead_time': label,
                'rmse_m': np.nan,
                'verification_date': str(verification_date),
                'block_lon': verification_lon,
                'block_lat': verification_lat,
                'truth_time_idx': verification_truth_idx,
                'emulator_time_idx': None,
                'error': f"Verification date not in emulator range"
            })
            continue

        # Compute RMSE at the blocking centroid using matched time indices
        try:
            from .scoring import extract_3x3_patch

            # Extract patches at the SAME DATE (verification date)
            truth_patch = extract_3x3_patch(
                truth_z500_anom,
                verification_lon,
                verification_lat,
                verification_truth_idx
            )
            emulator_patch = extract_3x3_patch(
                emulator_z500_anom,
                verification_lon,
                verification_lat,
                emu_verification_idx
            )

            # Compute RMSE
            diff = emulator_patch - truth_patch
            rmse = float(np.sqrt(np.mean(diff ** 2)))

            if verbose:
                print(f"RMSE = {rmse:.2f} m @ {verification_date} (emu_idx={emu_verification_idx})")

            results.append({
                'lead_time': label,
                'rmse_m': rmse,
                'block_lon': verification_lon,
                'block_lat': verification_lat,
                'matched_date': str(verification_date),
                'truth_time_idx': verification_truth_idx,
                'emulator_time_idx': emu_verification_idx
            })

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            results.append({
                'lead_time': label,
                'rmse_m': np.nan,
                'block_lon': verification_lon,
                'block_lat': verification_lat,
                'matched_date': str(verification_date),
                'truth_time_idx': verification_truth_idx,
                'emulator_time_idx': emu_verification_idx,
                'error': str(e)
            })

    # Create results DataFrame
    df_rmse = pd.DataFrame(results)

    # Sort by lead time if possible (extract numeric part)
    def extract_lead_days(label):
        if isinstance(label, str) and label.endswith('d'):
            try:
                return int(label[:-1])
            except ValueError:
                pass
        return 999  # Put non-numeric labels at end

    df_rmse['_sort_key'] = df_rmse['lead_time'].apply(extract_lead_days)
    df_rmse = df_rmse.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

    # Create metadata dict
    metadata = {
        'truth_file': truth_file,
        'truth_num_events': truth_result['num_events'],
        'truth_onset_lon': truth_onset_lon,
        'truth_onset_lat': truth_onset_lat,
        'truth_onset_time_idx': truth_onset_time_idx,
        'truth_onset_date': truth_onset_date,
        'clim_threshold_file': clim_threshold_file,
        'threshold_json_file': threshold_json_file,
    }

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("RMSE Results Summary")
        print(f"First truth onset: ({truth_onset_lon:.1f}°E, {truth_onset_lat:.1f}°N) on {truth_onset_date}")
        print("=" * 70)

        # Show columns that exist
        display_cols = ['lead_time', 'rmse_m']
        if 'matched_date' in df_rmse.columns:
            display_cols.append('matched_date')
        print(df_rmse[display_cols].to_string(index=False))

        valid_df = df_rmse.dropna(subset=['rmse_m'])
        if len(valid_df) > 0:
            best_idx = valid_df['rmse_m'].idxmin()
            worst_idx = valid_df['rmse_m'].idxmax()
            print(f"\nBest:  {valid_df.loc[best_idx, 'lead_time']} ({valid_df.loc[best_idx, 'rmse_m']:.2f} m)")
            print(f"Worst: {valid_df.loc[worst_idx, 'lead_time']} ({valid_df.loc[worst_idx, 'rmse_m']:.2f} m)")
        print("=" * 70)

    # Save to file if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'rmse_at_onset_by_leadtime.csv')
        df_rmse.to_csv(output_file, index=False)
        if verbose:
            print(f"\nResults saved to: {output_file}")

    return df_rmse, metadata
