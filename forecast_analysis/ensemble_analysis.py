#!/usr/bin/env python3
"""
Ensemble Forecast Analysis for Blocking Events.

This module provides functions for analyzing blocking events across
ensemble forecast members.

Author: AI-RES Project
"""

import glob
import os
from typing import Dict, Optional, Tuple

import pandas as pd

from .data_loading import load_climatology_and_thresholds
from .blocking_detection import process_single_forecast
from .animation import create_forecast_animation
from .scoring import rank_ensemble_scores, BlockingScorer


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
    from a long reference run.

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
        print(f"   Climatology loaded: shape={z500_clim.shape}")
        print(f"   Thresholds loaded: {threshold_90}")

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

        # Process this member (always use full simulation)
        result = process_single_forecast(
            nc_file,
            z500_clim,
            threshold_90,
            min_area=min_area,
            min_duration=min_duration,
            min_overlap=min_overlap,
            full_simulation=True
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
                print(f"  Animation saved: {gif_path}")

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


def analyze_and_rank_ensemble(
    ensemble_dir: str,
    clim_threshold_file: str,
    threshold_json_file: str,
    output_dir: Optional[str] = None,
    file_pattern: str = '*_postproc.nc',
    threshold_score: Optional[float] = None,
    scorer: Optional[BlockingScorer] = None,
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
    scorer : BlockingScorer, optional
        Scorer instance to use. If None, uses DriftPenalizedScorer.
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
    """
    # Step 1: Analyze ensemble
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
        scorer=scorer,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        gamma=gamma,
        verbose=verbose
    )

    return results, df_ranked
