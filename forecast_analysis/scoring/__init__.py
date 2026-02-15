#!/usr/bin/env python3
"""
Scoring Subpackage for Blocking Event Analysis.

This package provides two categories of scoring methods:

1. Grid-point-based scoring:
   - GridpointPersistenceScorer: DG-style per-gridpoint blocking with 5-day persistence
   - GridpointIntensityScorer: moving-average maximum anomaly intensity on blocked points

2. ANO-based event scoring:
   - ANOScorer: Unified scorer with configurable event detection and drift penalty
   - RMSEScorer: RMSE between emulator and truth Z500 anomalies at onset centroid

Usage:
    from forecast_analysis.scoring import ANOScorer, GridpointPersistenceScorer, GridpointIntensityScorer

    # ANO-based: auto event detection with drift penalty
    scorer = ANOScorer(mode="auto", use_drift_penalty=True, gamma=5.0)

    # ANO-based: onset-based tracking
    scorer = ANOScorer(mode="onset", onset_time_idx=10)

    # Grid-point-based
    scorer = GridpointPersistenceScorer()

Author: AI-RES Project
"""

from .base import BlockingScorer

# Grid-point-based scorers
from .gridpoint import (
    GridpointPersistenceScorer,
    GridpointIntensityScorer,
    compute_gridpoint_blocking_score,
    compute_gridpoint_intensity_score,
)

# ANO-based scorers
from .ano import (
    ANOScorer,
    RMSEScorer,
    compute_blocking_scores,
    compute_integrated_score,
    compute_rmse_score,
    REGION_BOUNDS,
    # Utility functions
    compute_blocking_centroid,
    compute_unweighted_centroid,
    compute_onset_centroid_from_event_info,
    select_block_at_onset,
    select_event_at_onset,
    track_block_through_time,
    extract_3x3_patch,
)

from .aggregation import (
    compute_member_scores,
    rank_ensemble_scores,
)

# ---------------------------------------------------------------------------
# Scorer Registry for RES experiments
# ---------------------------------------------------------------------------

# Registry mapping scorer names to their classes
SCORER_REGISTRY = {
    'ANOScorer': ANOScorer,
    'GridpointPersistenceScorer': GridpointPersistenceScorer,
    'GridpointIntensityScorer': GridpointIntensityScorer,
}


def get_scorer(name: str, **params):
    """
    Factory function to instantiate a scorer by name.

    Parameters
    ----------
    name : str
        Name of the scorer (e.g., 'ANOScorer', 'GridpointPersistenceScorer').
    **params
        Parameters to pass to the scorer constructor.

    Returns
    -------
    BlockingScorer
        An instance of the requested scorer.

    Raises
    ------
    ValueError
        If the scorer name is not in the registry.

    Example
    -------
    >>> scorer = get_scorer('ANOScorer', mode='onset', onset_time_idx=5, n_days=5)
    >>> scorer = get_scorer('ANOScorer', mode='auto', use_drift_penalty=True, gamma=5.0)
    """
    if name not in SCORER_REGISTRY:
        available = list(SCORER_REGISTRY.keys())
        raise ValueError(f"Unknown scorer: '{name}'. Available scorers: {available}")
    return SCORER_REGISTRY[name](**params)


def compute_res_score(
    scorer_name: str,
    scorer_params: dict,
    z500_anom,
    event_info: dict,
    region_bounds: dict,
    onset_time_idx: int = None,
    threshold_90: dict = None,
) -> float:
    """
    Unified function to compute a scalar score for RES experiments.

    This function provides a consistent interface for different scorers,
    returning a single scalar value suitable for RES resampling.

    Parameters
    ----------
    scorer_name : str
        Name of the scorer ('ANOScorer', 'GridpointPersistenceScorer',
        or 'GridpointIntensityScorer').
        For backward compatibility, 'IntegratedScorer' and 'DriftPenalizedScorer'
        are mapped to ANOScorer with appropriate settings.
    scorer_params : dict
        Parameters for the scorer constructor.
    z500_anom : xr.DataArray
        Z500 anomaly data with dimensions (time, lat, lon).
    event_info : dict
        Event information dict containing 'event_mask' and/or 'blocked_mask'.
        Not used by GridpointPersistenceScorer.
    region_bounds : dict
        Region bounds with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
    onset_time_idx : int, optional
        Time index of blocking onset. Required for onset mode and
        GridpointPersistenceScorer if not already in scorer_params.
    threshold_90 : dict, optional
        Monthly thresholds mapping month (int) -> threshold (float).
        Required for gridpoint scorers.

    Returns
    -------
    float
        Scalar score value. For ANOScorer onset mode this is B_int.
        For ANOScorer auto mode this is the max P_total across events.
        For GridpointPersistenceScorer this is mean blocked percentage (0-100).
        For GridpointIntensityScorer this is max moving-average intensity.

    Example
    -------
    >>> score = compute_res_score(
    ...     scorer_name='ANOScorer',
    ...     scorer_params={'mode': 'onset', 'n_days': 5},
    ...     z500_anom=z500_anom,
    ...     event_info={'event_mask': event_mask},
    ...     region_bounds={'lon_min': -60, 'lon_max': 0, 'lat_min': 55, 'lat_max': 75},
    ...     onset_time_idx=5,
    ... )
    """
    import numpy as np
    import xarray as xr

    lon_min = region_bounds['lon_min']
    lon_max = region_bounds['lon_max']
    lat_min = region_bounds['lat_min']
    lat_max = region_bounds['lat_max']

    # Handle backward compatibility for old scorer names
    if scorer_name == 'IntegratedScorer':
        # Map to ANOScorer with onset mode
        params = scorer_params.copy()
        if onset_time_idx is not None and 'onset_time_idx' not in params:
            params['onset_time_idx'] = onset_time_idx

        n_days = params.get('n_days', 5)
        oti = params.get('onset_time_idx', 0)

        result = compute_integrated_score(
            z500_anom=z500_anom,
            onset_time_idx=oti,
            n_days=n_days,
            event_mask=event_info.get('event_mask'),
            blocked_mask=event_info.get('blocked_mask'),
            region_lon_min=lon_min,
            region_lon_max=lon_max,
            region_lat_min=lat_min,
            region_lat_max=lat_max,
        )
        return result['B_int']

    elif scorer_name == 'DriftPenalizedScorer':
        # Map to ANOScorer with auto mode and drift penalty
        gamma = scorer_params.get('gamma', 5.0)
        integration_days = scorer_params.get('integration_days', 5)
        scorer = ANOScorer(
            mode="auto",
            use_drift_penalty=True,
            gamma=gamma,
            integration_days=integration_days,
            region_lon_min=lon_min,
            region_lon_max=lon_max,
            region_lat_min=lat_min,
            region_lat_max=lat_max,
        )

        # Need z500 (not just anomalies) for the scorer interface
        # But we can construct event_info with z500_anom
        event_info_copy = event_info.copy()
        event_info_copy['z500_anom'] = z500_anom

        # Also need blocked_mask for ANOScorer
        if 'blocked_mask' not in event_info_copy:
            # If we only have event_mask, convert it to blocked_mask
            event_mask = event_info.get('event_mask')
            if event_mask is not None:
                blocked_mask = (event_mask > 0).astype(float)
                blocked_mask = xr.DataArray(
                    blocked_mask,
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': z500_anom.time,
                        'lat': z500_anom.lat,
                        'lon': z500_anom.lon,
                    }
                )
                event_info_copy['blocked_mask'] = blocked_mask

        df = scorer.compute_event_scores(
            z500=z500_anom,  # Uses z500_anom from event_info anyway
            event_info=event_info_copy,
        )

        if len(df) == 0:
            return 0.0
        return float(df['P_total'].max())

    elif scorer_name == 'ANOScorer':
        params = scorer_params.copy()

        # Determine mode
        mode = params.get('mode', 'auto')

        if mode == 'onset':
            # Onset mode - similar to old IntegratedScorer
            if onset_time_idx is not None and 'onset_time_idx' not in params:
                params['onset_time_idx'] = onset_time_idx

            n_days = params.get('n_days', params.get('integration_days', 5))
            oti = params.get('onset_time_idx', 0)

            result = compute_integrated_score(
                z500_anom=z500_anom,
                onset_time_idx=oti,
                n_days=n_days,
                event_mask=event_info.get('event_mask'),
                blocked_mask=event_info.get('blocked_mask'),
                region_lon_min=lon_min,
                region_lon_max=lon_max,
                region_lat_min=lat_min,
                region_lat_max=lat_max,
            )
            return result['B_int']

        else:
            # Auto mode
            gamma = params.get('gamma', 5.0)
            use_drift_penalty = params.get('use_drift_penalty', True)
            integration_days = params.get('integration_days', params.get('n_days', 5))

            scorer = ANOScorer(
                mode="auto",
                use_drift_penalty=use_drift_penalty,
                gamma=gamma,
                integration_days=integration_days,
                region_lon_min=lon_min,
                region_lon_max=lon_max,
                region_lat_min=lat_min,
                region_lat_max=lat_max,
            )

            event_info_copy = event_info.copy()
            event_info_copy['z500_anom'] = z500_anom

            if 'blocked_mask' not in event_info_copy:
                event_mask = event_info.get('event_mask')
                if event_mask is not None:
                    blocked_mask = (event_mask > 0).astype(float)
                    blocked_mask = xr.DataArray(
                        blocked_mask,
                        dims=['time', 'lat', 'lon'],
                        coords={
                            'time': z500_anom.time,
                            'lat': z500_anom.lat,
                            'lon': z500_anom.lon,
                        }
                    )
                    event_info_copy['blocked_mask'] = blocked_mask

            df = scorer.compute_event_scores(
                z500=z500_anom,
                event_info=event_info_copy,
            )

            if len(df) == 0:
                return 0.0

            primary_col = scorer.get_primary_score_column()
            return float(df[primary_col].max())

    elif scorer_name in ('GridpointPersistenceScorer', 'GridpointIntensityScorer'):
        # Gridpoint scorers use DG-style per-gridpoint blocking and monthly thresholds.
        # They do not use event_info and work directly with anomalies.
        if threshold_90 is None:
            raise ValueError(
                f"{scorer_name} requires threshold_90 parameter. "
                "Pass the monthly thresholds dict to compute_res_score()."
            )

        n_days = scorer_params.get('n_days', 5)
        min_persistence = scorer_params.get('min_persistence', 5)

        if onset_time_idx is None:
            onset_time_idx = scorer_params.get('onset_time_idx', 0)

        if scorer_name == 'GridpointPersistenceScorer':
            scorer = GridpointPersistenceScorer(min_persistence=min_persistence)
            return scorer.compute_score_from_anomalies(
                z500_anom=z500_anom,
                threshold_90=threshold_90,
                onset_time_idx=onset_time_idx,
                duration_days=n_days,
                region_lon_min=lon_min,
                region_lon_max=lon_max,
                region_lat_min=lat_min,
                region_lat_max=lat_max,
            )

        running_mean_days = scorer_params.get('running_mean_days')
        fallback_to_nonblocked = scorer_params.get('fallback_to_nonblocked', False)
        scorer = GridpointIntensityScorer(min_persistence=min_persistence)
        return scorer.compute_intensity_score_from_anomalies(
            z500_anom=z500_anom,
            threshold_90=threshold_90,
            onset_time_idx=onset_time_idx,
            duration_days=n_days,
            region_lon_min=lon_min,
            region_lon_max=lon_max,
            region_lat_min=lat_min,
            region_lat_max=lat_max,
            running_mean_days=running_mean_days,
            fallback_to_nonblocked=fallback_to_nonblocked,
        )

    else:
        raise ValueError(f"Scorer '{scorer_name}' not supported for RES. "
                         f"Available: ['ANOScorer', 'GridpointPersistenceScorer', "
                         f"'GridpointIntensityScorer', 'IntegratedScorer', "
                         f"'DriftPenalizedScorer']")


def list_available_scorers():
    """Return list of available scorer names."""
    return list(SCORER_REGISTRY.keys())


__all__ = [
    # Base class
    "BlockingScorer",
    # Grid-point-based
    "GridpointPersistenceScorer",
    "GridpointIntensityScorer",
    "compute_gridpoint_blocking_score",
    "compute_gridpoint_intensity_score",
    # ANO-based
    "ANOScorer",
    "RMSEScorer",
    "compute_blocking_scores",
    "compute_integrated_score",
    "compute_rmse_score",
    # Utilities
    "REGION_BOUNDS",
    "compute_blocking_centroid",
    "compute_unweighted_centroid",
    "compute_onset_centroid_from_event_info",
    "select_block_at_onset",
    "select_event_at_onset",
    "track_block_through_time",
    "extract_3x3_patch",
    # Aggregation
    "compute_member_scores",
    "rank_ensemble_scores",
    # Registry
    "SCORER_REGISTRY",
    "get_scorer",
    "compute_res_score",
    "list_available_scorers",
]
