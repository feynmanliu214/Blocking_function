#!/usr/bin/env python3
"""
ANO-based Blocking Scorers Subpackage.

This subpackage provides ANO-based (Anomaly) blocking event scoring methods:

- ANOScorer: Unified scorer with configurable event detection and drift penalty
- RMSEScorer: RMSE between emulator and truth Z500 anomalies at onset centroid

Usage:
    from forecast_analysis.scoring.ano import ANOScorer, RMSEScorer

    # Auto mode with drift penalty
    scorer = ANOScorer(mode="auto", use_drift_penalty=True, gamma=5.0)
    df = scorer.compute_event_scores(z500, event_info)

    # Onset mode
    scorer = ANOScorer(mode="onset", onset_time_idx=10)
    df = scorer.compute_event_scores(z500, event_info)

Author: AI-RES Project
"""

from .scorer import (
    ANOScorer,
    compute_blocking_scores,
    compute_integrated_score,
)
from .rmse_scorer import (
    RMSEScorer,
    compute_rmse_score,
    compute_rmse,
    compute_onset_centroid_from_event_info,
)
from .utils import (
    REGION_BOUNDS,
    create_spatial_weight_mask,
    compute_blocking_centroid,
    compute_unweighted_centroid,
    select_block_at_onset,
    select_event_at_onset,
    track_block_through_time,
    extract_3x3_patch,
    _integrate_over_masks,
    _centroid_in_roi,
    _build_roi_mask,
)

__all__ = [
    # Main scorers
    "ANOScorer",
    "RMSEScorer",
    # Convenience functions
    "compute_blocking_scores",
    "compute_integrated_score",
    "compute_rmse_score",
    "compute_rmse",
    "compute_onset_centroid_from_event_info",
    # Constants
    "REGION_BOUNDS",
    # Utilities
    "create_spatial_weight_mask",
    "compute_blocking_centroid",
    "compute_unweighted_centroid",
    "select_block_at_onset",
    "select_event_at_onset",
    "track_block_through_time",
    "extract_3x3_patch",
    "_integrate_over_masks",
    "_centroid_in_roi",
    "_build_roi_mask",
]
