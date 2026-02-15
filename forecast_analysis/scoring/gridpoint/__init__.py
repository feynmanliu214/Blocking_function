#!/usr/bin/env python3
"""
Grid-point-based Blocking Scorers Subpackage.

This subpackage provides grid-point-based blocking scoring methods:

- GridpointPersistenceScorer: mean blocked-area percentage
- GridpointIntensityScorer: moving-average maximum intensity

Usage:
    from forecast_analysis.scoring.gridpoint import (
        GridpointPersistenceScorer,
        GridpointIntensityScorer,
    )

    scorer = GridpointPersistenceScorer()
    pct = scorer.compute_score(z500, start_time="2000-01-01", duration_days=7, region="NorthAtlantic")

Author: AI-RES Project
"""

from .persistence_scorer import (
    GridpointPersistenceScorer,
    compute_gridpoint_blocking_score,
    # Utility functions
    load_climatology,
    load_thresholds,
    compute_anomalies_from_climatology,
    apply_monthly_threshold,
    compute_dg_blocking_mask,
    create_region_mask,
    compute_area_weights,
    # Constants
    REGION_BOUNDS,
    DEFAULT_CLIMATOLOGY_PATH,
    DEFAULT_THRESHOLDS_PATH,
)
from .intensity_scorer import (
    GridpointIntensityScorer,
    compute_gridpoint_intensity_score,
)

__all__ = [
    # Main scorer
    "GridpointPersistenceScorer",
    "GridpointIntensityScorer",
    # Convenience functions
    "compute_gridpoint_blocking_score",
    "compute_gridpoint_intensity_score",
    # Utility functions
    "load_climatology",
    "load_thresholds",
    "compute_anomalies_from_climatology",
    "apply_monthly_threshold",
    "compute_dg_blocking_mask",
    "create_region_mask",
    "compute_area_weights",
    # Constants
    "REGION_BOUNDS",
    "DEFAULT_CLIMATOLOGY_PATH",
    "DEFAULT_THRESHOLDS_PATH",
]
