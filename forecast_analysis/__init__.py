#!/usr/bin/env python3
"""
Forecast Blocking Analysis Package.

This package provides a modular framework for analyzing blocking events in
climate model forecast outputs, including:
- Ensemble forecast analysis
- Deterministic forecast analysis across multiple lead times
- Blocking event animation generation
- Extensible blocking score computation and ranking

Module Structure
----------------
- data_loading: Data extraction and preprocessing
- blocking_detection: Single forecast processing
- animation: GIF animation generation
- ensemble_analysis: Ensemble-specific workflows
- deterministic_analysis: Deterministic forecast workflows
- scoring: Extensible scoring subpackage
  - base: Abstract BlockingScorer interface
  - ano: ANO-based scorers (ANOScorer, RMSEScorer)
  - gridpoint: Grid-point-based scorers (GridpointPersistenceScorer, GridpointIntensityScorer)
  - aggregation: Member-level score aggregation

Usage Examples
--------------
# Analyze ensemble forecast
>>> from forecast_analysis import analyze_ensemble_forecast, rank_ensemble_scores
>>> results = analyze_ensemble_forecast(
...     ensemble_dir='/path/to/ensemble/',
...     clim_threshold_file='/path/to/climatology.nc',
...     threshold_json_file='/path/to/thresholds.json',
...     output_dir='/path/to/output/',
...     create_animations=True
... )
>>> df_ranked = rank_ensemble_scores(results, threshold_score=300000)

# Analyze deterministic forecast
>>> from forecast_analysis import analyze_deterministic_forecast
>>> results = analyze_deterministic_forecast(
...     forecast_files=['/path/to/forecast_lead1.nc', ...],
...     clim_threshold_file='/path/to/climatology.nc',
...     threshold_json_file='/path/to/thresholds.json',
...     output_dir='/path/to/output/'
... )

# Use custom scorer
>>> from forecast_analysis.scoring import ANOScorer, compute_member_scores
>>> scorer = ANOScorer(mode="auto", use_drift_penalty=True, gamma=10.0)
>>> df = compute_member_scores(results, scorer=scorer)

Author: AI-RES Project
"""

# Data loading
from .data_loading import (
    load_climatology_and_thresholds,
    extract_z500_daily,
    compute_anomalies_with_climatology,
)

# Blocking detection
from .blocking_detection import process_single_forecast, load_forecast_anomalies_only

# Animation
from .animation import create_forecast_animation

# Ensemble analysis
from .ensemble_analysis import (
    analyze_ensemble_forecast,
    analyze_and_rank_ensemble,
)

# Deterministic analysis
from .deterministic_analysis import (
    analyze_deterministic_forecast,
    analyze_and_rank_deterministic,
    compute_rmse_by_leadtime,
    PLASIM_TRUTH_PATH_PATTERN,
    BLOCKING_REGIONS,
)

# Scoring (re-export main items for convenience)
from .scoring import (
    BlockingScorer,
    ANOScorer,
    RMSEScorer,
    GridpointPersistenceScorer,
    GridpointIntensityScorer,
    compute_blocking_scores,
    compute_integrated_score,
    compute_rmse_score,
    compute_gridpoint_blocking_score,
    compute_gridpoint_intensity_score,
    compute_blocking_centroid,
    compute_unweighted_centroid,
    compute_onset_centroid_from_event_info,
    extract_3x3_patch,
    compute_member_scores,
    rank_ensemble_scores,
    REGION_BOUNDS,
    get_scorer,
    compute_res_score,
)

__all__ = [
    # Data loading
    "load_climatology_and_thresholds",
    "extract_z500_daily",
    "compute_anomalies_with_climatology",
    # Blocking detection
    "process_single_forecast",
    "load_forecast_anomalies_only",
    # Animation
    "create_forecast_animation",
    # Ensemble analysis
    "analyze_ensemble_forecast",
    "analyze_and_rank_ensemble",
    # Deterministic analysis
    "analyze_deterministic_forecast",
    "analyze_and_rank_deterministic",
    "compute_rmse_by_leadtime",
    "PLASIM_TRUTH_PATH_PATTERN",
    "BLOCKING_REGIONS",
    # Scoring
    "BlockingScorer",
    "ANOScorer",
    "RMSEScorer",
    "GridpointPersistenceScorer",
    "GridpointIntensityScorer",
    "compute_blocking_scores",
    "compute_integrated_score",
    "compute_rmse_score",
    "compute_gridpoint_blocking_score",
    "compute_gridpoint_intensity_score",
    "compute_blocking_centroid",
    "compute_unweighted_centroid",
    "compute_onset_centroid_from_event_info",
    "extract_3x3_patch",
    "compute_member_scores",
    "rank_ensemble_scores",
    "REGION_BOUNDS",
    "get_scorer",
    "compute_res_score",
]

__version__ = "1.0.0"
