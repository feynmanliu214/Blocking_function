#!/usr/bin/env python3
"""
Scoring Subpackage for Blocking and Heatwave Event Analysis.

This package provides three categories of scoring methods:

1. Grid-point-based scoring (blocking / z500):
   - GridpointPersistenceScorer: DG-style per-gridpoint blocking with 5-day persistence
   - GridpointIntensityScorer: strict-window maximum anomaly intensity

2. ANO-based event scoring (blocking / z500):
   - ANOScorer: Unified scorer with configurable event detection and drift penalty
   - RMSEScorer: RMSE between emulator and truth Z500 anomalies at onset centroid

3. Heatwave scoring (tas):
   - HeatwaveMeanScorer: Spatiotemporal mean of T_2m over L-day regional box

Usage:
    from forecast_analysis.scoring import ANOScorer, GridpointPersistenceScorer, GridpointIntensityScorer
    from forecast_analysis.scoring import HeatwaveMeanScorer

    # ANO-based: auto event detection with drift penalty
    scorer = ANOScorer(mode="auto", use_drift_penalty=True, gamma=5.0)

    # ANO-based: onset-based tracking
    scorer = ANOScorer(mode="onset", onset_time_idx=10)

    # Grid-point-based
    scorer = GridpointPersistenceScorer()

    # Heatwave mean scorer
    scorer = HeatwaveMeanScorer(n_days=7)

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

# Heatwave scorers
from .heatwave import HeatwaveMeanScorer

from .aggregation import (
    compute_member_scores,
    rank_ensemble_scores,
)

# Context and builder
from .context import ScorerContext, build_scorer_context

# ---------------------------------------------------------------------------
# Scorer Registry for RES experiments
# ---------------------------------------------------------------------------

# Registry mapping scorer names to their classes
SCORER_REGISTRY = {
    'ANOScorer': ANOScorer,
    'GridpointPersistenceScorer': GridpointPersistenceScorer,
    'GridpointIntensityScorer': GridpointIntensityScorer,
    'HeatwaveMeanScorer': HeatwaveMeanScorer,
}


def scorer_requires_blocking_detection(scorer_name: str) -> bool:
    """Check whether a scorer needs upstream blocking mask/event detection.

    Uses the ``requires_blocking_detection`` class attribute from the
    scorer registry.  For legacy scorer names not in the registry
    (e.g. ``IntegratedScorer``, ``DriftPenalizedScorer``), defaults to True.
    """
    scorer_cls = SCORER_REGISTRY.get(scorer_name)
    if scorer_cls is None:
        return True
    return scorer_cls.requires_blocking_detection


# ---------------------------------------------------------------------------
# Legacy scorer name mappings
# ---------------------------------------------------------------------------

_KNOWN_LEGACY_NAMES = {"IntegratedScorer", "DriftPenalizedScorer", "RMSEScorer"}

_LEGACY_REQUIRED_VARIABLE = {
    "IntegratedScorer": "z500",
    "DriftPenalizedScorer": "z500",
    "RMSEScorer": "z500",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_scorer_name(scorer_name: str) -> None:
    """Raise ``ValueError`` if *scorer_name* is not a known scorer.

    Accepts both registry names and legacy aliases.
    """
    if scorer_name in SCORER_REGISTRY:
        return
    if scorer_name in _KNOWN_LEGACY_NAMES:
        return
    available = sorted(set(list(SCORER_REGISTRY.keys()) + list(_KNOWN_LEGACY_NAMES)))
    raise ValueError(
        f"Unknown scorer: '{scorer_name}'. Known scorers: {available}"
    )


def scorer_requires_anomaly(scorer_name: str) -> bool:
    """Return whether *scorer_name* operates on anomaly fields.

    Legacy names are assumed to require anomalies (z500-based).
    """
    scorer_cls = SCORER_REGISTRY.get(scorer_name)
    if scorer_cls is None:
        # Legacy names all require anomalies
        return True
    return getattr(scorer_cls, 'requires_anomaly', True)


def scorer_required_variable(scorer_name: str) -> str:
    """Return the required climate variable for *scorer_name*.

    Returns ``"z500"`` for legacy names.
    """
    scorer_cls = SCORER_REGISTRY.get(scorer_name)
    if scorer_cls is not None:
        return getattr(scorer_cls, 'required_variable', 'z500')
    return _LEGACY_REQUIRED_VARIABLE.get(scorer_name, 'z500')


def validate_scorer_region(scorer_name: str, region: str) -> None:
    """Raise ``ValueError`` if *region* is not allowed for *scorer_name*."""
    scorer_cls = SCORER_REGISTRY.get(scorer_name)
    if scorer_cls is None:
        # Legacy names — no restriction (caller is responsible)
        return
    allowed = getattr(scorer_cls, 'allowed_regions', None)
    if allowed is not None and region not in allowed:
        raise ValueError(
            f"Region '{region}' is not allowed for {scorer_name}. "
            f"Allowed regions: {list(allowed)}"
        )


def validate_scorer_variable(scorer_name: str, variable: str) -> None:
    """Raise ``ValueError`` if *variable* does not match *scorer_name*."""
    validate_scorer_name(scorer_name)  # fail-fast on unknown names
    required = scorer_required_variable(scorer_name)
    if variable != required:
        raise ValueError(
            f"Variable '{variable}' does not match {scorer_name} "
            f"(requires '{required}')"
        )


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
    # Heatwave
    "HeatwaveMeanScorer",
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
    "scorer_requires_blocking_detection",
    "list_available_scorers",
    # Validation helpers
    "validate_scorer_name",
    "scorer_requires_anomaly",
    "scorer_required_variable",
    "validate_scorer_region",
    "validate_scorer_variable",
    # Context
    "ScorerContext",
    "build_scorer_context",
]
