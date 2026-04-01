# forecast_analysis/scoring/context.py
"""Single authoritative scorer-construction layer.

All forecast-stage files (PFS.py, PanguPlasimFS.py) and the final scoring
file (compute_plasim_scores.py) must call build_scorer_context() before
any scoring loop.  No direct scorer_json parsing is permitted outside
this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BlockingScorer


@dataclass(frozen=True)
class ScorerContext:
    """Fully resolved scorer context produced by build_scorer_context().

    Attributes
    ----------
    scorer : BlockingScorer
        Pre-instantiated scorer object (with region bounds baked in where
        required, e.g. ANOScorer).
    variable : str
        Climate variable ("z500" or "tas").
    region_bounds : dict
        Scorer-appropriate region selector.  HeatwaveMeanScorer:
        ``{lon: [...], lat: [...]}``.  All others: bounding box
        ``{lon_min, lon_max, lat_min, lat_max}``.
        Always derived from ``regions.json`` — never from hardcoded
        scorer-class constants.
    onset_time_idx : int
        Daily time index for the scoring window start.
    scorer_params : dict
        Merged, normalised, legacy-alias-resolved params.  Used by
        score_from_anomaly() implementations to recover scorer-specific
        settings (n_days, fallback_to_nonblocked, etc.).
    """
    scorer: BlockingScorer
    variable: str
    region_bounds: dict
    onset_time_idx: int
    scorer_params: dict


# Legacy scorer name → default params to inject into merged_params
_LEGACY_ALIAS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "IntegratedScorer":     {"mode": "onset"},
    "DriftPenalizedScorer": {"mode": "auto", "use_drift_penalty": True},
}

# Legacy scorer name → canonical class name in SCORER_REGISTRY
_LEGACY_CANONICAL_CLASS: Dict[str, str] = {
    "IntegratedScorer":     "ANOScorer",
    "DriftPenalizedScorer": "ANOScorer",
}


def build_scorer_context(
    scorer_json: dict,
    region: str,
    regions_json: Path,
    onset_time_idx: Optional[int] = None,
) -> ScorerContext:
    """Parse scorer config into a fully resolved ScorerContext.

    Parameters
    ----------
    scorer_json : dict
        The ``scorer`` block from the experiment JSON.
    region : str
        Top-level ``region`` field from the experiment config.
    regions_json : Path
        Path to ``regions.json`` (single source of truth for region coords).
    onset_time_idx : int, optional
        Explicit window-start (daily index).  When omitted, derived from
        merged params (``onset_time_idx`` or legacy ``lead_time`` key).
        PanguPlasimFS and PFS pass ``config['lead_time']`` here explicitly.

    Returns
    -------
    ScorerContext
    """
    from . import validate_scorer_name, validate_scorer_variable, validate_scorer_region

    # 1. Parse required fields
    scorer_name: str = scorer_json.get("name", "")
    if not scorer_name:
        raise ValueError("'scorer.name' is required.")
    variable: str = scorer_json.get("variable", "")
    if not variable:
        raise ValueError("'scorer.variable' is required.")

    # 2. Validate against registry (accepts legacy aliases too)
    validate_scorer_name(scorer_name)
    validate_scorer_variable(scorer_name, variable)
    # Region validation: use the canonical class name so legacy aliases
    # (IntegratedScorer, DriftPenalizedScorer) get the same region restriction
    # as ANOScorer.  validate_scorer_region() returns without restriction for
    # legacy names because they have no class in SCORER_REGISTRY.
    _canonical_for_validation = _LEGACY_CANONICAL_CLASS.get(scorer_name, scorer_name)
    validate_scorer_region(_canonical_for_validation, region)

    # 3. Merge params: top-level keys + nested "params" dict; nested wins
    nested: dict = scorer_json.get("params", {})
    if not isinstance(nested, dict):
        nested = {}
    top_level: dict = {
        k: v for k, v in scorer_json.items()
        if k not in {"name", "params", "variable"}
    }
    merged_params: Dict[str, Any] = {**top_level, **nested}

    # 4. Resolve legacy alias defaults into merged_params so that
    #    ctx.scorer_params is canonical (e.g. "mode"="onset" for IntegratedScorer)
    if scorer_name in _LEGACY_ALIAS_DEFAULTS:
        alias_defaults = _LEGACY_ALIAS_DEFAULTS[scorer_name]
        # alias defaults do NOT override explicit user params
        merged_params = {**alias_defaults, **merged_params}

    # 5. Apply policy normalization (fallback_to_nonblocked for GridpointIntensityScorer)
    merged_params = _apply_score_policy(scorer_name, merged_params)

    # 6. Resolve onset_time_idx
    if onset_time_idx is None:
        onset_time_idx = _resolve_onset_time_idx(merged_params)

    # 7. Region selector — always from regions.json
    region_bounds = _resolve_region_bounds(scorer_name, region, regions_json)

    # 8. Instantiate scorer
    scorer = _instantiate_scorer(scorer_name, merged_params, onset_time_idx, region_bounds)

    return ScorerContext(
        scorer=scorer,
        variable=variable,
        region_bounds=region_bounds,
        onset_time_idx=onset_time_idx,
        scorer_params=merged_params,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_score_policy(scorer_name: str, params: dict) -> dict:
    """Normalize GridpointIntensityScorer fallback_to_nonblocked."""
    if scorer_name != "GridpointIntensityScorer":
        return params
    params = dict(params)
    raw = params.get("fallback_to_nonblocked", None)
    if raw is None:
        params["fallback_to_nonblocked"] = False
    elif isinstance(raw, str):
        val = raw.strip().lower()
        if val != "always":
            params["fallback_to_nonblocked"] = val in {"1", "true", "yes", "y", "t"}
        # else preserve "always" as string
    else:
        params["fallback_to_nonblocked"] = bool(raw)
    return params


def _resolve_onset_time_idx(params: dict) -> int:
    """Derive onset_time_idx from params; prefer legacy lead_time key."""
    if "lead_time" in params and params["lead_time"] is not None:
        return int(params["lead_time"])
    return int(params.get("onset_time_idx", 0) or 0)


def _resolve_region_bounds(scorer_name: str, region: str, regions_json: Path) -> dict:
    """Load region selector from regions.json (single source of truth).

    HeatwaveMeanScorer → explicit point lists {lon: [...], lat: [...]}.
    All others → bounding box {lon_min, lon_max, lat_min, lat_max}.
    """
    with regions_json.open("r", encoding="utf-8") as f:
        regions = json.load(f)
    region_cfg = regions.get(region)
    if not isinstance(region_cfg, dict):
        raise ValueError(f"Region '{region}' not found in {regions_json}")
    lons = region_cfg.get("lon", [])
    lats = region_cfg.get("lat", [])
    if not lons or not lats:
        raise ValueError(f"Region '{region}' has no lon/lat values in {regions_json}")

    if scorer_name == "HeatwaveMeanScorer":
        return {
            "lon": [float(v) for v in lons],
            "lat": [float(v) for v in lats],
        }
    return {
        "lon_min": float(min(lons)),
        "lon_max": float(max(lons)),
        "lat_min": float(min(lats)),
        "lat_max": float(max(lats)),
    }


def _instantiate_scorer(
    scorer_name: str,
    params: dict,
    onset_time_idx: int,
    region_bounds: dict,
) -> BlockingScorer:
    """Instantiate scorer, resolving legacy aliases and wiring in region/onset."""
    from . import SCORER_REGISTRY

    canonical_name = _LEGACY_CANONICAL_CLASS.get(scorer_name, scorer_name)
    cls = SCORER_REGISTRY[canonical_name]

    import inspect
    sig = inspect.signature(cls.__init__)
    known_params = set(sig.parameters) - {"self"}

    # Build init kwargs from merged params, restricted to known constructor params
    init_kwargs: Dict[str, Any] = {
        k: v for k, v in params.items()
        if k in known_params and k not in {"lead_time"}
    }

    # ANOScorer(mode="onset") requires onset_time_idx in its constructor
    if canonical_name == "ANOScorer" and init_kwargs.get("mode") == "onset":
        init_kwargs["onset_time_idx"] = onset_time_idx

    # ANOScorer stores region bounds in its constructor
    if canonical_name == "ANOScorer":
        if "region_lon_min" in known_params:
            init_kwargs["region_lon_min"] = region_bounds["lon_min"]
            init_kwargs["region_lon_max"] = region_bounds["lon_max"]
            init_kwargs["region_lat_min"] = region_bounds["lat_min"]
            init_kwargs["region_lat_max"] = region_bounds["lat_max"]

    return cls(**init_kwargs)


__all__ = ["ScorerContext", "build_scorer_context"]
