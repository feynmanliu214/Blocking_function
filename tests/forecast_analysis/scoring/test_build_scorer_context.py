# tests/forecast_analysis/scoring/test_build_scorer_context.py
import json
from pathlib import Path
import pytest
from forecast_analysis.scoring.context import build_scorer_context, ScorerContext
from forecast_analysis.scoring import GridpointIntensityScorer, HeatwaveMeanScorer, ANOScorer


REGIONS_JSON_CONTENT = {
    "NorthAtlantic": {"lon": [-60.0, 0.0], "lat": [55.0, 75.0]},
    "Chicago": {
        "lon": [270.0, 272.8125, 275.625],
        "lat": [37.67308963, 40.46364818, 43.25419467],
    },
    "France": {
        "lon": [0.0, 2.8125, 5.625],
        "lat": [48.83524097, 46.04472663, 43.25419467],
    },
}


@pytest.fixture
def regions_json(tmp_path):
    p = tmp_path / "regions.json"
    p.write_text(json.dumps(REGIONS_JSON_CONTENT))
    return p


def test_returns_scorer_context(regions_json):
    scorer_json = {"name": "GridpointIntensityScorer", "variable": "z500",
                   "params": {"n_days": 7, "min_persistence": 5}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=3)
    assert isinstance(ctx, ScorerContext)
    assert isinstance(ctx.scorer, GridpointIntensityScorer)
    assert ctx.variable == "z500"
    assert ctx.onset_time_idx == 3


# --- Params merge ---

def test_full_params_merge_top_level_merged(regions_json):
    """Top-level keys outside name/variable/params are merged into scorer_params."""
    scorer_json = {
        "name": "GridpointIntensityScorer",
        "variable": "z500",
        "fallback_to_nonblocked": "always",  # top-level
        "params": {"n_days": 7},
    }
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=0)
    assert ctx.scorer_params["fallback_to_nonblocked"] == "always"
    assert ctx.scorer_params["n_days"] == 7


def test_nested_params_win_on_collision(regions_json):
    """Nested params take precedence over top-level keys."""
    scorer_json = {
        "name": "GridpointIntensityScorer",
        "variable": "z500",
        "n_days": 5,
        "params": {"n_days": 7},  # nested wins
    }
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=0)
    assert ctx.scorer_params["n_days"] == 7


# --- Policy normalization ---

def test_policy_normalizes_fallback_missing(regions_json):
    """GridpointIntensityScorer: missing fallback_to_nonblocked defaults to False."""
    scorer_json = {"name": "GridpointIntensityScorer", "variable": "z500",
                   "params": {"n_days": 7, "min_persistence": 5}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=0)
    assert ctx.scorer_params["fallback_to_nonblocked"] == False


def test_policy_preserves_configured_always(regions_json):
    scorer_json = {"name": "GridpointIntensityScorer", "variable": "z500",
                   "params": {"n_days": 7, "fallback_to_nonblocked": "always"}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=0)
    assert ctx.scorer_params["fallback_to_nonblocked"] == "always"


def test_policy_not_applied_to_other_scorers(regions_json):
    scorer_json = {"name": "HeatwaveMeanScorer", "variable": "tas", "params": {"n_days": 7}}
    ctx = build_scorer_context(scorer_json, "Chicago", regions_json, onset_time_idx=0)
    assert "fallback_to_nonblocked" not in ctx.scorer_params


# --- Legacy alias ---

def test_integrated_scorer_alias_produces_ano_scorer(regions_json):
    """IntegratedScorer is an alias for ANOScorer(mode='onset')."""
    scorer_json = {"name": "IntegratedScorer", "variable": "z500", "params": {"n_days": 5}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=7)
    assert isinstance(ctx.scorer, ANOScorer)
    assert ctx.scorer_params.get("mode") == "onset"


def test_integrated_scorer_onset_time_idx_in_scorer(regions_json):
    """ANOScorer(mode='onset') is instantiated with the resolved onset_time_idx."""
    scorer_json = {"name": "IntegratedScorer", "variable": "z500", "params": {"n_days": 5}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=7)
    assert ctx.scorer.onset_time_idx == 7


def test_drift_penalized_scorer_alias_produces_ano_scorer(regions_json):
    scorer_json = {"name": "DriftPenalizedScorer", "variable": "z500", "params": {"n_days": 5}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=7)
    assert isinstance(ctx.scorer, ANOScorer)
    assert ctx.scorer_params.get("mode") == "auto"
    assert ctx.scorer_params.get("use_drift_penalty") == True


# --- Region selector ---

def test_heatwave_region_uses_explicit_points_from_regions_json(regions_json):
    """HeatwaveMeanScorer gets {lon, lat} from regions.json, not hardcoded class constants."""
    scorer_json = {"name": "HeatwaveMeanScorer", "variable": "tas", "params": {"n_days": 7}}
    ctx = build_scorer_context(scorer_json, "Chicago", regions_json, onset_time_idx=0)
    assert "lon" in ctx.region_bounds
    assert "lat" in ctx.region_bounds
    assert "lon_min" not in ctx.region_bounds
    # Values must come from regions_json, not HeatwaveMeanScorer.canonical_region_points
    assert ctx.region_bounds["lon"] == REGIONS_JSON_CONTENT["Chicago"]["lon"]


def test_blocking_region_uses_bounding_box(regions_json):
    scorer_json = {"name": "GridpointIntensityScorer", "variable": "z500",
                   "params": {"n_days": 7}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=0)
    assert "lon_min" in ctx.region_bounds
    assert "lon" not in ctx.region_bounds


# --- onset_time_idx ---

def test_onset_from_params_when_not_provided(regions_json):
    scorer_json = {"name": "GridpointIntensityScorer", "variable": "z500",
                   "params": {"n_days": 7, "onset_time_idx": 5}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json)
    assert ctx.onset_time_idx == 5


def test_explicit_onset_overrides_params(regions_json):
    scorer_json = {"name": "GridpointIntensityScorer", "variable": "z500",
                   "params": {"n_days": 7, "onset_time_idx": 5}}
    ctx = build_scorer_context(scorer_json, "NorthAtlantic", regions_json, onset_time_idx=3)
    assert ctx.onset_time_idx == 3


def test_lead_time_in_params_used_as_onset(regions_json):
    scorer_json = {"name": "HeatwaveMeanScorer", "variable": "tas",
                   "params": {"n_days": 7, "lead_time": 4}}
    ctx = build_scorer_context(scorer_json, "Chicago", regions_json)
    assert ctx.onset_time_idx == 4


# --- Validation ---

def test_unknown_scorer_raises(regions_json):
    with pytest.raises(ValueError, match="Unknown scorer"):
        build_scorer_context({"name": "BogusScorer", "variable": "z500", "params": {}},
                             "NorthAtlantic", regions_json, onset_time_idx=0)


def test_wrong_variable_raises(regions_json):
    with pytest.raises(ValueError):
        build_scorer_context({"name": "HeatwaveMeanScorer", "variable": "z500", "params": {}},
                             "Chicago", regions_json, onset_time_idx=0)


def test_wrong_region_raises(regions_json):
    with pytest.raises(ValueError):
        build_scorer_context({"name": "HeatwaveMeanScorer", "variable": "tas",
                              "params": {"n_days": 7}},
                             "NorthAtlantic", regions_json, onset_time_idx=0)


def test_scorer_context_is_frozen(regions_json):
    ctx = build_scorer_context({"name": "HeatwaveMeanScorer", "variable": "tas",
                                "params": {"n_days": 7}},
                               "Chicago", regions_json, onset_time_idx=0)
    with pytest.raises(Exception):
        ctx.variable = "z500"
