"""Scorer detection and confirmation from experiment outputs.

Detects the scorer configuration from an AI-RES experiment by checking
(in order of precedence): used_config JSON, base_config JSON, forecast log.
Provides interactive and CLI-based scorer confirmation, plus profile
resolution for plotting labels and metadata.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_CLIM_FILE = "/glade/u/home/zhil/project/AI-RES/Blocking/data/ano_climatology_thresholds.nc"
DEFAULT_THRESHOLD_JSON_FILE = "/glade/u/home/zhil/project/AI-RES/Blocking/data/ano_thresholds.json"

SCORER_LOG_RE = re.compile(
    r"Using scorer:\s*(?P<name>[A-Za-z0-9_]+)"
    r"(?:\s+with params:\s*(?P<params>\{[^{}]*\}))?"
)

SCORER_CHOICES = [
    "ANOScorer",
    "IntegratedScorer",
    "DriftPenalizedScorer",
    "GridpointPersistenceScorer",
    "GridpointIntensityScorer",
    "RMSEScorer",
    "HeatwaveMeanScorer",
]


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _extract_scorer_spec(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return None
    params = payload.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return {"name": name, "params": params}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def enforce_plasim_score_policy(
    scorer_name: str, scorer_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Normalize PlaSim scorer params for return-curve plotting.

    For GridpointIntensityScorer, respect the configured
    fallback_to_nonblocked value from experiment config (auto behavior).
    """
    params = dict(scorer_params or {})
    if scorer_name != "GridpointIntensityScorer":
        return params, None

    configured = params.get("fallback_to_nonblocked", None)
    if configured is None:
        params["fallback_to_nonblocked"] = False
        note = (
            "PlaSim score policy: fallback_to_nonblocked is not set; "
            "defaulting to False for GridpointIntensityScorer."
        )
    elif isinstance(configured, str):
        value = configured.strip().lower()
        if value == "always":
            params["fallback_to_nonblocked"] = "always"
            note = (
                "PlaSim score policy: using configured "
                "fallback_to_nonblocked='always' for GridpointIntensityScorer."
            )
        else:
            parsed_bool = _parse_bool(value)
            params["fallback_to_nonblocked"] = parsed_bool
            note = (
                "PlaSim score policy: using configured "
                f"fallback_to_nonblocked={parsed_bool} for GridpointIntensityScorer."
            )
    else:
        parsed_bool = _parse_bool(configured)
        params["fallback_to_nonblocked"] = parsed_bool
        note = (
            "PlaSim score policy: using configured "
            f"fallback_to_nonblocked={parsed_bool} for GridpointIntensityScorer."
        )

    return params, note


def _parse_scorer_from_log(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None

    match = SCORER_LOG_RE.search(text)
    if match is None:
        return None

    name = match.group("name")
    params_raw = match.group("params")
    params: Dict[str, Any] = {}
    if params_raw:
        try:
            parsed = ast.literal_eval(params_raw)
            if isinstance(parsed, dict):
                params = parsed
        except Exception:
            params = {}

    return {"name": name, "params": params}


def detect_scorer_from_experiment(exp_path: Path) -> Dict[str, Any]:
    exp_name = exp_path.name
    used_cfg_path = exp_path / f"{exp_name}_used_config.json"
    if not used_cfg_path.exists():
        matches = sorted(exp_path.glob("*_used_config.json"))
        used_cfg_path = matches[0] if matches else Path("")

    used_cfg = _safe_load_json(used_cfg_path) if used_cfg_path else {}
    scorer_from_used = _extract_scorer_spec(used_cfg.get("scorer"))

    base_cfg: Dict[str, Any] = {}
    base_cfg_path_raw = used_cfg.get("path_config_file")
    if isinstance(base_cfg_path_raw, str) and base_cfg_path_raw:
        base_cfg = _safe_load_json(Path(base_cfg_path_raw))
    scorer_from_base = _extract_scorer_spec(base_cfg.get("scorer"))

    scorer_from_log = _parse_scorer_from_log(exp_path / "outerr_forecasts.log")

    if scorer_from_used is not None:
        selected = scorer_from_used
        selected_source = "used_config"
    elif scorer_from_base is not None:
        selected = scorer_from_base
        selected_source = "base_config"
    elif scorer_from_log is not None:
        selected = scorer_from_log
        selected_source = "forecast_log"
    else:
        selected = None
        selected_source = "unknown"

    region = used_cfg.get("region") or base_cfg.get("region") or "unknown_region"
    clim_file = used_cfg.get("clim_file") or base_cfg.get("clim_file") or DEFAULT_CLIM_FILE
    threshold_json_file = (
        used_cfg.get("threshold_json_file")
        or base_cfg.get("threshold_json_file")
        or DEFAULT_THRESHOLD_JSON_FILE
    )
    divide_by_parent_weights = _parse_bool(
        used_cfg.get(
            "divide_by_parent_weights",
            base_cfg.get("divide_by_parent_weights", True),
        )
    )

    return {
        "used_config_path": str(used_cfg_path) if used_cfg_path else None,
        "base_config_path": base_cfg_path_raw if isinstance(base_cfg_path_raw, str) else None,
        "scorer_from_used_config": scorer_from_used,
        "scorer_from_base_config": scorer_from_base,
        "scorer_from_forecast_log": scorer_from_log,
        "selected_scorer": selected,
        "selected_source": selected_source,
        "region": region,
        "clim_file": clim_file,
        "threshold_json_file": threshold_json_file,
        "divide_by_parent_weights": divide_by_parent_weights,
    }


def resolve_scorer_profile(scorer_name: str, scorer_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(scorer_params or {})
    canonical_name = scorer_name
    alias_note = None

    if scorer_name == "IntegratedScorer":
        canonical_name = "ANOScorer"
        params.setdefault("mode", "onset")
        params.setdefault("use_drift_penalty", False)
        alias_note = "Legacy alias for ANOScorer(mode='onset', use_drift_penalty=False)"
    elif scorer_name == "DriftPenalizedScorer":
        canonical_name = "ANOScorer"
        params.setdefault("mode", "auto")
        params.setdefault("use_drift_penalty", True)
        alias_note = "Legacy alias for ANOScorer(mode='auto', use_drift_penalty=True)"

    if scorer_name == "GridpointPersistenceScorer":
        y_label = r"Mean blocked area $(\%)$"
        score_key = "blocking_area_pct"
        score_title = "Area-weighted mean blocked area"
        implementation = (
            "forecast_analysis/scoring/gridpoint/persistence_scorer.py"
            "::GridpointPersistenceScorer"
        )
    elif scorer_name == "GridpointIntensityScorer":
        y_label = r"Max blocking intensity $(\mathrm{m})$"
        score_key = "max_blocking_intensity"
        score_title = "Maximum running-mean Z500 anomaly at blocked grid points"
        implementation = (
            "forecast_analysis/scoring/gridpoint/intensity_scorer.py"
            "::GridpointIntensityScorer"
        )
    elif scorer_name == "RMSEScorer":
        y_label = r"Z500 anomaly RMSE $(\mathrm{m})$"
        score_key = "rmse"
        score_title = "Z500 anomaly RMSE"
        implementation = "forecast_analysis/scoring/ano/rmse_scorer.py::RMSEScorer"
    elif scorer_name == "HeatwaveMeanScorer":
        y_label = r"Mean temperature $(\mathrm{K})$"
        score_key = "mean_tas"
        score_title = "Spatiotemporal mean near-surface temperature"
        implementation = (
            "forecast_analysis/scoring/heatwave/mean_scorer.py"
            "::HeatwaveMeanScorer"
        )
    elif canonical_name == "ANOScorer":
        mode = str(params.get("mode", "auto"))
        use_drift_penalty = bool(params.get("use_drift_penalty", False))
        implementation = "forecast_analysis/scoring/ano/scorer.py::ANOScorer"

        if mode == "onset":
            if use_drift_penalty:
                y_label = r"Drift-penalized integrated anomaly score $(\mathrm{m^3\,day})$"
                score_key = "b_total"
                score_title = "Drift-penalized integrated anomaly score"
            else:
                y_label = r"Integrated positive Z500 anomaly $(\mathrm{m})$"
                score_key = "b_int"
                score_title = "Integrated positive Z500 anomaly"
        else:
            if use_drift_penalty:
                y_label = r"Drift-penalized blocking score $(\mathrm{m^3\,day})$"
                score_key = "p_total"
                score_title = "Drift-penalized blocking score"
            else:
                y_label = r"Integrated positive Z500 anomaly $(\mathrm{m})$"
                score_key = "p_block"
                score_title = "Integrated positive Z500 anomaly"
    else:
        y_label = "Score"
        score_key = "score"
        score_title = "Score"
        implementation = "forecast_analysis/scoring/<unknown>"

    return {
        "raw_name": scorer_name,
        "canonical_name": canonical_name,
        "params": params,
        "alias_note": alias_note,
        "implementation": implementation,
        "y_label": y_label,
        "score_key": score_key,
        "score_title": score_title,
        "metric_label": score_title,
    }


def format_spec(spec: Optional[Dict[str, Any]]) -> str:
    if spec is None:
        return "None"
    return f"{spec['name']} params={spec.get('params', {})}"


def prompt_scorer_confirmation(
    detected_spec: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], str]:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Interactive scorer confirmation requires a TTY. "
            "Use --confirm-scorer to confirm non-interactively."
        )

    options = []
    if detected_spec is not None:
        options.append(
            (
                f"Use detected scorer ({detected_spec['name']})",
                detected_spec["name"],
                detected_spec.get("params", {}),
                "detected",
            )
        )
    for scorer_name in SCORER_CHOICES:
        if detected_spec is not None and scorer_name == detected_spec["name"]:
            continue
        options.append((f"Override to {scorer_name}", scorer_name, {}, "override"))
    options.append(("Abort", "", {}, "abort"))

    print("\nScorer confirmation required. Choose one option:")
    for idx, (label, _, _, _) in enumerate(options, start=1):
        print(f"  {idx}) {label}")

    while True:
        sys.stdout.write(f"Select [1-{len(options)}]: ")
        sys.stdout.flush()
        choice = sys.stdin.readline().strip()
        if not choice.isdigit():
            print("Invalid choice. Enter the option number.")
            continue
        idx = int(choice)
        if idx < 1 or idx > len(options):
            print("Invalid choice. Enter a valid option number.")
            continue

        _, scorer_name, scorer_params, origin = options[idx - 1]
        if origin == "abort":
            raise RuntimeError("Aborted by user before plotting.")
        return scorer_name, scorer_params, origin


def resolve_confirmed_scorer(
    detected_spec: Optional[Dict[str, Any]],
    confirm_scorer_arg: Optional[str],
) -> Tuple[str, Dict[str, Any], str]:
    if confirm_scorer_arg:
        if confirm_scorer_arg == "detected":
            if detected_spec is None:
                raise ValueError(
                    "--confirm-scorer detected was requested, "
                    "but no scorer could be detected from experiment output."
                )
            return detected_spec["name"], detected_spec.get("params", {}), "cli-detected"
        return confirm_scorer_arg, {}, "cli-override"
    return prompt_scorer_confirmation(detected_spec)
