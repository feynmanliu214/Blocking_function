#!/usr/bin/env python3
"""
Return Period Curve for AI-RES outputs.

This script computes y-axis scores from PlaSim (physics-model) trajectories at
`step_{K}/particle_{i}/output/plasim_out.step_{K}.particle_{i}.nc`
using the experiment's scorer configuration.

When the confirmed scorer is GridpointIntensityScorer, DNS ground-truth curves
are overlaid by default (`--show-ground-truth`) using files in
`data/return_curve_ground_truth/`.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root and src/ are on sys.path before local imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from forecast_analysis.scoring.detection import (
    SCORER_CHOICES,
    detect_scorer_from_experiment,
    enforce_plasim_score_policy,
    format_spec,
    resolve_confirmed_scorer,
    resolve_scorer_profile,
)
from src.return_curve import (
    build_name_index,
    compute_aires_return_period_years,
    estimate_proba_res,
    format_region_label,
    get_particle_indices_for_step,
    get_product_of_weights,
    resolve_return_curve_y_limits,
    setup_publication_style,
    restore_style,
    CLR_AIRES,
    CLR_AIRES_EDGE,
    CLR_FULL,
    CLR_SUBSET,
)


# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

DEFAULT_EXP_PATH = (
    "/glade/derecho/scratch/zhil/PLASIM/RES/experiments/EXP28_AIRES_NorthAtlantic/"
    "EXP28_AIRES_NorthAtlantic_4"
)
DEFAULT_OUTPUT_DIR = "/glade/u/home/zhil/project/AI-RES/Blocking/figures/AI-RES-return_curve"
DEFAULT_K = 7
DEFAULT_SEASON_LABEL = "DJF"
DEFAULT_CAP_RP_LOWER_AT_TENTH = True
DEFAULT_SHOW_GROUND_TRUTH = True
DEFAULT_DNS_GROUND_TRUTH_WINDOW = "12-24_to_12-30"
DEFAULT_PLOT_STYLE_SCALE = 2.5
DEFAULT_REGIONS_JSON = (
    _PROJECT_ROOT / "AI-RES" / "RES" / "regions.json"
)
DEFAULT_GROUND_TRUTH_DIR = _PROJECT_ROOT / "data" / "return_curve_ground_truth"
GROUND_TRUTH_12_24_TO_12_30_FULL_RETRY2 = (
    "return_curve_block_maxima_always_12-24_to_12-30_retry2.npz"
)
GROUND_TRUTH_12_24_TO_12_30_SUBSET_RETRY2 = (
    "return_curve_block_maxima_always_12-24_to_12-30_retry2_subset400.npz"
)
GROUND_TRUTH_12_24_TO_12_30_FULL_LEGACY = (
    "return_curve_block_maxima_always_12-24_to_12-30.npz"
)
GROUND_TRUTH_12_24_TO_12_30_SUBSET_LEGACY = (
    "return_curve_block_maxima_always_12-24_to_12-30_subset400.npz"
)
GROUND_TRUTH_WINDOW_CHOICES = ["djf", "12-24_to_12-30"]
DEFAULT_PDF_OVERLAY = True
DEFAULT_PDF_FILE = (
    _PROJECT_ROOT
    / "outputs"
    / "plasim"
    / "sim52"
    / "z500_anomaly_pdf_northatlantic_djf_daily_weighted.nc"
)

FALLBACK_REGION_BOUNDS = {
    "Eurasia": (30.0, 100.0, 55.0, 75.0),
    "NorthAtlantic": (-59.0625, -2.8125, 57.20397269, 73.94716987),
}


def resolve_subset_ground_truth_file(
    *,
    pattern: str,
    fallback_name: str,
) -> Path:
    matches = sorted(DEFAULT_GROUND_TRUTH_DIR.glob(pattern))
    if not matches:
        return DEFAULT_GROUND_TRUTH_DIR / fallback_name

    def subset_order_key(path: Path) -> Tuple[int, str]:
        match = re.search(r"_subset(\d+)\.npz$", path.name)
        subset_size = int(match.group(1)) if match else -1
        return (subset_size, path.name)

    return max(matches, key=subset_order_key)


def prefer_existing_path(preferred: Path, fallback: Path) -> Path:
    return preferred if preferred.exists() else fallback


def resolve_dns_ground_truth_files(window: str) -> Tuple[Path, Path]:
    if window == "djf":
        return (
            DEFAULT_GROUND_TRUTH_DIR / "return_curve_block_maxima.npz",
            DEFAULT_GROUND_TRUTH_DIR / "return_curve_block_maxima_subset400.npz",
        )
    if window == "12-24_to_12-30":
        legacy_subset = resolve_subset_ground_truth_file(
            pattern="return_curve_block_maxima_always_12-24_to_12-30_subset*.npz",
            fallback_name=GROUND_TRUTH_12_24_TO_12_30_SUBSET_LEGACY,
        )
        return (
            prefer_existing_path(
                DEFAULT_GROUND_TRUTH_DIR / GROUND_TRUTH_12_24_TO_12_30_FULL_RETRY2,
                DEFAULT_GROUND_TRUTH_DIR / GROUND_TRUTH_12_24_TO_12_30_FULL_LEGACY,
            ),
            prefer_existing_path(
                DEFAULT_GROUND_TRUTH_DIR / GROUND_TRUTH_12_24_TO_12_30_SUBSET_RETRY2,
                legacy_subset,
            ),
        )
    raise ValueError(
        f"Unsupported DNS ground-truth window: {window!r}. "
        f"Expected one of {GROUND_TRUTH_WINDOW_CHOICES}."
    )


def format_dns_ground_truth_window_label(window: str) -> str:
    if window == "djf":
        return "DJF full-season maxima"
    if window == "12-24_to_12-30":
        return "12/24-12/30 window maxima"
    return window


def load_baseline_pdf(pdf_file: Path) -> Dict[str, np.ndarray]:
    """Load KDE curve and compute mu, sigma from the baseline PDF NetCDF.

    Returns dict with keys: kde_x, kde_density, mu, sigma.
    """
    import xarray as xr

    ds = xr.open_dataset(pdf_file)
    kde_x = ds["kde_x"].values.astype(float)
    kde_density = ds["kde_density"].values.astype(float)
    ds.close()

    # Compute weighted mean and std from the KDE curve (trapezoidal integration)
    dx = np.diff(kde_x)
    mid_x = 0.5 * (kde_x[:-1] + kde_x[1:])
    mid_d = 0.5 * (kde_density[:-1] + kde_density[1:])
    total = np.sum(mid_d * dx)
    mu = np.sum(mid_x * mid_d * dx) / total
    sigma = np.sqrt(np.sum((mid_x - mu) ** 2 * mid_d * dx) / total)

    return {"kde_x": kde_x, "kde_density": kde_density, "mu": mu, "sigma": sigma}


def ensure_project_import_paths() -> None:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))



def load_region_bounds(region: str, regions_json_path: Path = DEFAULT_REGIONS_JSON) -> Dict[str, float]:
    if regions_json_path.exists():
        with regions_json_path.open("r", encoding="utf-8") as f:
            regions = json.load(f)
        region_cfg = regions.get(region)
        if isinstance(region_cfg, dict):
            lon_values = region_cfg.get("lon")
            lat_values = region_cfg.get("lat")
            if isinstance(lon_values, list) and isinstance(lat_values, list):
                if lon_values and lat_values:
                    return {
                        "lon_min": float(min(lon_values)),
                        "lon_max": float(max(lon_values)),
                        "lat_min": float(min(lat_values)),
                        "lat_max": float(max(lat_values)),
                    }

    fallback = FALLBACK_REGION_BOUNDS.get(region)
    if fallback is not None:
        lon_min, lon_max, lat_min, lat_max = fallback
        return {
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
        }

    raise ValueError(
        f"Could not resolve bounds for region '{region}'. "
        f"Checked {regions_json_path} and fallback presets {list(FALLBACK_REGION_BOUNDS)}."
    )


def resolve_plasim_step_file(exp_path: Path, K: int, particle_idx: int) -> Path:
    return exp_path / f"step_{K}" / f"particle_{particle_idx}" / "output" / (
        f"plasim_out.step_{K}.particle_{particle_idx}.nc"
    )


def load_scores_from_plasim(
    exp_path: Path,
    K: int,
    particle_indices: List[int],
    scorer_name: str,
    scorer_params: Dict[str, Any],
    region: str,
    clim_file: Path,
    threshold_json_file: Path,
) -> np.ndarray:
    ensure_project_import_paths()

    import xarray as xr
    from forecast_analysis.data_loading import load_climatology_and_thresholds
    from forecast_analysis.scoring import scorer_required_variable
    from forecast_analysis.scoring.context import build_scorer_context
    from forecast_analysis.scoring.pipeline import (
        extract_variable,
        prepare_daily_field,
        score_single_member,
    )

    if scorer_name == "RMSEScorer":
        raise ValueError(
            "RMSEScorer is an analysis-only scorer and cannot be used for "
            "PlaSim trajectory rescoring. Provide pre-computed scores at "
            f"{exp_path}/resampling/plasim_scores_step_{K}.npy instead."
        )

    scorer_params_eff, policy_note = enforce_plasim_score_policy(
        scorer_name=scorer_name, scorer_params=scorer_params
    )
    if policy_note:
        print(policy_note)

    # Build scorer context once (validates scorer before loading any files)
    scorer_json = {
        "name": scorer_name,
        "variable": scorer_required_variable(scorer_name),
        "params": dict(scorer_params_eff),
    }
    ctx = build_scorer_context(
        scorer_json=scorer_json,
        region=region,
        regions_json=DEFAULT_REGIONS_JSON,
        onset_time_idx=None,  # let build_scorer_context derive from params
    )

    missing_files = []
    nc_files: List[Path] = []
    for idx in particle_indices:
        nc_path = resolve_plasim_step_file(exp_path, K, idx)
        if not nc_path.exists():
            missing_files.append(nc_path)
        else:
            nc_files.append(nc_path)

    if missing_files:
        preview = "\n".join(f"  - {p}" for p in missing_files[:10])
        more = ""
        if len(missing_files) > 10:
            more = f"\n  ... and {len(missing_files) - 10} more"
        raise FileNotFoundError(
            "PlaSim scoring requires one trajectory file per particle at final step K.\n"
            f"Missing {len(missing_files)} files. Examples:\n{preview}{more}\n"
            "Expected pattern: step_K/particle_i/output/plasim_out.step_K.particle_i.nc\n"
            "Return-curve plotting does not fall back to panguplasim_in files."
        )

    # Load clim/thresholds only if needed
    z500_clim, threshold_90 = None, None
    if ctx.scorer.requires_anomaly:
        if not clim_file.exists():
            raise FileNotFoundError(
                f"Missing climatology file required for PlaSim scoring: {clim_file}"
            )
        if not threshold_json_file.exists():
            raise FileNotFoundError(
                f"Missing threshold file required for PlaSim scoring: {threshold_json_file}"
            )
        print(f"Loading climatology: {clim_file}")
        print(f"Loading thresholds: {threshold_json_file}")
        z500_clim, threshold_90 = load_climatology_and_thresholds(
            str(clim_file),
            str(threshold_json_file),
        )

    scores: List[float] = []
    total = len(nc_files)
    print(f"Computing PlaSim trajectory scores for {total} particles at step K={K}...")
    for i, (_particle_idx, nc_path) in enumerate(zip(particle_indices, nc_files), start=1):
        with xr.open_dataset(str(nc_path)) as ds:
            field = extract_variable(ds, ctx.variable)
            field_daily = prepare_daily_field(field, filter_nh=(ctx.variable == "z500"))
            score = score_single_member(field_daily, ctx, z500_clim, threshold_90)
        scores.append(float(score))
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  scored {i}/{total} particles")

    return np.asarray(scores, dtype=float)


def _load_ground_truth_curve_npz(
    curve_path: Path,
    *,
    curve_kind: str,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[str]]:
    if not curve_path.exists():
        raise FileNotFoundError(f"Missing {curve_kind} ground-truth curve file: {curve_path}")

    with np.load(curve_path) as payload:
        required = ["al_values", "return_periods", "n_seasons"]
        missing = [k for k in required if k not in payload.files]
        if missing:
            raise KeyError(f"Missing required key(s) {missing} in {curve_path}")

        al_values = np.asarray(payload["al_values"], dtype=float).reshape(-1)
        return_periods = np.asarray(payload["return_periods"], dtype=float).reshape(-1)
        n_seasons = int(np.asarray(payload["n_seasons"]).reshape(-1)[0])
        method: Optional[str] = None
        if "method" in payload.files:
            try:
                method_arr = np.asarray(payload["method"]).reshape(-1)
                if method_arr.size > 0:
                    method = str(method_arr[0])
            except Exception:
                method = None

    if al_values.size == 0 or return_periods.size == 0:
        raise ValueError(f"{curve_kind} ground-truth curve file is empty: {curve_path}")
    if al_values.size != return_periods.size:
        raise ValueError(
            f"{curve_kind} ground-truth curve length mismatch in {curve_path}: "
            f"len(al_values)={al_values.size}, len(return_periods)={return_periods.size}"
        )
    valid_mask = (
        np.isfinite(al_values)
        & np.isfinite(return_periods)
        & (al_values > 0.0)
        & (return_periods > 0.0)
    )
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        raise ValueError(
            f"{curve_kind} ground-truth curve has no finite positive values: {curve_path}"
        )
    n_dropped = int(al_values.size - n_valid)
    if n_dropped > 0:
        print(
            f"Warning: dropped {n_dropped}/{al_values.size} {curve_kind} "
            "ground-truth points with non-finite or non-positive values "
            f"from {curve_path}"
        )
    al_values = al_values[valid_mask]
    return_periods = return_periods[valid_mask]

    return al_values, return_periods, n_seasons, method


def load_ground_truth_full_curve(
    full_curve_path: Path,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[str]]:
    return _load_ground_truth_curve_npz(
        full_curve_path,
        curve_kind="full-ensemble",
    )


def load_ground_truth_subset_curve(
    subset_curve_path: Path,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[str]]:
    return _load_ground_truth_curve_npz(
        subset_curve_path,
        curve_kind="subset",
    )


def _prepare_dns_curve_for_plot(
    al_values: np.ndarray,
    return_periods: np.ndarray,
    method: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Prepare DNS curve arrays for visualization.

    For block-maxima curves, hide the zero-mass branch (al_values == 0) and use a
    left-to-right ordering for step rendering on a log x-axis.
    """
    x_plot = np.asarray(return_periods, dtype=float).reshape(-1)
    y_plot = np.asarray(al_values, dtype=float).reshape(-1)
    is_block_maxima = (method or "").strip() == "block_maxima"

    if not is_block_maxima:
        return x_plot, y_plot, False

    positive_mask = y_plot > 0.0
    x_plot = x_plot[positive_mask]
    y_plot = y_plot[positive_mask]

    if x_plot.size >= 2 and x_plot[0] > x_plot[-1]:
        x_plot = x_plot[::-1]
        y_plot = y_plot[::-1]

    return x_plot, y_plot, True


def _plot_dns_curve(
    ax: plt.Axes,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    is_step: bool,
    label: str,
    color: str,
    alpha: float = 0.9,
    linestyle: Any = "-",
    linewidth: float = 3.6,
    marker: Optional[str] = None,
    markersize: float = 5.2,
    markevery: Optional[int] = None,
    zorder: int = 3,
    tail_threshold_rp: Optional[float] = None,
) -> None:
    """Plot a DNS ground-truth curve, choosing between step and line rendering.

    When *tail_threshold_rp* is set, points with return period above the
    threshold are rendered as discrete dots instead of a continuous line.
    """
    xp = np.asarray(x_plot, dtype=float).reshape(-1)
    yp = np.asarray(y_plot, dtype=float).reshape(-1)

    if tail_threshold_rp is not None:
        line_mask = xp < tail_threshold_rp
        tail_mask = ~line_mask
    else:
        line_mask = np.ones(xp.size, dtype=bool)
        tail_mask = np.zeros(xp.size, dtype=bool)

    has_tail = bool(np.any(tail_mask))

    if np.any(line_mask):
        effective_label = label
        if has_tail:
            effective_label = (
                f"{label}\n(dots: RP $\\geq$ {tail_threshold_rp:,.0f} yr)"
            )
        marker_kw: Dict[str, Any] = {}
        if marker is not None:
            n_line = int(np.count_nonzero(line_mask))
            marker_kw = dict(
                marker=marker,
                markersize=markersize,
                markevery=markevery if markevery else max(1, n_line // 80),
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.3,
            )

        if is_step:
            ax.step(
                xp[line_mask], yp[line_mask],
                where="post",
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=effective_label,
                linestyle=linestyle,
                zorder=zorder,
                **marker_kw,
            )
        else:
            ax.plot(
                xp[line_mask], yp[line_mask],
                linestyle=linestyle,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=effective_label,
                zorder=zorder,
                **marker_kw,
            )

    if has_tail:
        tail_label = label if not np.any(line_mask) else "_nolegend_"
        ax.scatter(
            xp[tail_mask], yp[tail_mask],
            s=22,
            c=color,
            marker="o",
            alpha=alpha,
            edgecolors="white",
            linewidths=0.3,
            zorder=zorder,
            label=tail_label,
        )


def plot_return_curve(
    exp_path: Path,
    output_dir: Path,
    K: int,
    season_label: str,
    scorer_name: str,
    scorer_params: Dict[str, Any],
    scorer_profile: Dict[str, Any],
    region: str,
    divide_by_parent_weights: bool,
    clim_file: Path,
    threshold_json_file: Path,
    show_ground_truth: bool = DEFAULT_SHOW_GROUND_TRUTH,
    ground_truth_window: str = DEFAULT_DNS_GROUND_TRUTH_WINDOW,
    ground_truth_full_file: Optional[Path] = None,
    ground_truth_subset_file: Optional[Path] = None,
    cap_rp_lower_at_tenth: bool = DEFAULT_CAP_RP_LOWER_AT_TENTH,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    show_pdf_overlay: bool = DEFAULT_PDF_OVERLAY,
    pdf_file: Optional[Path] = None,
    save_fig: bool = True,
    show_fig: bool = False,
) -> Dict[str, Any]:
    exp_name = exp_path.name

    if ground_truth_full_file is None or ground_truth_subset_file is None:
        resolved_full_file, resolved_subset_file = resolve_dns_ground_truth_files(
            ground_truth_window
        )
        if ground_truth_full_file is None:
            ground_truth_full_file = resolved_full_file
        if ground_truth_subset_file is None:
            ground_truth_subset_file = resolved_subset_file

    tree_path = exp_path / "working_tree.pkl"
    with tree_path.open("rb") as f:
        working_tree = pickle.load(f)
    name_index = build_name_index(working_tree)
    print("Loaded working tree")

    particle_indices = get_particle_indices_for_step(K, name_index)
    n_particles = len(particle_indices)
    print(
        f"Found {n_particles} particles at step {K}: "
        f"indices [{particle_indices[0]} ... {particle_indices[-1]}]"
    )

    # Try saved PlaSim scores first; recompute from .nc only if missing
    saved_path = exp_path / "resampling" / f"plasim_scores_step_{K}.npy"
    if saved_path.exists():
        print(f"Loading pre-computed PlaSim scores from: {saved_path}")
        all_scores = np.load(saved_path)
        scores = np.asarray(
            [float(all_scores[idx]) for idx in particle_indices], dtype=float,
        )
    else:
        print(
            f"No saved PlaSim scores at {saved_path}; "
            "recomputing from plasim_out .nc files..."
        )
        scores = load_scores_from_plasim(
            exp_path=exp_path,
            K=K,
            particle_indices=particle_indices,
            scorer_name=scorer_name,
            scorer_params=scorer_params,
            region=region,
            clim_file=clim_file,
            threshold_json_file=threshold_json_file,
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Score array contains non-finite values.")
    print(
        f"Loaded PlaSim scores: N={n_particles}, "
        f"range=[{scores.min():.4f}, {scores.max():.4f}]"
    )

    print(f"Computing weights for {n_particles} particles...")
    prod_weights = np.array(
        [
            get_product_of_weights(
                i,
                K,
                name_index,
                divide_by_parent_weights=divide_by_parent_weights,
            )
            for i in particle_indices
        ],
        dtype=float,
    )

    # Free working tree memory now that weights are computed
    del name_index, working_tree

    if divide_by_parent_weights:
        print("Weight mode: single-parent correction (divide_by_parent_weights=true)")
    else:
        print("Weight mode: ancestry-product correction (divide_by_parent_weights=false)")
    print(
        f"Product of weights: min={prod_weights.min():.4f}, "
        f"max={prod_weights.max():.4f}, mean={prod_weights.mean():.4f}"
    )

    rv_exp = np.sort(scores)
    print("Computing RES probability estimates...")
    proba_exp = np.array(
        [estimate_proba_res(scores, rv, prod_weights) for rv in rv_exp],
        dtype=float,
    )

    print(
        "Return-period conversion: fixed annual window (12/24-12/30), "
        "RP_years = 1 / p_window"
    )
    rp_exp = compute_aires_return_period_years(proba_exp)

    print(f"Return period range: {rp_exp.min():.4f} to {rp_exp.max():.4f} years")
    print(f"Score range: {rv_exp.min():.4f} to {rv_exp.max():.4f}")

    # ── Publication-quality style setup ──────────────────────────────
    _rc_prev = setup_publication_style()

    # Pre-check whether we'll draw the PDF panel (determines layout)
    _want_pdf_panel = (
        show_pdf_overlay
        and scorer_profile["raw_name"] == "GridpointIntensityScorer"
    )
    if _want_pdf_panel:
        fig = plt.figure(figsize=(10.5, 6.5))
        gs = fig.add_gridspec(
            1, 2, width_ratios=[4, 1], wspace=0.0,
        )
        ax = fig.add_subplot(gs[0])
        ax_pdf = fig.add_subplot(gs[1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax_pdf = None

    # --- AI+RES scatter ---
    ax.scatter(
        rp_exp,
        rv_exp,
        s=29,
        c=CLR_AIRES,
        marker="^",
        label=rf"AI+RES $(\mathrm{{N}} = {n_particles})$",
        alpha=0.80,
        edgecolors=CLR_AIRES_EDGE,
        linewidths=0.3,
        zorder=5,
    )

    # --- Ground-truth overlay ---
    ground_truth_curve: Optional[Dict[str, Any]] = None
    dns_block_maxima_visualization_used = False
    if show_ground_truth:
        if scorer_profile["raw_name"] != "GridpointIntensityScorer":
            print(
                "Ground-truth overlay skipped: scorer is "
                f"{scorer_profile['raw_name']} (only GridpointIntensityScorer is supported)."
            )
        else:
            try:
                (
                    al_values,
                    return_periods,
                    n_seasons,
                    full_method,
                ) = load_ground_truth_full_curve(ground_truth_full_file)
                full_x_plot, full_y_plot, full_is_step = _prepare_dns_curve_for_plot(
                    al_values,
                    return_periods,
                    full_method,
                )
                if full_is_step:
                    dns_block_maxima_visualization_used = True
                if full_x_plot.size == 0:
                    print(
                        "Warning: full-ensemble DNS ground-truth curve has no positive "
                        "values to display after hiding the zero branch for "
                        "block_maxima visualization."
                    )
                else:
                    _plot_dns_curve(
                        ax, full_x_plot, full_y_plot,
                        is_step=full_is_step,
                        label=f"Full ensemble ({n_seasons:,} seasons)",
                        color=CLR_FULL,
                        zorder=3,
                        tail_threshold_rp=5e3,
                    )

                subset_al = subset_rp = None
                subset_n = None
                subset_method: Optional[str] = None
                try:
                    (
                        subset_al,
                        subset_rp,
                        subset_n,
                        subset_method,
                    ) = load_ground_truth_subset_curve(ground_truth_subset_file)
                except Exception as exc:
                    print(
                        "Warning: could not load subset ground-truth return curve "
                        f"from {ground_truth_subset_file}: {exc}"
                    )

                if (
                    subset_al is not None
                    and subset_rp is not None
                    and subset_n is not None
                ):
                    (
                        subset_x_plot,
                        subset_y_plot,
                        subset_is_step,
                    ) = _prepare_dns_curve_for_plot(
                        subset_al,
                        subset_rp,
                        subset_method,
                    )
                    if subset_is_step:
                        dns_block_maxima_visualization_used = True
                    if subset_x_plot.size == 0:
                        print(
                            "Warning: subset DNS ground-truth curve has no positive values "
                            "to display after hiding the zero branch for block_maxima "
                            "visualization."
                        )
                    else:
                        _plot_dns_curve(
                            ax, subset_x_plot, subset_y_plot,
                            is_step=subset_is_step,
                            label=f"First {subset_n:,} seasons",
                            color=CLR_SUBSET,
                            alpha=0.85,
                            linestyle=(0, (1.5, 2)),
                            marker="s",
                            zorder=4,
                        )

                ground_truth_curve = {
                    "full_al_values": al_values,
                    "full_return_periods": return_periods,
                    "full_n_seasons": n_seasons,
                    "full_method": full_method,
                    "subset_al": subset_al,
                    "subset_rp": subset_rp,
                    "subset_n": subset_n,
                    "subset_method": subset_method,
                }
            except Exception as exc:
                print(
                    "Warning: could not load full-ensemble ground-truth return curve "
                    f"from {ground_truth_full_file}: {exc}"
                )

    # --- Sideways baseline PDF panel (GridpointIntensityScorer only) ---
    CLR_PDF = "#8B5CF6"
    if _want_pdf_panel:
        _pdf_path = pdf_file if pdf_file is not None else DEFAULT_PDF_FILE
        try:
            pdf_data = load_baseline_pdf(Path(_pdf_path))
            kde_x = pdf_data["kde_x"]
            kde_density = pdf_data["kde_density"]
            mu = pdf_data["mu"]
            sigma = pdf_data["sigma"]

            ax_pdf.fill_betweenx(
                kde_x, 0, kde_density,
                alpha=0.12, color=CLR_PDF, zorder=1,
            )
            ax_pdf.plot(
                kde_density, kde_x,
                color=CLR_PDF, linewidth=1.5, alpha=0.7,
                label="Baseline PDF", zorder=2,
            )
            ax_pdf.set_xlim(0, None)

            ax_pdf.set_xlabel("")
            ax_pdf.tick_params(
                axis="x", labelbottom=False, length=0,
            )
            ax_pdf.tick_params(axis="y", labelleft=False)
            for spine in ("top", "right"):
                ax_pdf.spines[spine].set_visible(False)
            ax_pdf.spines["left"].set_visible(False)
            ax_pdf.grid(True, which="major", axis="y", linewidth=0.6,
                        alpha=0.30, color="#888888")

            ax_pdf.annotate(
                "North Atlantic DJF\nZ500 anomaly PDF",
                xy=(1.0, 0.5),
                xycoords="axes fraction",
                fontsize=9, color="#666666",
                ha="left", va="center",
                rotation=-90,
                xytext=(8, 0), textcoords="offset points",
            )

            # Sigma lines
            max_score = float(rv_exp.max())
            max_sigma = int(math.ceil((max_score - mu) / sigma))
            max_sigma = max(max_sigma, 1)
            for n in range(1, max_sigma + 1):
                y_sigma = mu + n * sigma
                ax.axhline(
                    y=y_sigma,
                    color="#666666", linewidth=0.8, linestyle="--", alpha=0.5,
                    zorder=1,
                )
                ax_pdf.axhline(
                    y=y_sigma,
                    color="#666666", linewidth=0.8, linestyle="--", alpha=0.5,
                    zorder=1,
                )
                ax_pdf.annotate(
                    f"{n}σ",
                    xy=(0.92, y_sigma),
                    xycoords=("axes fraction", "data"),
                    fontsize=9, color="#666666", alpha=0.7,
                    ha="right", va="bottom",
                    xytext=(0, 2), textcoords="offset points",
                )

            print(
                f"PDF overlay: μ={mu:.1f}, σ={sigma:.1f}, "
                f"σ-lines up to {max_sigma}σ (y={mu + max_sigma * sigma:.1f})"
            )
        except Exception as exc:
            print(f"Warning: could not load baseline PDF overlay: {exc}")
            ax_pdf.set_visible(False)

    # --- Axes labels & title ---
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)", fontsize=13, labelpad=8)
    ax.set_ylabel(scorer_profile["y_label"], fontsize=13, labelpad=8)
    region_label = format_region_label(region)
    metric_label = scorer_profile.get("metric_label", scorer_profile["score_title"])
    ax.set_title(
        (
            f"Return-period curve of blocking in {region_label} "
            "(12/24–12/30)\n"
            f"{metric_label}"
        ),
        fontsize=14,
        fontweight="semibold",
        pad=12,
    )

    # --- Grid ---
    ax.grid(True, which="major", linewidth=0.6, alpha=0.30, color="#888888")
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.15, color="#aaaaaa")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # --- X / Y limits ---
    x_lower = 0.1 if cap_rp_lower_at_tenth else 0.01
    rp_max_values = [float(rp_exp.max())]
    if ground_truth_curve is not None:
        rp_max_values.append(float(np.max(ground_truth_curve["full_return_periods"])))
        subset_rp = ground_truth_curve["subset_rp"]
        if subset_rp is not None:
            rp_max_values.append(float(np.max(subset_rp)))
    x_upper = max(max(rp_max_values) * 2.0, 100.0)
    if x_upper <= x_lower:
        x_upper = x_lower * 10.0
    ax.set_xlim(x_lower, x_upper)

    y_series = [rv_exp]
    if ground_truth_curve is not None:
        y_series.append(ground_truth_curve["full_al_values"])
        subset_al = ground_truth_curve["subset_al"]
        if subset_al is not None:
            y_series.append(subset_al)
    baseline_y_lim = (
        (0.0, 100.0) if scorer_profile["score_key"] == "blocking_area_pct" else None
    )
    resolved_y_lim = resolve_return_curve_y_limits(
        score_arrays=y_series,
        y_lim=baseline_y_lim,
        y_min=y_min,
        y_max=y_max,
    )
    if resolved_y_lim is not None:
        lower, upper = resolved_y_lim
        if dns_block_maxima_visualization_used and y_min is None:
            lower = max(0.0, lower)
        ax.set_ylim(lower, upper)

    # --- Legend ---
    legend = ax.legend(
        loc="lower right",
        fontsize=10.5,
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        edgecolor="#cccccc",
        borderpad=0.8,
        handlelength=2.0,
        handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.tight_layout()

    output_path = output_dir / (
        f"return_curve_{exp_name}_{scorer_profile['score_key']}_plasim.png"
    )
    if save_fig:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"\nFigure saved to: {output_path}")

    restore_style(_rc_prev)

    if show_fig:
        plt.show()
    else:
        plt.close()

    return {
        "rv_exp": rv_exp,
        "rp_exp": rp_exp,
        "proba_exp": proba_exp,
        "prod_weights": prod_weights,
        "scores": scores,
        "particle_indices": particle_indices,
        "output_path": str(output_path),
        "scorer_profile": scorer_profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot return period curve for AI-RES outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (PlaSim trajectory-based y-axis) with interactive scorer confirmation
    python plot_return_curve_blocking_area_pct.py --exp_path /path/to/experiment

    # Non-interactive scorer confirmation
    python plot_return_curve_blocking_area_pct.py \\
        --exp_path /path/to/experiment \\
        --confirm-scorer detected
        """,
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default=DEFAULT_EXP_PATH,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        help="Final resampling step number",
    )
    parser.add_argument(
        "--season_label",
        type=str,
        default=DEFAULT_SEASON_LABEL,
        help="Season label shown on the x-axis/title (default: DJF)",
    )
    parser.add_argument(
        "--confirm-scorer",
        type=str,
        default=None,
        choices=["detected", *SCORER_CHOICES],
        help=(
            "Skip interactive prompt and confirm scorer directly. "
            "Use 'detected' to accept the scorer detected from experiment outputs."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure (in addition to saving)",
    )
    parser.add_argument(
        "--show-ground-truth",
        dest="show_ground_truth",
        action="store_true",
        default=DEFAULT_SHOW_GROUND_TRUTH,
        help=(
            "Overlay ground-truth return curves from DNS files "
            "(enabled by default; only used for GridpointIntensityScorer). "
            "Use --dns-ground-truth-window to choose the DNS window."
        ),
    )
    parser.add_argument(
        "--no-show-ground-truth",
        dest="show_ground_truth",
        action="store_false",
        help="Disable ground-truth return-curve overlay.",
    )
    parser.add_argument(
        "--dns-ground-truth-window",
        type=str,
        default=DEFAULT_DNS_GROUND_TRUTH_WINDOW,
        choices=GROUND_TRUTH_WINDOW_CHOICES,
        help=(
            "DNS ground-truth window selector for the overlay only. "
            "'12-24_to_12-30' (default) loads block-maxima curves from "
            "return_curve_block_maxima_always_12-24_to_12-30_retry2*.npz "
            "(falls back to legacy return_curve_block_maxima_always_12-24_to_12-30*.npz "
            "if retry2 files are unavailable), computed "
            "from the Dec 24-Dec 30 window within each DJF season; "
            "'djf' loads full-season DJF block-maxima curves. "
            "Does not change AI+RES x-axis conversion "
            "(fixed-window annual return period, RP = 1/p_window)."
        ),
    )
    parser.add_argument(
        "--cap-rp-lower-at-tenth",
        dest="cap_rp_lower_at_tenth",
        action="store_true",
        default=DEFAULT_CAP_RP_LOWER_AT_TENTH,
        help=(
            "Cap the lower x-axis bound for return period at 10^-1 years "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--no-cap-rp-lower-at-tenth",
        dest="cap_rp_lower_at_tenth",
        action="store_false",
        help="Disable lower x-axis cap and use the legacy lower bound (10^-2 years).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower y-axis bound. Unspecified upper bound still auto-scales.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper y-axis bound. Unspecified lower bound still auto-scales.",
    )
    parser.add_argument(
        "--pdf-overlay",
        dest="pdf_overlay",
        action="store_true",
        default=DEFAULT_PDF_OVERLAY,
        help=(
            "Overlay the baseline Z500 anomaly PDF sideways on the return curve "
            "(enabled by default; only used for GridpointIntensityScorer)."
        ),
    )
    parser.add_argument(
        "--no-pdf-overlay",
        dest="pdf_overlay",
        action="store_false",
        help="Disable the baseline PDF overlay.",
    )
    parser.add_argument(
        "--pdf-file",
        type=str,
        default=str(DEFAULT_PDF_FILE),
        help=(
            "Path to the baseline Z500 anomaly PDF NetCDF file "
            "(default: outputs/plasim/sim52/z500_anomaly_pdf_northatlantic_djf_daily_weighted.nc)."
        ),
    )
    args = parser.parse_args()

    exp_path = Path(args.exp_path)
    output_dir = Path(args.output_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")
    if not (exp_path / "working_tree.pkl").exists():
        raise FileNotFoundError(f"Missing working tree: {exp_path / 'working_tree.pkl'}")

    detection = detect_scorer_from_experiment(exp_path)
    print("\n=== Scorer Detection (precedence: used_config -> base_config -> forecast_log) ===")
    print(f"used_config path: {detection['used_config_path']}")
    print(f"base_config path: {detection['base_config_path']}")
    print(f"from used_config: { format_spec(detection['scorer_from_used_config']) }")
    print(f"from base_config: { format_spec(detection['scorer_from_base_config']) }")
    print(f"from forecast_log: { format_spec(detection['scorer_from_forecast_log']) }")
    print(f"selected ({detection['selected_source']}): { format_spec(detection['selected_scorer']) }")

    confirmed_name, confirmed_params, confirm_origin = resolve_confirmed_scorer(
        detection["selected_scorer"],
        args.confirm_scorer,
    )
    scorer_profile = resolve_scorer_profile(confirmed_name, confirmed_params)

    print("\n=== Confirmed Scorer ===")
    print(f"choice origin: {confirm_origin}")
    print(f"scorer: {scorer_profile['raw_name']}")
    if scorer_profile["alias_note"]:
        print(f"alias mapping: {scorer_profile['alias_note']}")
    print(f"implementation: {scorer_profile['implementation']}")
    print(f"y-axis label: {scorer_profile['y_label']}")

    clim_file = Path(str(detection["clim_file"]))
    threshold_json_file = Path(str(detection["threshold_json_file"]))
    print("\n=== Score Source ===")
    print("score source: plasim")
    print(f"climatology file: {clim_file}")
    print(f"threshold json file: {threshold_json_file}")
    print(f"divide_by_parent_weights: {detection['divide_by_parent_weights']}")
    print(
        "x-axis lower bound mode: "
        + ("cap at 10^-1 years" if args.cap_rp_lower_at_tenth else "legacy 10^-2 years")
    )
    print(f"show ground-truth overlay: {args.show_ground_truth}")
    print(f"show PDF overlay: {args.pdf_overlay}")
    if args.pdf_overlay:
        print(f"PDF file: {args.pdf_file}")

    ground_truth_full_file, ground_truth_subset_file = resolve_dns_ground_truth_files(
        args.dns_ground_truth_window
    )
    if args.show_ground_truth:
        print(f"DNS ground-truth window: {args.dns_ground_truth_window}")
        print(
            "DNS ground-truth window label: "
            f"{format_dns_ground_truth_window_label(args.dns_ground_truth_window)}"
        )
        print(f"ground-truth full file: {ground_truth_full_file}")
        print(f"ground-truth subset file: {ground_truth_subset_file}")

    plot_return_curve(
        exp_path=exp_path,
        output_dir=output_dir,
        K=args.K,
        season_label=args.season_label,
        scorer_name=confirmed_name,
        scorer_params=confirmed_params,
        scorer_profile=scorer_profile,
        region=detection["region"],
        divide_by_parent_weights=bool(detection["divide_by_parent_weights"]),
        clim_file=clim_file,
        threshold_json_file=threshold_json_file,
        show_ground_truth=args.show_ground_truth,
        ground_truth_window=args.dns_ground_truth_window,
        ground_truth_full_file=ground_truth_full_file,
        ground_truth_subset_file=ground_truth_subset_file,
        cap_rp_lower_at_tenth=args.cap_rp_lower_at_tenth,
        y_min=args.y_min,
        y_max=args.y_max,
        show_pdf_overlay=args.pdf_overlay,
        pdf_file=Path(args.pdf_file),
        save_fig=True,
        show_fig=args.show,
    )


if __name__ == "__main__":
    main()
