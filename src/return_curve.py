#!/usr/bin/env python3
"""
Shared return-curve utilities.

Provides reusable helpers for return-period computation, working-tree weight
calculation, DNS data loading, and publication-quality plot rendering.
Used by both the blocking and heatwave return-curve wrapper scripts.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Working tree / weight helpers
# ═══════════════════════════════════════════════════════════════════════════


def build_name_index(tree_root: Any) -> Dict[str, Any]:
    """Build a name -> node lookup from a working tree root."""
    index = {tree_root.name: tree_root}
    for node in tree_root.descendants:
        index[node.name] = node
    return index


def get_particle_indices_for_step(K: int, name_index: Dict[str, Any]) -> List[int]:
    """Return sorted particle indices present in the working tree at step K."""
    prefix = f"step_{K}_particle_"
    particle_indices = []
    for name in name_index:
        if not name.startswith(prefix):
            continue
        try:
            particle_indices.append(int(name.split("_")[-1]))
        except Exception:
            continue
    particle_indices = sorted(set(particle_indices))
    if not particle_indices:
        raise ValueError(f"No particles found in working tree for step {K}")
    return particle_indices


def get_product_of_weights(
    children_index: int,
    K: int,
    name_index: Dict[str, Any],
    divide_by_parent_weights: bool = True,
) -> float:
    """Compute the importance-sampling weight correction for a particle."""
    key = f"step_{K}_particle_{children_index}"
    node = name_index.get(key)
    if node is None:
        print(f"Warning: No node found for {key}")
        return 1.0

    parent = node.parent
    if parent is None:
        print(f"Warning: Node {key} has no parent; using neutral weight 1.0")
        return 1.0

    if divide_by_parent_weights:
        if hasattr(parent, "V") and parent.V is not None:
            if hasattr(parent, "norm_constant") and parent.norm_constant is not None:
                return float(np.exp(parent.V) / parent.norm_constant)
        print(
            f"Warning: Missing parent V/norm_constant for {key}; "
            "using neutral weight 1.0"
        )
        return 1.0

    # Legacy mode: product over all ancestors.
    prod = 1.0
    current = parent
    while current is not None and current.parent is not None:
        if hasattr(current, "V") and current.V is not None:
            if hasattr(current, "norm_constant") and current.norm_constant is not None:
                prod *= np.exp(current.V) / current.norm_constant
        current = current.parent
    return prod


def load_working_tree(exp_path: Path) -> Any:
    """Load and return the working tree from an experiment directory."""
    tree_path = exp_path / "working_tree.pkl"
    with tree_path.open("rb") as f:
        return pickle.load(f)


def compute_particle_weights(
    particle_indices: List[int],
    K: int,
    name_index: Dict[str, Any],
    divide_by_parent_weights: bool,
) -> np.ndarray:
    """Compute importance-sampling weights for all particles at step K."""
    return np.array(
        [
            get_product_of_weights(
                i, K, name_index, divide_by_parent_weights=divide_by_parent_weights,
            )
            for i in particle_indices
        ],
        dtype=float,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Return-period math
# ═══════════════════════════════════════════════════════════════════════════


def estimate_proba_res(
    scores: np.ndarray, quantile: float, prod_weights: np.ndarray
) -> float:
    """Estimate exceedance probability using RES importance-sampling weights."""
    safe_weights = np.where(prod_weights > 0, prod_weights, np.nan)
    inv_weights = 1.0 / safe_weights
    estimate = np.nanmean(inv_weights * (scores >= quantile))
    if np.isnan(estimate):
        return 0.0
    return float(estimate)


def compute_aires_return_period_years(proba_exp: np.ndarray) -> np.ndarray:
    """Convert exceedance probabilities to return periods in years."""
    proba = np.asarray(proba_exp, dtype=float).reshape(-1)
    return np.array([1.0 / p if p > 0 else 1e10 for p in proba], dtype=float)


def compute_weibull_return_periods(
    sorted_scores_desc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute return periods from descending-sorted scores using Weibull plotting position.

    Parameters
    ----------
    sorted_scores_desc : np.ndarray
        Scores sorted in descending order (largest first).

    Returns
    -------
    scores : np.ndarray
        The input scores (unchanged).
    return_periods : np.ndarray
        Weibull return periods: RP_i = (N+1) / rank_i.
    """
    scores = np.asarray(sorted_scores_desc, dtype=float).reshape(-1)
    n = scores.size
    if n == 0:
        return scores, np.array([], dtype=float)
    ranks = np.arange(1, n + 1, dtype=float)
    return_periods = (n + 1) / ranks
    return scores, return_periods


def compute_aires_curve(
    scores: np.ndarray, prod_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the full AI+RES return-period curve.

    Returns
    -------
    rv_exp : np.ndarray
        Sorted score values (ascending).
    rp_exp : np.ndarray
        Corresponding return periods in years.
    proba_exp : np.ndarray
        Exceedance probabilities.
    """
    rv_exp = np.sort(scores)
    proba_exp = np.array(
        [estimate_proba_res(scores, rv, prod_weights) for rv in rv_exp],
        dtype=float,
    )
    rp_exp = compute_aires_return_period_years(proba_exp)
    return rv_exp, rp_exp, proba_exp


# ═══════════════════════════════════════════════════════════════════════════
# DNS data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_dns_scores_from_npy(
    path: Path, subset_n: int = 400
) -> Tuple[np.ndarray, np.ndarray]:
    """Load DNS scores from a flat .npy file and return full + subset arrays.

    Parameters
    ----------
    path : Path
        Path to the .npy file containing one score per year/season.
    subset_n : int
        Number of years for the subset curve (first N entries).

    Returns
    -------
    full_scores : np.ndarray
        All scores from the file.
    subset_scores : np.ndarray
        First ``min(subset_n, len(full))`` scores.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DNS scores file not found: {path}")
    full_scores = np.load(path).astype(float).reshape(-1)
    actual_subset = min(subset_n, full_scores.size)
    subset_scores = full_scores[:actual_subset]
    return full_scores, subset_scores


def compute_dns_return_curve_from_npy(
    path: Path, subset_n: int = 400
) -> Dict[str, Any]:
    """Load DNS .npy and compute Weibull return curves for full + subset.

    Returns
    -------
    dict with keys:
        full_scores, full_rp, full_n,
        subset_scores, subset_rp, subset_n
    """
    full_scores, subset_scores = load_dns_scores_from_npy(path, subset_n=subset_n)

    full_sorted = np.sort(full_scores)[::-1]
    full_al, full_rp = compute_weibull_return_periods(full_sorted)

    subset_sorted = np.sort(subset_scores)[::-1]
    subset_al, subset_rp = compute_weibull_return_periods(subset_sorted)

    return {
        "full_scores": full_al,
        "full_rp": full_rp,
        "full_n": full_scores.size,
        "subset_scores": subset_al,
        "subset_rp": subset_rp,
        "subset_n": subset_scores.size,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════

# Color palette
CLR_AIRES = "#2563EB"
CLR_AIRES_EDGE = "#1E3A8A"
CLR_FULL = "#0EA5E9"
CLR_SUBSET = "#E11D48"


def format_region_label(region: str) -> str:
    """Format a region identifier into a human-readable label."""
    if not region:
        return "Unknown region"
    label = region.replace("_", " ")
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    if label.islower():
        label = label.title()
    return label


def setup_publication_style() -> dict:
    """Apply publication-quality rcParams. Returns previous values for restore."""
    prev = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
    })
    return prev


def restore_style(prev_params: dict) -> None:
    """Restore rcParams to the state captured by setup_publication_style."""
    plt.rcParams.update(prev_params)


def create_return_curve_figure(
    *, has_pdf_panel: bool = False
) -> Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]]:
    """Create a figure for return-curve plotting.

    Parameters
    ----------
    has_pdf_panel : bool
        If True, create a two-panel layout with a narrow right panel for
        the baseline PDF sidebar (blocking-specific).

    Returns
    -------
    fig, ax, ax_pdf : Tuple
        ax_pdf is None when has_pdf_panel is False.
    """
    if has_pdf_panel:
        fig = plt.figure(figsize=(10.5, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.0)
        ax = fig.add_subplot(gs[0])
        ax_pdf = fig.add_subplot(gs[1], sharey=ax)
        return fig, ax, ax_pdf
    else:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        return fig, ax, None


def plot_aires_scatter(
    ax: plt.Axes,
    rp_exp: np.ndarray,
    rv_exp: np.ndarray,
    n_particles: int,
) -> None:
    """Plot AI+RES scatter points on a return-curve axis."""
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


def plot_dns_line(
    ax: plt.Axes,
    return_periods: np.ndarray,
    scores: np.ndarray,
    *,
    label: str,
    color: str,
    alpha: float = 0.9,
    linestyle: Any = "-",
    linewidth: float = 3.6,
    marker: Optional[str] = None,
    markersize: float = 5.2,
    markevery: Optional[int] = None,
    zorder: int = 3,
) -> None:
    """Plot a DNS ground-truth curve as a line on a return-curve axis."""
    marker_kw: Dict[str, Any] = {}
    if marker is not None:
        marker_kw = dict(
            marker=marker,
            markersize=markersize,
            markevery=markevery if markevery else max(1, len(return_periods) // 80),
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.3,
        )
    ax.plot(
        return_periods,
        scores,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
        **marker_kw,
    )


def finalize_return_curve_axes(
    ax: plt.Axes,
    *,
    x_label: str = "Return period (years)",
    y_label: str = "Score",
    title: str = "",
    cap_rp_lower_at_tenth: bool = True,
    rp_arrays: Optional[List[np.ndarray]] = None,
    score_arrays: Optional[List[np.ndarray]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """Apply labels, grid, limits, legend, and styling to a return-curve axis."""
    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=13, labelpad=8)
    ax.set_ylabel(y_label, fontsize=13, labelpad=8)
    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.30, color="#888888")
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.15, color="#aaaaaa")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # X limits
    x_lower = 0.1 if cap_rp_lower_at_tenth else 0.01
    if rp_arrays:
        x_upper = max(max(float(np.max(rp)) for rp in rp_arrays) * 2.0, 100.0)
    else:
        x_upper = 100.0
    if x_upper <= x_lower:
        x_upper = x_lower * 10.0
    ax.set_xlim(x_lower, x_upper)

    # Y limits
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    elif score_arrays:
        y_min = min(float(np.min(v)) for v in score_arrays)
        y_max = max(float(np.max(v)) for v in score_arrays)
        if y_min == y_max:
            pad = max(1.0, abs(y_min) * 0.1)
        else:
            pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    # Legend
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


def save_return_curve_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    dpi: int = 200,
    show: bool = False,
) -> None:
    """Save (and optionally display) the return-curve figure."""
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=dpi, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    print(f"\nFigure saved to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()
