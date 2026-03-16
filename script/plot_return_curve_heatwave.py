#!/usr/bin/env python3
"""
Return Period Curve for AI-RES heatwave outputs.

Plots AI+RES scatter points overlaid on DNS ground-truth return curves
for HeatwaveMeanScorer experiments. Scores are plotted in °C.

DNS ground truth is loaded from a flat .npy file (one score per year)
and converted to return periods using Weibull plotting position.

Unlike the blocking return-curve script, this does NOT include a PDF
sidebar panel or date histogram.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

import numpy as np

# Ensure project root and src/ are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from forecast_analysis.scoring.detection import (
    detect_scorer_from_experiment,
    format_spec,
    resolve_confirmed_scorer,
    resolve_scorer_profile,
)
from src.return_curve import (
    CLR_FULL,
    CLR_SUBSET,
    build_name_index,
    compute_aires_curve,
    compute_dns_return_curve_from_npy,
    compute_particle_weights,
    create_return_curve_figure,
    finalize_return_curve_axes,
    format_region_label,
    get_particle_indices_for_step,
    load_working_tree,
    plot_aires_scatter,
    plot_dns_line,
    save_return_curve_figure,
    setup_publication_style,
    restore_style,
)


# =========================================================================
# Configuration
# =========================================================================

DEFAULT_EXP_PATH = (
    "/glade/derecho/scratch/zhil/PLASIM/RES/experiments/EXP15_AIRES_France/"
    "EXP15_AIRES_France_0"
)
DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "figures" / "AI-RES-return_curve")
DEFAULT_K = 7
DEFAULT_DNS_DATA_PATH = str(
    _PROJECT_ROOT
    / "AI-RES"
    / "RES"
    / "experiments"
    / "saved_results"
    / "EXP15_AIRES_France"
    / "EXP15_AIRES_France_T.7_rv_control.npy"
)
DEFAULT_DNS_SUBSET_N = 400
DEFAULT_SHOW_GROUND_TRUTH = True
DEFAULT_CAP_RP_LOWER_AT_TENTH = True
KELVIN_OFFSET = 273.15


# =========================================================================
# Main plotting
# =========================================================================


def plot_heatwave_return_curve(
    exp_path: Path,
    output_dir: Path,
    K: int,
    scorer_name: str,
    scorer_params: Dict[str, Any],
    scorer_profile: Dict[str, Any],
    region: str,
    divide_by_parent_weights: bool,
    show_ground_truth: bool = DEFAULT_SHOW_GROUND_TRUTH,
    dns_data_path: Path | None = None,
    dns_subset_n: int = DEFAULT_DNS_SUBSET_N,
    cap_rp_lower_at_tenth: bool = DEFAULT_CAP_RP_LOWER_AT_TENTH,
    save_fig: bool = True,
    show_fig: bool = False,
) -> Dict[str, Any]:
    """Plot the heatwave return-period curve. Scores converted K -> °C."""
    import matplotlib.pyplot as plt

    exp_name = exp_path.name

    working_tree = load_working_tree(exp_path)
    name_index = build_name_index(working_tree)
    print("Loaded working tree")

    particle_indices = get_particle_indices_for_step(K, name_index)
    n_particles = len(particle_indices)
    print(
        f"Found {n_particles} particles at step {K}: "
        f"indices [{particle_indices[0]} ... {particle_indices[-1]}]"
    )

    saved_path = exp_path / "resampling" / f"plasim_scores_step_{K}.npy"
    if not saved_path.exists():
        raise FileNotFoundError(
            f"No saved PlaSim scores at {saved_path}. "
            "HeatwaveMeanScorer requires pre-computed scores from the QDMC run."
        )
    print(f"Loading pre-computed PlaSim scores from: {saved_path}")
    all_scores = np.load(saved_path)
    scores_k = np.asarray(
        [float(all_scores[idx]) for idx in particle_indices], dtype=float,
    )

    scores_c = scores_k - KELVIN_OFFSET
    print(
        f"Scores (°C): N={n_particles}, "
        f"range=[{scores_c.min():.2f}, {scores_c.max():.2f}]"
    )

    print(f"Computing weights for {n_particles} particles...")
    prod_weights = compute_particle_weights(
        particle_indices, K, name_index, divide_by_parent_weights,
    )
    del name_index, working_tree

    if divide_by_parent_weights:
        print("Weight mode: single-parent correction")
    else:
        print("Weight mode: ancestry-product correction")
    print(
        f"Product of weights: min={prod_weights.min():.4f}, "
        f"max={prod_weights.max():.4f}, mean={prod_weights.mean():.4f}"
    )

    rv_exp, rp_exp, proba_exp = compute_aires_curve(scores_c, prod_weights)
    print(f"Return period range: {rp_exp.min():.4f} to {rp_exp.max():.4f} years")

    dns_curve = None
    if show_ground_truth and dns_data_path is not None:
        dns_path = Path(dns_data_path)
        if dns_path.exists():
            print(f"Loading DNS ground truth from: {dns_path}")
            dns_curve = compute_dns_return_curve_from_npy(
                dns_path, subset_n=dns_subset_n,
            )
            print(
                f"DNS: {dns_curve['full_n']} years (full), "
                f"{dns_curve['subset_n']} years (subset)"
            )
        else:
            print(f"Warning: DNS data not found at {dns_path}, skipping overlay.")

    _rc_prev = setup_publication_style()
    fig, ax, _ = create_return_curve_figure(has_pdf_panel=False)

    plot_aires_scatter(ax, rp_exp, rv_exp, n_particles)

    rp_arrays = [rp_exp]
    score_arrays = [rv_exp]

    if dns_curve is not None:
        plot_dns_line(
            ax, dns_curve["full_rp"], dns_curve["full_scores"],
            label=f"Full ensemble ({dns_curve['full_n']:,} years)",
            color=CLR_FULL, zorder=3,
        )
        plot_dns_line(
            ax, dns_curve["subset_rp"], dns_curve["subset_scores"],
            label=f"First {dns_curve['subset_n']:,} years",
            color=CLR_SUBSET, alpha=0.85,
            linestyle=(0, (1.5, 2)), marker="s", zorder=4,
        )
        rp_arrays.extend([dns_curve["full_rp"], dns_curve["subset_rp"]])
        score_arrays.extend([dns_curve["full_scores"], dns_curve["subset_scores"]])

    region_label = format_region_label(region)
    metric_label = scorer_profile.get("metric_label", scorer_profile["score_title"])
    finalize_return_curve_axes(
        ax,
        x_label="Return period (years)",
        y_label=r"Mean temperature ($\degree$C)",
        title=(
            f"Return-period curve of heatwave in {region_label}\n"
            f"{metric_label}"
        ),
        cap_rp_lower_at_tenth=cap_rp_lower_at_tenth,
        rp_arrays=rp_arrays,
        score_arrays=score_arrays,
    )

    output_path = output_dir / (
        f"return_curve_{exp_name}_{scorer_profile['score_key']}_plasim.png"
    )
    if save_fig:
        save_return_curve_figure(fig, output_path, show=show_fig)
    else:
        if show_fig:
            plt.show()
        else:
            plt.close()

    restore_style(_rc_prev)

    return {
        "rv_exp": rv_exp,
        "rp_exp": rp_exp,
        "proba_exp": proba_exp,
        "prod_weights": prod_weights,
        "scores_celsius": scores_c,
        "particle_indices": particle_indices,
        "output_path": str(output_path),
        "scorer_profile": scorer_profile,
    }


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot return period curve for AI-RES heatwave outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python script/plot_return_curve_heatwave.py
    python script/plot_return_curve_heatwave.py \\
        --exp_path /path/to/experiment \\
        --confirm-scorer detected
        """,
    )
    parser.add_argument("--exp_path", type=str, default=DEFAULT_EXP_PATH,
                        help="Path to experiment directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save output figures")
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help="Final resampling step number")
    parser.add_argument(
        "--confirm-scorer", type=str, default=None,
        choices=["detected", "HeatwaveMeanScorer"],
        help=(
            "Skip interactive prompt and confirm scorer directly. "
            "Only 'detected' and 'HeatwaveMeanScorer' are accepted; "
            "this wrapper is exclusively for heatwave experiments."
        ),
    )
    parser.add_argument("--dns-data-path", type=str, default=DEFAULT_DNS_DATA_PATH,
                        help="Path to DNS ground-truth scores .npy file")
    parser.add_argument("--dns-subset-n", type=int, default=DEFAULT_DNS_SUBSET_N,
                        help="Number of years for the DNS subset curve")
    parser.add_argument("--show-ground-truth", dest="show_ground_truth",
                        action="store_true", default=DEFAULT_SHOW_GROUND_TRUTH,
                        help="Overlay DNS ground-truth return curves (default: enabled).")
    parser.add_argument("--no-show-ground-truth", dest="show_ground_truth",
                        action="store_false",
                        help="Disable DNS ground-truth overlay.")
    parser.add_argument("--cap-rp-lower-at-tenth", dest="cap_rp_lower_at_tenth",
                        action="store_true", default=DEFAULT_CAP_RP_LOWER_AT_TENTH,
                        help="Cap lower x-axis bound at 10^-1 years (default: enabled).")
    parser.add_argument("--no-cap-rp-lower-at-tenth", dest="cap_rp_lower_at_tenth",
                        action="store_false", help="Use legacy 10^-2 lower bound.")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure interactively.")
    args = parser.parse_args()

    exp_path = Path(args.exp_path)
    output_dir = Path(args.output_dir)
    if not exp_path.exists():
        print(f"Error: Experiment path does not exist: {exp_path}", file=sys.stderr)
        sys.exit(1)
    if not (exp_path / "working_tree.pkl").exists():
        print(f"Error: Missing working tree: {exp_path / 'working_tree.pkl'}", file=sys.stderr)
        sys.exit(1)

    detection = detect_scorer_from_experiment(exp_path)
    print("\n=== Scorer Detection ===")
    print(f"selected ({detection['selected_source']}): {format_spec(detection['selected_scorer'])}")

    confirmed_name, confirmed_params, confirm_origin = resolve_confirmed_scorer(
        detection["selected_scorer"],
        args.confirm_scorer,
    )

    # Fail fast: this wrapper only supports HeatwaveMeanScorer
    if confirmed_name != "HeatwaveMeanScorer":
        print(
            f"Error: This script only supports HeatwaveMeanScorer, "
            f"but the confirmed scorer is '{confirmed_name}'.\n"
            "Use script/plot_return_curve_blocking_area_pct.py for blocking scorers.",
            file=sys.stderr,
        )
        sys.exit(1)

    scorer_profile = resolve_scorer_profile(confirmed_name, confirmed_params)

    print(f"\n=== Confirmed Scorer ===")
    print(f"scorer: {scorer_profile['raw_name']}")
    print(f"implementation: {scorer_profile['implementation']}")

    print(f"\n=== Score Source ===")
    print("score source: plasim (saved scores only)")
    print(f"divide_by_parent_weights: {detection['divide_by_parent_weights']}")
    print(f"DNS data path: {args.dns_data_path}")
    print(f"DNS subset size: {args.dns_subset_n}")

    plot_heatwave_return_curve(
        exp_path=exp_path, output_dir=output_dir, K=args.K,
        scorer_name=confirmed_name, scorer_params=confirmed_params,
        scorer_profile=scorer_profile, region=detection["region"],
        divide_by_parent_weights=bool(detection["divide_by_parent_weights"]),
        show_ground_truth=args.show_ground_truth,
        dns_data_path=Path(args.dns_data_path),
        dns_subset_n=args.dns_subset_n,
        cap_rp_lower_at_tenth=args.cap_rp_lower_at_tenth,
        save_fig=True, show_fig=args.show,
    )


if __name__ == "__main__":
    main()
