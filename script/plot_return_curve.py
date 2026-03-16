#!/usr/bin/env python3
"""
Unified return-period curve dispatcher.

Auto-detects the scorer type from an experiment and dispatches to the
correct plotting function:

  - HeatwaveMeanScorer  -> script.plot_return_curve_heatwave
  - All other scorers   -> script.plot_return_curve_blocking_area_pct

Usage::

    python script/plot_return_curve.py --exp_path /path/to/experiment --confirm-scorer detected
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")

# Ensure project root and src/ are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from forecast_analysis.scoring.detection import (
    SCORER_CHOICES,
    detect_scorer_from_experiment,
    format_spec,
    resolve_confirmed_scorer,
    resolve_scorer_profile,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEATWAVE_SCORERS = {"HeatwaveMeanScorer"}

LATEST_EXP_PATH_FILE = (
    _PROJECT_ROOT / "AI-RES" / "RES" / "experiments" / "logs" / "latest_exp_path.txt"
)

DEFAULT_HEATWAVE_DNS_PATH = str(
    _PROJECT_ROOT
    / "AI-RES"
    / "RES"
    / "experiments"
    / "saved_results"
    / "EXP15_AIRES_France"
    / "EXP15_AIRES_France_T.7_rv_control.npy"
)

DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "figures")
DEFAULT_K = 7
DEFAULT_SEASON_LABEL = "DJF"
DEFAULT_DNS_SUBSET_N = 400
DEFAULT_DNS_GROUND_TRUTH_WINDOW = "12-24_to_12-30"


# ---------------------------------------------------------------------------
# Helper: resolve default experiment path
# ---------------------------------------------------------------------------


def resolve_default_exp_path(latest_file: Path) -> Optional[str]:
    """Read the ``output_path`` line from *latest_file*.

    Returns ``None`` when the file is missing or does not contain the key.
    """
    if not latest_file.exists():
        return None
    try:
        text = latest_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("output_path:"):
            return line.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------


def detect_and_dispatch(
    exp_path: Path | str,
    K: int,
    output_dir: Path | str,
    confirm_scorer: Optional[str] = None,
    show_ground_truth: bool = True,
    dns_data_path: Optional[str] = None,
    dns_subset_n: int = DEFAULT_DNS_SUBSET_N,
    dns_ground_truth_window: str = DEFAULT_DNS_GROUND_TRUTH_WINDOW,
    season_label: str = DEFAULT_SEASON_LABEL,
    cap_rp_lower_at_tenth: bool = True,
    show_pdf_overlay: bool = True,
    pdf_file: Optional[str] = None,
    save_fig: bool = True,
    show_fig: bool = False,
) -> Dict[str, Any]:
    """Detect the scorer from *exp_path* and dispatch to the right plotter.

    Returns ``{"scorer_type": "heatwave"|"blocking",
               "scorer_name": str,
               "plot_result": <dict from the downstream plotter>}``.
    """
    exp_path = Path(exp_path)
    output_dir = Path(output_dir)

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {exp_path}")
    if not (exp_path / "working_tree.pkl").exists():
        raise FileNotFoundError(
            f"Missing working tree: {exp_path / 'working_tree.pkl'}"
        )

    # ---- Detect & confirm scorer ----
    detection = detect_scorer_from_experiment(exp_path)
    print("\n=== Scorer Detection ===")
    print(
        f"selected ({detection['selected_source']}): "
        f"{format_spec(detection['selected_scorer'])}"
    )

    confirmed_name, confirmed_params, confirm_origin = resolve_confirmed_scorer(
        detection["selected_scorer"],
        confirm_scorer,
    )
    scorer_profile = resolve_scorer_profile(confirmed_name, confirmed_params)

    print("\n=== Confirmed Scorer ===")
    print(f"choice origin: {confirm_origin}")
    print(f"scorer: {scorer_profile['raw_name']}")
    if scorer_profile["alias_note"]:
        print(f"alias mapping: {scorer_profile['alias_note']}")
    print(f"implementation: {scorer_profile['implementation']}")

    # ---- Dispatch ----
    if confirmed_name in HEATWAVE_SCORERS:
        scorer_type = "heatwave"

        from script.plot_return_curve_heatwave import plot_heatwave_return_curve

        effective_dns = dns_data_path
        if effective_dns is None:
            effective_dns = DEFAULT_HEATWAVE_DNS_PATH

        plot_result = plot_heatwave_return_curve(
            exp_path=exp_path,
            output_dir=output_dir,
            K=K,
            scorer_name=confirmed_name,
            scorer_params=confirmed_params,
            scorer_profile=scorer_profile,
            region=detection["region"],
            divide_by_parent_weights=bool(detection["divide_by_parent_weights"]),
            show_ground_truth=show_ground_truth,
            dns_data_path=Path(effective_dns) if effective_dns else None,
            dns_subset_n=dns_subset_n,
            cap_rp_lower_at_tenth=cap_rp_lower_at_tenth,
            save_fig=save_fig,
            show_fig=show_fig,
        )
    else:
        scorer_type = "blocking"

        from script.plot_return_curve_blocking_area_pct import plot_return_curve

        plot_result = plot_return_curve(
            exp_path=exp_path,
            output_dir=output_dir,
            K=K,
            season_label=season_label,
            scorer_name=confirmed_name,
            scorer_params=confirmed_params,
            scorer_profile=scorer_profile,
            region=detection["region"],
            divide_by_parent_weights=bool(detection["divide_by_parent_weights"]),
            clim_file=Path(str(detection["clim_file"])),
            threshold_json_file=Path(str(detection["threshold_json_file"])),
            show_ground_truth=show_ground_truth,
            ground_truth_window=dns_ground_truth_window,
            cap_rp_lower_at_tenth=cap_rp_lower_at_tenth,
            show_pdf_overlay=show_pdf_overlay,
            pdf_file=Path(pdf_file) if pdf_file else None,
            save_fig=save_fig,
            show_fig=show_fig,
        )

    return {
        "scorer_type": scorer_type,
        "scorer_name": confirmed_name,
        "plot_result": plot_result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified return-period curve plotter. "
            "Auto-detects scorer type and dispatches to the correct backend."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use latest experiment (from logs/latest_exp_path.txt)
    python script/plot_return_curve.py --confirm-scorer detected

    # Explicit experiment path
    python script/plot_return_curve.py \\
        --exp_path /path/to/experiment \\
        --confirm-scorer detected

    # Override scorer
    python script/plot_return_curve.py \\
        --exp_path /path/to/experiment \\
        --confirm-scorer GridpointIntensityScorer
        """,
    )

    parser.add_argument(
        "--exp_path",
        type=str,
        default=None,
        help=(
            "Path to experiment directory. "
            "If omitted, reads from AI-RES/RES/experiments/logs/latest_exp_path.txt."
        ),
    )
    parser.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        help="Final resampling step number (default: %(default)s)",
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
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures (default: %(default)s)",
    )
    parser.add_argument(
        "--show-ground-truth",
        dest="show_ground_truth",
        action="store_true",
        default=True,
        help="Overlay DNS ground-truth return curves (default: enabled).",
    )
    parser.add_argument(
        "--no-show-ground-truth",
        dest="show_ground_truth",
        action="store_false",
        help="Disable ground-truth return-curve overlay.",
    )
    parser.add_argument(
        "--dns-data-path",
        type=str,
        default=None,
        help=(
            "Path to DNS ground-truth data file. "
            "For heatwave: .npy file. For blocking: resolved automatically."
        ),
    )
    parser.add_argument(
        "--dns-subset-n",
        type=int,
        default=DEFAULT_DNS_SUBSET_N,
        help="Number of years for the DNS subset curve (default: %(default)s)",
    )
    parser.add_argument(
        "--dns-ground-truth-window",
        type=str,
        default=DEFAULT_DNS_GROUND_TRUTH_WINDOW,
        help=(
            "DNS ground-truth window selector for blocking overlay "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--season_label",
        type=str,
        default=DEFAULT_SEASON_LABEL,
        help="Season label for blocking plots (default: %(default)s)",
    )
    parser.add_argument(
        "--cap-rp-lower-at-tenth",
        dest="cap_rp_lower_at_tenth",
        action="store_true",
        default=True,
        help="Cap lower x-axis bound at 10^-1 years (default: enabled).",
    )
    parser.add_argument(
        "--no-cap-rp-lower-at-tenth",
        dest="cap_rp_lower_at_tenth",
        action="store_false",
        help="Use legacy 10^-2 lower bound.",
    )
    parser.add_argument(
        "--pdf-overlay",
        dest="pdf_overlay",
        action="store_true",
        default=True,
        help="Overlay baseline Z500 anomaly PDF (blocking only, default: enabled).",
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
        default=None,
        help="Path to the baseline Z500 anomaly PDF NetCDF file (blocking only).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively (in addition to saving).",
    )

    args = parser.parse_args()

    # ---- Resolve experiment path ----
    if args.exp_path is not None:
        exp_path_str = args.exp_path
    else:
        exp_path_str = resolve_default_exp_path(LATEST_EXP_PATH_FILE)
        if exp_path_str is None:
            parser.error(
                "No --exp_path supplied and could not resolve a default from "
                f"{LATEST_EXP_PATH_FILE}. Please provide --exp_path explicitly."
            )

    exp_path = Path(exp_path_str)
    if not exp_path.exists():
        print(f"Error: Experiment path does not exist: {exp_path}", file=sys.stderr)
        sys.exit(1)

    result = detect_and_dispatch(
        exp_path=exp_path,
        K=args.K,
        output_dir=args.output_dir,
        confirm_scorer=args.confirm_scorer,
        show_ground_truth=args.show_ground_truth,
        dns_data_path=args.dns_data_path,
        dns_subset_n=args.dns_subset_n,
        dns_ground_truth_window=args.dns_ground_truth_window,
        season_label=args.season_label,
        cap_rp_lower_at_tenth=args.cap_rp_lower_at_tenth,
        show_pdf_overlay=args.pdf_overlay,
        pdf_file=args.pdf_file,
        save_fig=True,
        show_fig=args.show,
    )

    print(f"\n=== Dispatch Summary ===")
    print(f"scorer type: {result['scorer_type']}")
    print(f"scorer name: {result['scorer_name']}")
    if "output_path" in result["plot_result"]:
        print(f"output: {result['plot_result']['output_path']}")


if __name__ == "__main__":
    main()
