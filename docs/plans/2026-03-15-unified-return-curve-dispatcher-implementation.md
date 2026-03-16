# Unified Score-Agnostic Return-Curve Dispatcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a unified `script/plot_return_curve.py` that auto-detects the scorer type from an experiment and dispatches to the correct plotting function, plus rewrite the skill doc to be score-agnostic.

**Architecture:** New dispatcher script imports plotting functions from both existing scripts. It reads `latest_exp_path.txt` for the default experiment, calls `detect_scorer_from_experiment()`, then branches: HeatwaveMeanScorer → `plot_heatwave_return_curve()`, all others → `plot_return_curve()`. Scorer-specific defaults (DNS paths, PDF panel) are resolved internally.

**Tech Stack:** Python 3, matplotlib, numpy, argparse, pytest

**Design doc:** `docs/plans/2026-03-15-unified-return-curve-dispatcher-design.md`

---

### Task 1: Write tests for the unified dispatcher

**Files:**
- Create: `tests/script/test_plot_return_curve_unified.py`

**Context:**
- The existing test helper `_FakeNode` and `_make_fake_experiment` in `tests/script/test_plot_return_curve_heatwave.py` create minimal experiment directories with `working_tree.pkl`, `used_config.json`, and `plasim_scores_step_K.npy`. Reuse the same pattern.
- The unified script will be at `script/plot_return_curve.py`.
- It will have a function `resolve_default_exp_path()` that reads `latest_exp_path.txt`.
- It will have a function `dispatch_plot(exp_path, args)` that detects the scorer and calls the right plotting function.
- The existing `detect_scorer_from_experiment` (in `forecast_analysis/scoring/detection.py`) reads `<exp_name>_used_config.json` from the experiment directory.

**Step 1: Write the test file**

```python
"""Tests for script/plot_return_curve.py (unified dispatcher)."""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class _FakeNode:
    """Pickle-friendly stand-in for a working-tree node."""

    def __init__(self, name, parent=None, V=None, norm_constant=None):
        self.name = name
        self.parent = parent
        self.V = V
        self.norm_constant = norm_constant
        self.descendants = []


def _make_fake_experiment(tmp_path, scorer_name="HeatwaveMeanScorer", K=7, n_particles=10):
    """Create a minimal experiment directory."""
    exp_name = "FakeExp_0"
    exp_dir = tmp_path / exp_name
    exp_dir.mkdir()

    config = {
        "scorer": {"name": scorer_name, "params": {}},
        "region": "France",
        "divide_by_parent_weights": True,
    }
    (exp_dir / f"{exp_name}_used_config.json").write_text(json.dumps(config))

    root = _FakeNode("root")
    children = []
    for i in range(n_particles):
        parent_node = _FakeNode(
            f"parent_{K}_{i}", parent=root, V=0.0, norm_constant=1.0,
        )
        child = _FakeNode(f"step_{K}_particle_{i}", parent=parent_node)
        children.append(child)
    root.descendants = children
    root.parent = None

    with (exp_dir / "working_tree.pkl").open("wb") as f:
        pickle.dump(root, f)

    scores = np.linspace(299.0, 307.0, n_particles)
    resampling_dir = exp_dir / "resampling"
    resampling_dir.mkdir()
    np.save(resampling_dir / f"plasim_scores_step_{K}.npy", scores)

    return exp_dir


class TestUnifiedCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "exp_path" in result.stdout
        assert "confirm-scorer" in result.stdout

    def test_missing_exp_path_errors(self):
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve.py",
             "--exp_path", "/nonexistent/path",
             "--confirm-scorer", "detected"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestDefaultExpPath:
    def test_resolve_default_exp_path_reads_latest(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import resolve_default_exp_path

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        latest = logs_dir / "latest_exp_path.txt"
        latest.write_text(
            "job_id: 123\n"
            "exp_name: TestExp_0\n"
            "output_path: /some/experiment/path\n"
            "submitted: Mon Jan 01 12:00:00 PM MDT 2026\n"
        )

        result = resolve_default_exp_path(latest)
        assert result == "/some/experiment/path"

    def test_resolve_default_exp_path_missing_file(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import resolve_default_exp_path

        result = resolve_default_exp_path(tmp_path / "nonexistent.txt")
        assert result is None


class TestScorerDispatch:
    def test_dispatches_heatwave(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import detect_and_dispatch

        K = 7
        n_particles = 10
        exp_dir = _make_fake_experiment(
            tmp_path, scorer_name="HeatwaveMeanScorer", K=K, n_particles=n_particles,
        )

        dns_path = tmp_path / "dns_scores.npy"
        np.save(dns_path, np.random.default_rng(42).normal(28.0, 2.0, size=500))

        result = detect_and_dispatch(
            exp_path=exp_dir, K=K, output_dir=tmp_path / "figures",
            confirm_scorer="detected", show_ground_truth=True,
            dns_data_path=str(dns_path), dns_subset_n=100,
            save_fig=True, show_fig=False,
        )

        assert result["scorer_type"] == "heatwave"
        assert "scores_celsius" in result["plot_result"]
        assert Path(result["plot_result"]["output_path"]).exists()

    def test_dispatches_blocking(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import detect_and_dispatch

        K = 7
        n_particles = 10
        exp_dir = _make_fake_experiment(
            tmp_path, scorer_name="GridpointIntensityScorer", K=K, n_particles=n_particles,
        )

        result = detect_and_dispatch(
            exp_path=exp_dir, K=K, output_dir=tmp_path / "figures",
            confirm_scorer="detected", show_ground_truth=False,
            save_fig=True, show_fig=False,
        )

        assert result["scorer_type"] == "blocking"
        assert "scores" in result["plot_result"]
        assert Path(result["plot_result"]["output_path"]).exists()
```

**Step 2: Run the tests to verify they fail**

Run: `cd /glade/u/home/zhil/project/AI-RES/Blocking && PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH" python -m pytest tests/script/test_plot_return_curve_unified.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'script.plot_return_curve'`

**Step 3: Commit the tests**

```bash
git add tests/script/test_plot_return_curve_unified.py
git commit -m "test: add tests for unified return-curve dispatcher"
```

---

### Task 2: Create the unified dispatcher script

**Files:**
- Create: `script/plot_return_curve.py`

**Context:**
- Must import `plot_heatwave_return_curve` from `script.plot_return_curve_heatwave`
- Must import `plot_return_curve` from `script.plot_return_curve_blocking_area_pct`
- Must import `detect_scorer_from_experiment`, `resolve_confirmed_scorer`, `resolve_scorer_profile`, `format_spec` from `forecast_analysis.scoring.detection`
- The blocking script's `plot_return_curve()` function signature requires: `exp_path, output_dir, K, season_label, scorer_name, scorer_params, scorer_profile, region, divide_by_parent_weights, clim_file, threshold_json_file` plus optional ground-truth and PDF overlay args.
- The heatwave script's `plot_heatwave_return_curve()` function signature requires: `exp_path, output_dir, K, scorer_name, scorer_params, scorer_profile, region, divide_by_parent_weights` plus optional ground-truth args.
- `latest_exp_path.txt` is at `AI-RES/RES/experiments/logs/latest_exp_path.txt` relative to project root. Format is `key: value` lines; we need the `output_path` value.
- Default heatwave DNS path: `AI-RES/RES/experiments/saved_results/EXP15_AIRES_France/EXP15_AIRES_France_T.7_rv_control.npy`
- Default blocking DNS: resolved by `resolve_dns_ground_truth_files()` in the blocking script.

**Step 1: Write the dispatcher script**

```python
#!/usr/bin/env python3
"""
Unified return-curve plotting for AI-RES experiments.

Auto-detects the scorer type from an experiment directory and dispatches
to the appropriate plotting function (blocking or heatwave). This script
is the recommended entry point for return-curve generation.

For direct access to scorer-specific scripts:
  - script/plot_return_curve_blocking_area_pct.py  (blocking scorers)
  - script/plot_return_curve_heatwave.py           (HeatwaveMeanScorer)
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


# =========================================================================
# Configuration
# =========================================================================

LATEST_EXP_PATH_FILE = (
    _PROJECT_ROOT / "AI-RES" / "RES" / "experiments" / "logs" / "latest_exp_path.txt"
)
DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "figures")
DEFAULT_K = 7
DEFAULT_SEASON_LABEL = "DJF"
DEFAULT_DNS_SUBSET_N = 400

# Heatwave-specific defaults
DEFAULT_HEATWAVE_DNS_PATH = str(
    _PROJECT_ROOT
    / "AI-RES"
    / "RES"
    / "experiments"
    / "saved_results"
    / "EXP15_AIRES_France"
    / "EXP15_AIRES_France_T.7_rv_control.npy"
)

HEATWAVE_SCORERS = {"HeatwaveMeanScorer"}


# =========================================================================
# Default experiment path resolution
# =========================================================================


def resolve_default_exp_path(latest_file: Path = LATEST_EXP_PATH_FILE) -> Optional[str]:
    """Read ``output_path`` from the latest experiment log file.

    Returns None if the file does not exist or cannot be parsed.
    """
    if not latest_file.exists():
        return None
    try:
        text = latest_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("output_path:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# =========================================================================
# Dispatch logic
# =========================================================================


def detect_and_dispatch(
    exp_path: Path,
    K: int,
    output_dir: Path,
    confirm_scorer: Optional[str] = None,
    show_ground_truth: bool = True,
    dns_data_path: Optional[str] = None,
    dns_subset_n: int = DEFAULT_DNS_SUBSET_N,
    dns_ground_truth_window: str = "12-24_to_12-30",
    season_label: str = DEFAULT_SEASON_LABEL,
    cap_rp_lower_at_tenth: bool = True,
    show_pdf_overlay: bool = True,
    pdf_file: Optional[str] = None,
    save_fig: bool = True,
    show_fig: bool = False,
) -> Dict[str, Any]:
    """Detect scorer from experiment and dispatch to the right plotter.

    Returns
    -------
    dict with keys:
        scorer_type : str
            "heatwave" or "blocking"
        scorer_name : str
            Confirmed scorer name.
        plot_result : dict
            Result dict from the plotting function.
    """
    exp_path = Path(exp_path)

    detection = detect_scorer_from_experiment(exp_path)
    print("\n=== Scorer Detection ===")
    print(f"selected ({detection['selected_source']}): {format_spec(detection['selected_scorer'])}")

    confirmed_name, confirmed_params, confirm_origin = resolve_confirmed_scorer(
        detection["selected_scorer"],
        confirm_scorer,
    )
    scorer_profile = resolve_scorer_profile(confirmed_name, confirmed_params)

    print(f"\n=== Confirmed Scorer ===")
    print(f"scorer: {scorer_profile['raw_name']}")
    print(f"implementation: {scorer_profile['implementation']}")

    region = detection["region"]
    divide_by_parent_weights = bool(detection["divide_by_parent_weights"])

    if confirmed_name in HEATWAVE_SCORERS:
        return _dispatch_heatwave(
            exp_path=exp_path, K=K, output_dir=output_dir,
            confirmed_name=confirmed_name, confirmed_params=confirmed_params,
            scorer_profile=scorer_profile, region=region,
            divide_by_parent_weights=divide_by_parent_weights,
            show_ground_truth=show_ground_truth,
            dns_data_path=dns_data_path, dns_subset_n=dns_subset_n,
            cap_rp_lower_at_tenth=cap_rp_lower_at_tenth,
            save_fig=save_fig, show_fig=show_fig,
        )
    else:
        return _dispatch_blocking(
            exp_path=exp_path, K=K, output_dir=output_dir,
            confirmed_name=confirmed_name, confirmed_params=confirmed_params,
            scorer_profile=scorer_profile, region=region,
            divide_by_parent_weights=divide_by_parent_weights,
            detection=detection, season_label=season_label,
            show_ground_truth=show_ground_truth,
            dns_ground_truth_window=dns_ground_truth_window,
            cap_rp_lower_at_tenth=cap_rp_lower_at_tenth,
            show_pdf_overlay=show_pdf_overlay, pdf_file=pdf_file,
            save_fig=save_fig, show_fig=show_fig,
        )


def _dispatch_heatwave(
    exp_path, K, output_dir,
    confirmed_name, confirmed_params, scorer_profile,
    region, divide_by_parent_weights,
    show_ground_truth, dns_data_path, dns_subset_n,
    cap_rp_lower_at_tenth, save_fig, show_fig,
):
    """Dispatch to the heatwave plotting function."""
    from script.plot_return_curve_heatwave import plot_heatwave_return_curve

    if dns_data_path is None:
        dns_data_path = DEFAULT_HEATWAVE_DNS_PATH

    print(f"\n=== Dispatching: Heatwave Return Curve ===")
    print(f"DNS data path: {dns_data_path}")

    plot_result = plot_heatwave_return_curve(
        exp_path=Path(exp_path), output_dir=Path(output_dir), K=K,
        scorer_name=confirmed_name, scorer_params=confirmed_params,
        scorer_profile=scorer_profile, region=region,
        divide_by_parent_weights=divide_by_parent_weights,
        show_ground_truth=show_ground_truth,
        dns_data_path=Path(dns_data_path), dns_subset_n=dns_subset_n,
        cap_rp_lower_at_tenth=cap_rp_lower_at_tenth,
        save_fig=save_fig, show_fig=show_fig,
    )
    return {"scorer_type": "heatwave", "scorer_name": confirmed_name, "plot_result": plot_result}


def _dispatch_blocking(
    exp_path, K, output_dir,
    confirmed_name, confirmed_params, scorer_profile,
    region, divide_by_parent_weights, detection,
    season_label, show_ground_truth, dns_ground_truth_window,
    cap_rp_lower_at_tenth, show_pdf_overlay, pdf_file,
    save_fig, show_fig,
):
    """Dispatch to the blocking plotting function."""
    from script.plot_return_curve_blocking_area_pct import plot_return_curve

    clim_file = Path(str(detection["clim_file"]))
    threshold_json_file = Path(str(detection["threshold_json_file"]))

    print(f"\n=== Dispatching: Blocking Return Curve ===")
    print(f"climatology file: {clim_file}")
    print(f"threshold json file: {threshold_json_file}")

    plot_kwargs = dict(
        exp_path=Path(exp_path), output_dir=Path(output_dir), K=K,
        season_label=season_label,
        scorer_name=confirmed_name, scorer_params=confirmed_params,
        scorer_profile=scorer_profile, region=region,
        divide_by_parent_weights=divide_by_parent_weights,
        clim_file=clim_file, threshold_json_file=threshold_json_file,
        show_ground_truth=show_ground_truth,
        ground_truth_window=dns_ground_truth_window,
        cap_rp_lower_at_tenth=cap_rp_lower_at_tenth,
        show_pdf_overlay=show_pdf_overlay,
        save_fig=save_fig, show_fig=show_fig,
    )
    if pdf_file is not None:
        plot_kwargs["pdf_file"] = Path(pdf_file)

    plot_result = plot_return_curve(**plot_kwargs)
    return {"scorer_type": "blocking", "scorer_name": confirmed_name, "plot_result": plot_result}


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    default_exp = resolve_default_exp_path()

    parser = argparse.ArgumentParser(
        description=(
            "Unified return-curve plotter for AI-RES experiments. "
            "Auto-detects the scorer type and dispatches to the correct plotter."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot latest experiment (auto-detect scorer)
    python script/plot_return_curve.py --confirm-scorer detected

    # Specify experiment path
    python script/plot_return_curve.py --exp_path /path/to/experiment --confirm-scorer detected

    # Override DNS data path for heatwave
    python script/plot_return_curve.py --exp_path /path/to/experiment \\
        --confirm-scorer detected --dns-data-path /path/to/dns_scores.npy
        """,
    )
    parser.add_argument(
        "--exp_path", type=str, default=default_exp,
        help=(
            "Path to experiment directory. "
            "Default: read from AI-RES/RES/experiments/logs/latest_exp_path.txt"
        ),
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save output figures")
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help="Final resampling step number")
    parser.add_argument(
        "--confirm-scorer", type=str, default=None,
        choices=["detected", *SCORER_CHOICES],
        help="Skip interactive prompt and confirm scorer directly.",
    )
    parser.add_argument("--season_label", type=str, default=DEFAULT_SEASON_LABEL,
                        help="Season label (blocking only, default: DJF)")
    parser.add_argument("--show-ground-truth", dest="show_ground_truth",
                        action="store_true", default=True)
    parser.add_argument("--no-show-ground-truth", dest="show_ground_truth",
                        action="store_false")
    parser.add_argument("--dns-data-path", type=str, default=None,
                        help="Override DNS data path (heatwave: .npy file)")
    parser.add_argument("--dns-subset-n", type=int, default=DEFAULT_DNS_SUBSET_N,
                        help="Number of years for DNS subset curve (heatwave only)")
    parser.add_argument("--dns-ground-truth-window", type=str, default="12-24_to_12-30",
                        choices=["djf", "12-24_to_12-30"],
                        help="DNS ground-truth window (blocking only)")
    parser.add_argument("--cap-rp-lower-at-tenth", dest="cap_rp_lower_at_tenth",
                        action="store_true", default=True)
    parser.add_argument("--no-cap-rp-lower-at-tenth", dest="cap_rp_lower_at_tenth",
                        action="store_false")
    parser.add_argument("--pdf-overlay", dest="pdf_overlay",
                        action="store_true", default=True,
                        help="Show PDF sidebar (blocking intensity only)")
    parser.add_argument("--no-pdf-overlay", dest="pdf_overlay",
                        action="store_false")
    parser.add_argument("--pdf-file", type=str, default=None,
                        help="Override baseline PDF file path (blocking only)")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure interactively")
    args = parser.parse_args()

    if args.exp_path is None:
        print(
            "Error: No experiment path provided and could not resolve default from\n"
            f"  {LATEST_EXP_PATH_FILE}\n"
            "Use --exp_path to specify the experiment directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    exp_path = Path(args.exp_path)
    if not exp_path.exists():
        print(f"Error: Experiment path does not exist: {exp_path}", file=sys.stderr)
        sys.exit(1)
    if not (exp_path / "working_tree.pkl").exists():
        print(f"Error: Missing working tree: {exp_path / 'working_tree.pkl'}", file=sys.stderr)
        sys.exit(1)

    detect_and_dispatch(
        exp_path=exp_path, K=args.K, output_dir=Path(args.output_dir),
        confirm_scorer=args.confirm_scorer,
        show_ground_truth=args.show_ground_truth,
        dns_data_path=args.dns_data_path, dns_subset_n=args.dns_subset_n,
        dns_ground_truth_window=args.dns_ground_truth_window,
        season_label=args.season_label,
        cap_rp_lower_at_tenth=args.cap_rp_lower_at_tenth,
        show_pdf_overlay=args.pdf_overlay, pdf_file=args.pdf_file,
        save_fig=True, show_fig=args.show,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Run the tests to verify they pass**

Run: `cd /glade/u/home/zhil/project/AI-RES/Blocking && PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH" python -m pytest tests/script/test_plot_return_curve_unified.py -v`
Expected: All 6 tests pass.

**Step 3: Commit**

```bash
git add script/plot_return_curve.py
git commit -m "feat: add unified score-agnostic return-curve dispatcher"
```

---

### Task 3: Rewrite the skill doc

**Files:**
- Modify: `.skills/plot/plot_return_curve.md`

**Context:**
- The skill doc is the primary interface for the AI agent.
- Lead with the unified script as the default workflow.
- Include "Direct Script Access" section for the two individual scripts.
- Keep environment activation instructions.
- Keep the "Finding The Latest Experiment Path" section since the unified script reads it automatically.

**Step 1: Rewrite the skill doc**

Replace the entire contents of `.skills/plot/plot_return_curve.md` with:

```markdown
# Plot Return Curve for AI-RES Output

Compact runbook for generating return-period curves from any AI-RES experiment.
The unified script auto-detects the scorer type and dispatches to the correct plotter.

Script:
`/glade/u/home/zhil/project/AI-RES/Blocking/script/plot_return_curve.py`

Detailed CLI/reference/troubleshooting content lives in:
- `plot_return_curve_reference.md`

## Quick Start

Activate the AI-RES environment first:

```bash
source /glade/u/apps/opt/miniforge/25.11/etc/profile.d/conda.sh
conda activate /glade/work/zhil/conda_envs/aires
```

Plot the latest experiment (scorer auto-detected):

```bash
python script/plot_return_curve.py --confirm-scorer detected
```

Plot a specific experiment:

```bash
python script/plot_return_curve.py --exp_path /path/to/experiment --confirm-scorer detected
```

The script reads `AI-RES/RES/experiments/logs/latest_exp_path.txt` for the
default experiment path when `--exp_path` is not provided.

## How It Works

1. Reads the experiment's `_used_config.json` to detect the scorer
2. Based on detected scorer:
   - **HeatwaveMeanScorer** -> heatwave plotter (K->C conversion, .npy DNS, no PDF panel)
   - **Blocking scorers** (GridpointIntensityScorer, GridpointPersistenceScorer, ANOScorer, etc.) -> blocking plotter (NPZ DNS, optional PDF sidebar)
3. DNS ground-truth defaults are resolved automatically per scorer type
4. Figure saved to `figures/` by default

## Required Inputs (Minimal Checklist)

1. Experiment directory with `working_tree.pkl`
2. Pre-computed scores: `resampling/plasim_scores_step_{K}.npy`
3. Scorer metadata: `<exp_name>_used_config.json` (preferred) or `outerr_forecasts.log` (fallback)

## How To Pick `K`

Use the highest `resampling/plasim_scores_step_*.npy` (typically one step ahead
of the last resampling step in standard QDMC runs):

```bash
ls /path/to/experiment/resampling/plasim_scores_step_*.npy
```

## Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--exp_path` | from `latest_exp_path.txt` | Experiment directory |
| `--K` | 7 | Final resampling step |
| `--confirm-scorer` | None (interactive) | `detected` to accept auto-detected scorer |
| `--output_dir` | `figures/` | Output directory |
| `--show-ground-truth` / `--no-show-ground-truth` | enabled | Toggle DNS overlay |
| `--show` | false | Display figure interactively |

### Blocking-only flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dns-ground-truth-window` | `12-24_to_12-30` | DNS window (`djf` or `12-24_to_12-30`) |
| `--pdf-overlay` / `--no-pdf-overlay` | enabled | PDF sidebar (GridpointIntensityScorer only) |
| `--pdf-file` | built-in default | Override baseline PDF NetCDF path |
| `--season_label` | `DJF` | Season label on title |

### Heatwave-only flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dns-data-path` | auto-resolved | Path to DNS scores `.npy` file |
| `--dns-subset-n` | 400 | Number of years for subset curve |

For full CLI:

```bash
python script/plot_return_curve.py --help
```

## Scorer-Specific Defaults

| Scorer | DNS format | DNS default | Unit conversion | PDF panel |
|--------|-----------|-------------|-----------------|-----------|
| HeatwaveMeanScorer | .npy (Weibull) | `AI-RES/.../EXP15_AIRES_France_T.7_rv_control.npy` | K -> C | No |
| GridpointIntensityScorer | .npz (block maxima) | `data/return_curve_ground_truth/` | None | Yes |
| Other blocking scorers | .npz | `data/return_curve_ground_truth/` | None | No |

## Common Issues (Quick Triage)

- Missing `working_tree.pkl`: wrong or incomplete experiment path
- Missing `plasim_scores_step_K.npy`: check the experiment completed the QDMC run
- Non-interactive environment: pass `--confirm-scorer detected`
- DNS overlay warnings: check DNS data files exist at default or specified paths
- No default experiment: provide `--exp_path` explicitly

## Finding The Latest Experiment Path

The unified script reads the default from:

```text
/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/experiments/logs/latest_exp_path.txt
```

The `output_path` value is used as `--exp_path`.

## Direct Script Access

For explicit control, use the scorer-specific scripts directly:

### Blocking

```bash
python script/plot_return_curve_blocking_area_pct.py --exp_path /path/to/experiment --confirm-scorer detected
```

Supports: GridpointIntensityScorer, GridpointPersistenceScorer, ANOScorer,
IntegratedScorer, DriftPenalizedScorer, RMSEScorer.
Features: NPZ DNS overlay, PDF sidebar (intensity scorer), .nc score fallback.

### Heatwave

```bash
python script/plot_return_curve_heatwave.py --exp_path /path/to/experiment --confirm-scorer detected
```

Supports: HeatwaveMeanScorer only.
Features: K->C conversion, .npy DNS overlay, no PDF panel.
Default DNS: `AI-RES/RES/experiments/saved_results/EXP15_AIRES_France/EXP15_AIRES_France_T.7_rv_control.npy`

## Related Docs

- `plot_return_curve_reference.md` (full CLI, workflow, troubleshooting)
- `../reference/scoring_reference.md`
- `../reference/aires_config.md`
- `../diagnostics/genealogical_collapse.md`
```

**Step 2: Commit**

```bash
git add .skills/plot/plot_return_curve.md
git commit -m "docs: rewrite skill doc as score-agnostic with unified entry point"
```

---

### Task 4: End-to-end verification

**Files:**
- None (verification only)

**Step 1: Run all return-curve tests**

Run: `cd /glade/u/home/zhil/project/AI-RES/Blocking && PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH" python -m pytest tests/script/test_plot_return_curve_unified.py tests/script/test_plot_return_curve_heatwave.py tests/src/test_return_curve.py -v`
Expected: All tests pass (6 unified + 6 heatwave + 13 shared = 25 total).

**Step 2: Verify unified script --help**

Run: `cd /glade/u/home/zhil/project/AI-RES/Blocking && python script/plot_return_curve.py --help`
Expected: Help text with all flags, exit 0.

**Step 3: Verify blocking script still works**

Run: `cd /glade/u/home/zhil/project/AI-RES/Blocking && PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH" python script/plot_return_curve_blocking_area_pct.py --help`
Expected: Help text, exit 0. No changes to blocking script behavior.

**Step 4: Verify heatwave script still works**

Run: `cd /glade/u/home/zhil/project/AI-RES/Blocking && python script/plot_return_curve_heatwave.py --help`
Expected: Help text, exit 0. No changes to heatwave script behavior.
