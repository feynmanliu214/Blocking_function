# Unified Score-Agnostic Return-Curve Dispatcher Design

**Date**: 2026-03-15
**Status**: Approved

## Goal

Create a unified entry-point script (`script/plot_return_curve.py`) that auto-detects
the scorer type from an experiment and dispatches to the correct plotting function.
Rewrite the skill doc to be score-agnostic and unified-first.

## Requirements (from interview)

| Decision | Choice |
|----------|--------|
| Entry point | Unified `script/plot_return_curve.py` |
| Existing scripts | Kept as direct-call alternatives (coexist) |
| DNS resolution | Auto-resolved by scorer type (no flags needed for defaults) |
| Skill doc | Unified-first, individual scripts in "Direct Script Access" section |

## Architecture

```
script/plot_return_curve.py              <- NEW unified entry point
  |-- Auto-detects scorer from experiment (via detect_scorer_from_experiment)
  |-- Resolves scorer-specific defaults (DNS paths, unit conversion, panel layout)
  |-- Dispatches to existing plotting functions:
  |     |-- plot_return_curve() from blocking script
  |     +-- plot_heatwave_return_curve() from heatwave script
  +-- CLI: minimal flags (--exp_path, --K, --confirm-scorer, --output_dir)

script/plot_return_curve_blocking_area_pct.py  <- KEPT (direct access)
script/plot_return_curve_heatwave.py           <- KEPT (direct access)

.skills/plot/plot_return_curve.md              <- REWRITTEN: unified-first
```

## Unified Script Behavior

1. Accept `--exp_path` (default: read `latest_exp_path.txt`)
2. Call `detect_scorer_from_experiment(exp_path)` to get scorer name
3. Based on scorer name, dispatch:
   - `HeatwaveMeanScorer` -> import and call `plot_heatwave_return_curve()` with
     auto-resolved defaults (DNS `.npy`, K->C, no PDF panel)
   - All blocking scorers (`GridpointIntensityScorer`, `GridpointPersistenceScorer`,
     `ANOScorer`, etc.) -> import and call `plot_return_curve()` with auto-resolved
     defaults (NPZ DNS, PDF overlay for intensity scorer)
4. Scorer-specific DNS defaults baked into a dispatch table
5. Pass-through flags for overrides: `--dns-data-path`, `--dns-ground-truth-window`,
   `--show-ground-truth`, `--pdf-overlay`, `--show`

## CLI Interface

```
python script/plot_return_curve.py
python script/plot_return_curve.py --exp_path /path/to/experiment
python script/plot_return_curve.py --exp_path /path/to/experiment --confirm-scorer detected
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--exp_path` | from `latest_exp_path.txt` | Experiment directory |
| `--K` | 7 | Final resampling step |
| `--confirm-scorer` | None | Skip interactive prompt (`detected` or scorer name) |
| `--output_dir` | `figures/` | Output directory |
| `--show-ground-truth` / `--no-show-ground-truth` | enabled | Toggle DNS overlay |
| `--dns-ground-truth-window` | `12-24_to_12-30` | DNS window (blocking only) |
| `--dns-data-path` | auto-resolved | Override DNS data path |
| `--pdf-overlay` / `--no-pdf-overlay` | enabled | PDF sidebar (blocking intensity only) |
| `--show` | false | Display figure interactively |
| `--season_label` | `DJF` | Season label (blocking only) |

## Scorer Dispatch Table

| Scorer | Plotting function | DNS format | DNS default | Unit conversion | PDF panel |
|--------|------------------|------------|-------------|-----------------|-----------|
| HeatwaveMeanScorer | `plot_heatwave_return_curve` | .npy (Weibull) | `AI-RES/.../EXP15_AIRES_France_T.7_rv_control.npy` | K -> C | No |
| GridpointIntensityScorer | `plot_return_curve` | .npz (block maxima) | `data/return_curve_ground_truth/` | None | Yes |
| GridpointPersistenceScorer | `plot_return_curve` | .npz | `data/return_curve_ground_truth/` | None | No |
| ANOScorer (+ aliases) | `plot_return_curve` | .npz | `data/return_curve_ground_truth/` | None | No |
| RMSEScorer | `plot_return_curve` | .npz | `data/return_curve_ground_truth/` | None | No |

## Skill Doc Structure

1. Quick Start -- unified script, "just works"
2. How It Works -- scorer auto-detection, default resolution
3. Common Flags -- unified script flags
4. Direct Script Access -- brief section on individual scripts
5. Related Docs

## What Changes

| Component | Change |
|-----------|--------|
| `script/plot_return_curve.py` | NEW unified dispatcher |
| `.skills/plot/plot_return_curve.md` | REWRITTEN unified-first |
| `script/plot_return_curve_blocking_area_pct.py` | No change |
| `script/plot_return_curve_heatwave.py` | No change |
| `src/return_curve.py` | No change |

## Testing

- Unit test: `tests/script/test_plot_return_curve.py`
  - `--help` flag parses without error
  - Scorer dispatch: HeatwaveMeanScorer routes to heatwave function
  - Scorer dispatch: GridpointIntensityScorer routes to blocking function
  - Default exp path resolution from `latest_exp_path.txt`
  - Missing experiment path errors cleanly
