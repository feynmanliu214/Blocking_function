# Plot Return Curve for AI-RES Output

Compact runbook for generating return-period curves with scorer-aware labels/titles.

Script:
`/glade/u/home/zhil/project/AI-RES/Blocking/script/plot_return_curve_blocking_area_pct.py`

Use this file for the fast path. Detailed CLI/reference/troubleshooting content lives in:
- `plot_return_curve_reference.md`

## Defaults That Matter

- Default score source is `plasim`:
  - first tries `resampling/plasim_scores_step_K.npy`
  - falls back to recomputing from final-step PlaSim `.nc` outputs
- `plasim_forecast` mode is available when `.nc` outputs were cleaned and per-particle forecast score arrays were saved.
- DNS ground-truth overlay is enabled by default (only for `GridpointIntensityScorer`).
- Default DNS overlay window is `12-24_to_12-30`.
  - Defaults to `return_curve_block_maxima_always_12-24_to_12-30_retry2.npz` (full)
    and `return_curve_block_maxima_always_12-24_to_12-30_retry2_subset400.npz`
    (subset); falls back to legacy `return_curve_block_maxima_always_12-24_to_12-30*.npz`
    files if retry2 files are missing.
  - Use `--dns-ground-truth-window djf` for full-DJF DNS curves.

## Quick Start

Activate the AI-RES environment first:

```bash
source /glade/u/apps/opt/miniforge/25.11/etc/profile.d/conda.sh
conda activate /glade/work/zhil/conda_envs/aires
```

Run the plot script:

```bash
# Default (PlaSim score-based y-axis)
python script/plot_return_curve_blocking_area_pct.py

# Specify experiment
python script/plot_return_curve_blocking_area_pct.py --exp_path /path/to/experiment

# Use saved forecast score arrays (when final .nc output is unavailable)
python script/plot_return_curve_blocking_area_pct.py \
  --exp_path /path/to/experiment \
  --score-source plasim_forecast

# Use full-DJF DNS overlay curves instead of the default 12/24-12/30 window
python script/plot_return_curve_blocking_area_pct.py --dns-ground-truth-window djf
```

## Required Inputs (Minimal Checklist)

1. Experiment directory with `working_tree.pkl`
2. One of the score-source inputs:
   - `plasim` preferred: `resampling/plasim_scores_step_{K}.npy`
   - `plasim` fallback: final-step `step_{K}/particle_{i}/output/*.nc`
   - `plasim_forecast`: `step_{K}/particle_{i}/forecast/*_A_<region>.npy`
3. Scorer metadata for detection:
   - `<exp_name>_used_config.json` (preferred)
   - `outerr_forecasts.log` (fallback)
4. DNS ground-truth NPZs (only if overlay is enabled and scorer is intensity):
   - `data/return_curve_ground_truth/return_curve_block_maxima_always_12-24_to_12-30_retry2.npz` (default full curve for `12-24_to_12-30`; falls back to `return_curve_block_maxima_always_12-24_to_12-30.npz`)
   - `data/return_curve_ground_truth/return_curve_block_maxima_always_12-24_to_12-30_retry2_subset400.npz` (default subset curve for `12-24_to_12-30`; falls back to legacy `return_curve_block_maxima_always_12-24_to_12-30_subset*.npz`)
   - `data/return_curve_ground_truth/return_curve_block_maxima.npz` (DJF full curve)
   - `data/return_curve_ground_truth/return_curve_block_maxima_subset400.npz` (DJF subset curve)

## How To Pick `K`

- For `plasim`:
  - use the highest `resampling/plasim_scores_step_*.npy`
  - in standard QDMC runs this is typically one step ahead of the last resampling step
- For `plasim_forecast`:
  - use the final step that has `step_K/particle_*/forecast/*_A_<region>.npy`

Useful checks:

```bash
ls /glade/derecho/scratch/zhil/PLASIM/RES/experiments/<exp_basename>/<exp_name>/resampling/plasim_scores_step_*.npy
ls /glade/derecho/scratch/zhil/PLASIM/RES/experiments/<exp_basename>/<exp_name>/step_*/particle_*/forecast/*_A_<region>.npy
```

## Most-Used Flags

- `--exp_path`: experiment directory
- `--K`: step index used for particle selection and score loading
- `--score-source {plasim,plasim_forecast}`
- `--confirm-scorer detected` (skip interactive scorer confirmation)
- `--show-ground-truth` / `--no-show-ground-truth`
- `--dns-ground-truth-window {djf,12-24_to_12-30}` (default: `12-24_to_12-30`)
- `--pdf-overlay` / `--no-pdf-overlay` (default: enabled; sideways baseline Z500 PDF, GridpointIntensityScorer only)
- `--pdf-file` (path to baseline PDF NetCDF; default: `outputs/plasim/sim52/z500_anomaly_pdf_northatlantic_djf_daily_weighted.nc`)
- `--season_label` (label text shown on x-axis/title)

For full CLI details:

```bash
python script/plot_return_curve_blocking_area_pct.py --help
```

## DNS Overlay Behavior (Important)

- The DNS window selector changes only which DNS NPZ overlay files are plotted.
- It does not change AI+RES x-axis conversion (`RP = 1 / p_window` for the fixed annual window) or `season_label`.
- DNS plotting uses precomputed plotting-ready block-maxima NPZs (`return_curve_block_maxima*.npz`), not `all_scores.npz`.
- For DNS curves with `method == block_maxima`, visualization is step-based (`ax.step(..., where="post")`) and the zero-valued branch is hidden to avoid misleading continuity.
- Zeros remain in the NPZ statistics and return-period arrays; they are hidden only in the visualization.

## Common Issues (Quick Triage)

- Missing `working_tree.pkl`: wrong or incomplete experiment path
- Missing `plasim_scores_step_K.npy`: use `.nc` fallback (default `plasim`) or switch to `plasim_forecast`
- Missing forecast `*_A_<region>.npy`: verify `region` and per-particle forecast outputs
- Non-interactive environment: pass `--confirm-scorer detected`
- DNS overlay warnings: check the selected DNS window files under `data/return_curve_ground_truth/`

See `plot_return_curve_reference.md` for detailed troubleshooting entries.

## Finding The Latest Experiment Path

Check:

```text
/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/experiments/logs/latest_exp_path.txt
```

Use `output_path` from that file as `--exp_path`.

## Heatwave Return Curve

Script:
`/glade/u/home/zhil/project/AI-RES/Blocking/script/plot_return_curve_heatwave.py`

Plots AI+RES scatter points overlaid on DNS ground-truth return curves
for `HeatwaveMeanScorer` experiments. Scores are plotted in °C (AI+RES
Kelvin scores are converted by subtracting 273.15). No PDF sidebar or
date histogram panel.

### Quick Start

```bash
source /glade/u/apps/opt/miniforge/25.11/etc/profile.d/conda.sh
conda activate /glade/work/zhil/conda_envs/aires

# Default (EXP15 France, K=7)
python script/plot_return_curve_heatwave.py

# Specific experiment, non-interactive
python script/plot_return_curve_heatwave.py \
  --exp_path /path/to/experiment \
  --confirm-scorer detected
```

### Required Inputs

1. Experiment directory with `working_tree.pkl`
2. Pre-computed scores: `resampling/plasim_scores_step_{K}.npy`
3. Scorer metadata: `<exp_name>_used_config.json`
4. DNS ground-truth `.npy` (optional, for overlay)

### Most-Used Flags

- `--exp_path`: experiment directory
- `--K`: step index (default: 7)
- `--dns-data-path`: path to DNS scores `.npy` file
- `--dns-subset-n`: number of years for subset curve (default: 400)
- `--confirm-scorer detected` (skip interactive prompt)
- `--show-ground-truth` / `--no-show-ground-truth`
- `--output_dir`: figure output directory

### Default DNS Data

The default DNS ground-truth file is:
`AI-RES/RES/experiments/saved_results/EXP15_AIRES_France/EXP15_AIRES_France_T.7_rv_control.npy`

This contains 50,000 years of spatiotemporal mean near-surface temperature
scores (in °C). The first 400 years are used for the subset curve.

### Unit Conversion

- DNS scores: already in °C (no conversion)
- AI+RES scores: stored in Kelvin, converted to °C by subtracting 273.15

### Scorer Enforcement

This script only accepts `HeatwaveMeanScorer`. The `--confirm-scorer` flag
is restricted to `detected` and `HeatwaveMeanScorer`. If the experiment's
detected scorer is not `HeatwaveMeanScorer`, the script exits with an error.

## Related Docs

- `plot_return_curve_reference.md` (full CLI, workflow, troubleshooting)
- `../reference/scoring_reference.md`
- `../reference/aires_config.md`
- `../diagnostics/genealogical_collapse.md`
