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
   - **HeatwaveMeanScorer** -> heatwave plotter (K->°C conversion, .npy DNS, no PDF panel)
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

### Blocking-Only Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dns-ground-truth-window` | `12-24_to_12-30` | DNS window (`djf` or `12-24_to_12-30`) |
| `--pdf-overlay` / `--no-pdf-overlay` | enabled | PDF sidebar (GridpointIntensityScorer only) |
| `--pdf-file` | built-in default | Override baseline PDF NetCDF path |
| `--season_label` | `DJF` | Season label on title |

### Heatwave-Only Flags

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
| HeatwaveMeanScorer | .npy (Weibull) | `AI-RES/.../EXP15_AIRES_France_T.7_rv_control.npy` | K -> °C | No |
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
Features: K->°C conversion, .npy DNS overlay, no PDF panel.
Default DNS: `AI-RES/RES/experiments/saved_results/EXP15_AIRES_France/EXP15_AIRES_France_T.7_rv_control.npy`

## Related Docs

- `plot_return_curve_reference.md` (full CLI, workflow, troubleshooting)
- `../reference/scoring_reference.md`
- `../reference/aires_config.md`
- `../diagnostics/genealogical_collapse.md`
