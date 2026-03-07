# AI-RES Config Reference

Detailed configuration reference for AI-RES experiments. Read this only when you need to look up specific fields.

---

## Experiment JSON Fields

Config location: `/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/experiments/configs/derecho/AIRES/`

### Core Parameters

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `experiment_name` | string | Base name for experiment | `"EXP15_AIRES"` |
| `region` | string | Target region | `"NorthAtlantic"`, `"France"`, `"PNW"`, `"Chicago"` |
| `N_particles` | int | Number of ensemble members | `200` |
| `dtau` | int | Days between resampling | `5` |
| `num_steps` | int | Number of resampling iterations | `6` |
| `target_duration` | int | Scoring window (days) | `5` |

### Resampling Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resampling_method` | string | - | `"pivotal"` or `"multinomial"` |
| `step_first_resampling` | int | - | Skip resampling before this step |
| `resampling_last_step` | bool | - | Resample on final step |
| `theta_type` | string | - | `"ensemble_mean"`, `"ensemble_median"`, `"log_MGF_{beta}"` |
| `splitting_constant` | float | `1.0` | Selection strength (typical: 2.0) |
| `alpha` | float | `0.0` | Lead time weight decay |
| `use_quantile` | bool | `false` | Quantile mapping vs z-score |
| `divide_by_parent_weights` | bool | `true` | Divide by parent weights |

### Scorer Configuration

```json
"scorer": {
    "name": "ScorerName",
    "params": {...}
}
```

| Scorer | Parameters | Description |
|--------|------------|-------------|
| `IntegratedScorer` | `n_days` (default: 5) | Time-integrated Z500 anomaly (B_int) |
| `DriftPenalizedScorer` | `gamma` (default: 5.0), `integration_days` (default: 5, use `null` for entire event) | P_block + P_drift score |
| `GridpointPersistenceScorer` | `n_days` (default: 5, min: 5), `min_persistence` (default: 5) | DG-style per-gridpoint blocking. Returns mean blocked percentage (0-100). A grid cell is "blocked" if above threshold for ≥`min_persistence` consecutive days. |
| `GridpointIntensityScorer` | `n_days` (default: 5, min: 5), `min_persistence` (default: 5), `fallback_to_nonblocked` (default: `false`; accepts `false`, `true`, or `"always"`) | Strict per-window intensity score. Window length is fixed to `min_persistence`; `running_mean_days` is not supported. |

### Forecast Parameters

| Field | Type | Description |
|-------|------|-------------|
| `forecast_method` | string | `"Pangu-Plasim"`, `"PFS"`, `"control"`, `"Persistence"` |
| `N_members_fcst` | int | Forecast ensemble size (typical: 100) |
| `PanguPlasim_perturb` | float | Ensemble perturbation (typical: 0.1) |

### Output Retention

| Field | Type | Description |
|-------|------|-------------|
| `delete_output` | `true` / `false` / `"partial"` | Cleanup mode. `true`: delete old step directories (aggressive). `false`: keep all outputs. `"partial"`: delete heavy files in `step_*/particle_*/output/` but keep restart lineage and `working_tree.pkl` for trajectory reconstruction. |
| `save_srv_output` | bool | Save raw `plasim_output` files (default: `true`) |

### Perturbation

| Field | Type | Description |
|-------|------|-------------|
| `EPS_perturb` | float | PlaSim restart perturbation (typical: 0.003) |
| `perturb_from_past` | bool | Perturb from past vs current state |
| `create_ICs_script` | string | IC script path (e.g., `"create_ICs/blocking_1_ICs.sh"`) |

---

## Available Regions

From `/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/regions.json`:

| Region | Lon Range | Lat Range |
|--------|-----------|-----------|
| `NorthAtlantic` | 300-357°E | 57-74°N |
| `France` | 0-6°E | 43-49°N |
| `PNW` | 236-242°E | 43-49°N |
| `Chicago` | 270-276°E | 38-43°N |

---

## Example Complete Config

```json
{
    "experiment_name": "EXP15_AIRES",
    "region": "NorthAtlantic",
    "N_particles": 200,
    "dtau": 5,
    "num_steps": 6,
    "target_duration": 5,
    "EPS_perturb": 0.003,
    "create_ICs_script": "create_ICs/blocking_1_ICs.sh",
    "perturb_from_past": false,
    "resampling_method": "pivotal",
    "use_quantile": false,
    "step_first_resampling": 4,
    "resampling_last_step": true,
    "theta_type": "ensemble_mean",
    "splitting_constant": 2.0,
    "alpha": 0.02,
    "divide_by_parent_weights": true,
    "forecast_method": "Pangu-Plasim",
    "N_members_fcst": 100,
    "PanguPlasim_perturb": 0.1,
    "PATH_CODE": "/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES",
    "PATH_SCRATCH": "/glade/derecho/scratch/zhil/PLASIM",
    "PATH_REFERENCE_RUN_DIR": "/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/reference_run_dir/derecho/run",
    "POSTPROC_SCRIPT": "/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/PLASIM/postprocessor2.0/burn7/derecho/submit_burn.sh",
    "DIR_PANGUPLASIM_MODEL": "/glade/work/alancelin/PanguWeather-ens/v2.0",
    "PATH_POSTPROC_NL_DIR": "/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/namelists_postproc",
    "PATH_YAML_CONFIG": "/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/forecast_modules/PanguPlasim/yaml_config/PANGU_PLASIM_H5_DERECHO_0515_enstest_amaury.yaml",
    "run_num": "0515",
    "job_manager": "PBS",
    "debug_mode": false,
    "delete_output": "partial",
    "clim_file": "/glade/u/home/zhil/project/AI-RES/Blocking/data/ano_climatology_thresholds.nc",
    "threshold_json_file": "/glade/u/home/zhil/project/AI-RES/Blocking/data/ano_thresholds.json",
    "scorer": {
        "name": "IntegratedScorer",
        "params": {"n_days": 5}
    }
}
```

**Alternative scorer configurations:**

```json
// DriftPenalizedScorer
"scorer": {
    "name": "DriftPenalizedScorer",
    "params": {"gamma": 5.0, "integration_days": 5}
}

// GridpointPersistenceScorer (DG-style per-gridpoint blocking)
"scorer": {
    "name": "GridpointPersistenceScorer",
    "params": {"n_days": 7, "min_persistence": 5}
}

// GridpointIntensityScorer (strict-window max intensity)
"scorer": {
    "name": "GridpointIntensityScorer",
    "params": {"n_days": 5, "min_persistence": 5, "fallback_to_nonblocked": false}
}
```

---

## Output Structure

```
/glade/derecho/scratch/zhil/PLASIM/RES/experiments/<exp_basename>/<exp_name>/
├── outerr_forecasts.log      # Pangu-Plasim logs
├── outerr_postproc.log       # PlaSim postprocess logs
├── resampling/
│   ├── thetas_step_*.npy     # Scores per step
│   ├── weights_step_*.npy    # Particle weights
│   └── clones_step_*.npy     # Clone indices
├── step_*/                   # Per-step outputs
└── <exp_name>_used_config.json
```

---

## Related Skills

- [Running experiments](../runbooks/aires_runbook.md)
- [PBS job management](pbs_job_management.md)
- [Scoring reference](scoring_reference.md)
