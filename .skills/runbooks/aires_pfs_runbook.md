# AI-RES with PFS Forecast Method Runbook

Running AI-RES (QDMC) experiments using PlaSim-only PFS forecasts on Derecho.

---

## Quick Reference

**Submit:**
```bash
cd /glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES
qsub experiments/submit_pilot_PFS_parallel.pbs
```

**Monitor:** `qstat -u $USER`

**Output:** `/glade/derecho/scratch/zhil/PLASIM/RES/experiments/<experiment_name>_<scorer_name>_<region>/`

**Logs:** `AI-RES/RES/experiments/logs/<jobid>.desched1.OU`

---

## How PFS Differs from AIRES (Pangu-Plasim)

| Aspect | AIRES | PFS |
|--------|-------|-----|
| Forecast engine | Pangu-Plasim (GPU AI emulator) | PlaSim-only ensemble |
| Nodes | GPU + CPU nodes | CPU-only nodes |
| PBS resource | `select=4:ncpus=64:ngpus=4` | `select=4:ncpus=64` |
| Forecast module | `PanguPlasimFS.py` | `PFS_run.bash` + `PFS.py` |
| Per-particle forecast | Single AI inference | `N_members_fcst` PlaSim MPI runs |
| Scoring | AI forecast scores | PlaSim ensemble member scores |
| Key config field | — | `N_members_fcst` (members per particle) |

---

## Config File

Location: `AI-RES/RES/experiments/configs/derecho/PFS/`

**PFS-specific fields** (beyond standard QDMC config):

| Field | Description | Example |
|-------|-------------|---------|
| `forecast_method` | Must be `"PFS"` | `"PFS"` |
| `N_members_fcst` | PlaSim ensemble members per particle per PFS step | `50` |
| `scorer` | Scoring method (mandatory) | `{"name": "GridpointIntensityScorer", "variable": "z500", "params": {...}}` |
| `scorer.variable` | Climate variable the scorer operates on (required) | `"z500"` or `"tas"` |
| `POSTPROC_SCRIPT` | Postprocessor path (burn7 or postprocessor2.0) | see below |
| `PATH_POSTPROC_NL` | burn7 namelist path | see below |

**Example config** (`EXP15_PFS_pilot10.json`):
```json
{
    "experiment_name": "EXP15_PFS_pilot10",
    "region": "France",
    "N_particles": 10,
    "dtau": 5,
    "num_steps": 6,
    "target_duration": 7,
    "step_first_resampling": 4,
    "resampling_last_step": true,
    "forecast_method": "PFS",
    "N_members_fcst": 50,
    "scorer": {
        "name": "GridpointIntensityScorer",
        "variable": "z500",
        "params": {"n_days": 7, "min_persistence": 5, "fallback_to_nonblocked": "always"}
    },
    "splitting_constant": 2.0,
    "PATH_CODE": "/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES",
    "PATH_SCRATCH": "/glade/derecho/scratch/zhil/PLASIM",
    "PATH_REFERENCE_RUN_DIR": ".../reference_run_dir/derecho/run",
    "PATH_POSTPROC_NL": "/glade/work/alancelin/AI-RES/RES/namelists_postproc/lightweight_A.nl",
    "POSTPROC_SCRIPT": "/glade/work/alancelin/AI-RES/PLASIM/postprocessor2.0/burn7/derecho/submit_burn.sh",
    "job_manager": "PBS"
}
```

---

## PBS Submission Script

Location: `AI-RES/RES/experiments/submit_pilot_PFS_parallel.pbs`

Key PBS directives:
```bash
#PBS -A UCHI0014
#PBS -q main
#PBS -l select=4:ncpus=64      # 4 CPU-only nodes, 256 cores total
#PBS -l walltime=02:00:00       # 2h for pilot (10 particles)
```

---

## Parallelism: NUMA-Aware Auto-Tuning

`PFS_run.bash` automatically computes parallelism from hardware topology:

- **Derecho**: 2x AMD EPYC 7543 per node = 8 NUMA domains x 8 cores
- Each PlaSim run uses `ppr=8` MPI processes = 1 NUMA domain
- Default: 8 runs/node x 4 nodes = 32 concurrent PlaSim runs
- Deployment cap: 32 (safe default with host/CPU binding)

**MPI binding**: Each run is pinned to a specific node and core range:
```bash
mpiexec -host ${nodes[$ni]} -n $ppr --cpu-bind list:$sc-$ec ./most_plasim_t42_l10_p8.x
```

**Environment overrides:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PFS_MAX_PARALLEL` | auto (32) | Hard cap on concurrent PlaSim runs; bypasses deployment cap |
| `PFS_NUMA_DOMAINS` | `8` | NUMA domains per node |
| `PFS_LAUNCH_STAGGER` | `1` | Seconds between MPI launches |

---

## Timing Estimates (4 nodes x 64 CPUs)

| N_particles | N_members_fcst | PFS steps | Approx. walltime |
|-------------|----------------|-----------|-------------------|
| 10 | 50 | 3 (steps 4-6) | ~48 min |
| 400 | 50 | 3 | ~6-8 h (est.) |

Scale: each PFS step runs `N_particles x N_members_fcst` PlaSim members, batched 32 at a time.

---

## Verify Success

```bash
# 1. Check exit status
qstat -xf <JOBID> | grep Exit_status    # expect 0

# 2. Check all steps completed
tail -5 experiments/logs/<JOBID>.desched1.OU
# expect: "RES experiment finished"

# 3. Check resampling outputs exist
ls $PATH_EXP/resampling/thetas_step_*.npy

# 4. Check no empty scores (the old bug)
grep "WARNING.*0/.*paths exist" $PATH_EXP/outerr_forecasts.log
# expect: no output

# 5. Verify thetas are non-zero from step_first_resampling onward
python3 -c "
import numpy as np
for s in range(4, 7):
    t = np.load('$PATH_EXP/resampling/thetas_step_{}.npy'.format(s))
    print(f'step {s}: theta range [{t.min():.2e}, {t.max():.2e}]')
"
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Walltime exceeded during PFS | `max_parallel=1` (serial) | Ensure NUMA-aware code is in `PFS_run.bash` |
| PlaSim runs take hours instead of seconds | Missing MPI host/CPU binding | Add `-host ${nodes[$ni]} --cpu-bind list:$sc-$ec` to mpiexec |
| All PFS scores empty (`shape=(0,)`) | Output filename mismatch | burn7 output must use `_postproc.nc` suffix |
| burn7 segfault messages | burn7 cleanup bug (pre-existing) | **Not blocking** — output files are valid |
| `Resource temporarily unavailable` on mpiexec | PBS RPC race under high concurrency | Increase `PFS_LAUNCH_STAGGER` or reduce `PFS_MAX_PARALLEL` |

---

## Key Files

| File | Role |
|------|------|
| `RES/QDMC.sh` | Main driver; calls `run_PFS()` at forecast steps |
| `RES/forecast_modules/PFS/PFS_run.bash` | Launches parallel PlaSim ensemble + postprocessing |
| `RES/forecast_modules/PFS/PFS.py` | Scores PFS ensemble members, returns per-particle scores |
| `RES/resampling/resampling.py` | Resamples particles based on PFS scores |
| `RES/experiments/configs/derecho/PFS/` | PFS experiment config JSONs |
| `RES/experiments/submit_pilot_PFS_parallel.pbs` | PBS submission script |

---

## Related Skills

- [AI-RES runbook (AIRES method)](aires_runbook.md)
- [PFS standalone ensemble](pfs_ensemble_instructions.md)
- [PBS job management](../reference/pbs_job_management.md)
