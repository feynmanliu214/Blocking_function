# AI-RES Experiment Runbook

Instructions for running and configuring AI-RES experiments on Derecho.

---

## Quick Reference

**Submit:**
```bash
cd /glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES
python3 experiments/submit_aires.py --exp-name EXP15_AIRES
```
`submit_aires.py` chooses resources from `forecast_method` in the config:
- `PFS` (and other non-Pangu methods): CPU script (`submit_job_derecho_AIRES.pbs`)
- `Pangu-Plasim`: 4-node GPU script (`submit_job_derecho_AIRES_gpu.pbs`)

**Monitor:** `qstat -u $USER`

**Output:** `/glade/derecho/scratch/zhil/PLASIM/RES/experiments/<experiment_name>_<scorer_name>_<region>/`

**PlaSim netCDF outputs:**
- Postprocessed PlaSim netCDF files (`plasim_out.step_<k>.particle_<i>.nc`) are written at every step (all resampling steps, not just the last).
- Raw `plasim_output` files are only copied when `save_srv_output=true` (default).
- If `delete_output="partial"`, heavy files in `step_*/particle_*/output/` are pruned while restart lineage is preserved for later trajectory reconstruction.

**Lineage extraction helper (for option 2):**
```bash
python3 /glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/resampling/extract_extreme_lineage.py \
  --path_exp /glade/derecho/scratch/zhil/PLASIM/RES/experiments/<exp_basename>/<exp_name>
```

---

## Example Human Prompts

Always specify a base config and parameters to change.

### Example 1: Change hyperparameters
```
Based on EXP15_AIRES, run an experiment with:
- N_particles: 400
- use_quantile: true
- scorer: DriftPenalizedScorer with variable=z500, gamma=0, integration_days=7
```

### Example 2: Target different blocking event
```
Based on EXP15_AIRES, target blocking event on 0050-01-15.
Keep all other parameters the same.
```

### Example 3: Multiple changes
```
Based on EXP15_AIRES:
- Target event: 0055-02-20
- N_particles: 300
- splitting_constant: 2.5
- scorer: IntegratedScorer with variable=z500, n_days=7
- Region: France
```

### Example 4: Heatwave experiment
```
Based on EXP15_AIRES, run a heatwave experiment with:
- scorer: HeatwaveMeanScorer with variable=tas, n_days=7
- region: Chicago
```

### If parameters are unclear, ask:
- Missing target date → "What blocking event date? Default is 0047-12-05"
- Missing base config → "Which experiment config as base?"
- Ambiguous scorer → "What gamma value for DriftPenalizedScorer?"
- Missing variable → "What variable? z500 for blocking, tas for heatwave"

---

## Modifying Experiments

### Change hyperparameters

Edit the experiment JSON in:
```
/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/experiments/configs/derecho/AIRES/
```

**Commonly modified fields:**

| Field | Description | Example |
|-------|-------------|---------|
| `N_particles` | Number of ensemble members | `200`, `400` |
| `use_quantile` | Quantile mapping vs z-score | `true`, `false` |
| `splitting_constant` | Selection strength (1.0-3.0) | `2.0` |
| `region` | Target region | `"NorthAtlantic"`, `"France"`, `"PNW"`, `"Chicago"` |
| `scorer` | Scoring method (see below) | `{"name": "IntegratedScorer", "variable": "z500", "params": {"n_days": 5}}` |
| `scorer.variable` | Climate variable the scorer operates on (required) | `"z500"` or `"tas"` |

### Change scorer

The `scorer` block is mandatory — omitting it is a hard error.

```json
"scorer": {"name": "IntegratedScorer", "variable": "z500", "params": {"n_days": 5}}
"scorer": {"name": "DriftPenalizedScorer", "variable": "z500", "params": {"gamma": 5.0, "integration_days": 5}}
"scorer": {"name": "DriftPenalizedScorer", "variable": "z500", "params": {"gamma": 5.0, "integration_days": null}}  // entire event
"scorer": {"name": "GridpointIntensityScorer", "variable": "z500", "params": {"n_days": 5, "min_persistence": 5, "fallback_to_nonblocked": false}}
"scorer": {"name": "HeatwaveMeanScorer", "variable": "tas", "params": {"n_days": 7}}
```

**DriftPenalizedScorer parameters:**
- `gamma`: Drift penalty constant (default: 5.0)
- `integration_days`: Number of days to integrate from event start (default: 5). Use `null` to integrate entire event.

**GridpointIntensityScorer parameters:**
- `n_days`: Scoring window length in days (must be >= `min_persistence`).
- `min_persistence`: Strict exceedance window length (default: 5; must be >= 5).
- `fallback_to_nonblocked`: Controls blocking condition. `false` (default) restricts to blocked points. `true` falls back to regional max when no blocked points. `"always"` ignores blocking entirely and always uses regional max (fastest, skips all blocking detection).
- `running_mean_days` is removed and should not be set.

### Change target blocking event

Edit `EVENT_DATE` on line 34 of:
```
/glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES/create_ICs/blocking_1_ICs.sh
```

```bash
EVENT_DATE="YYYY-MM-DD"  # Valid range: 0006-08-26 to 0116-03-02
```

**How it works:**
- `EVENT_DATE` = target blocking event (onset/peak date)
- Init date = `EVENT_DATE - (num_steps × dtau)` days
- Example: EXP15 has `num_steps=6`, `dtau=5` → 30 days before event

**Verify restart file exists:**
```bash
ls /glade/u/home/zhil/project/AI-RES/Blocking/data/PlaSim/sim52/restart_files/MOST.<init_date>_00:00:00
```

---

## Running an Experiment

1. **Modify config** (if needed) - see above
2. **Modify EVENT_DATE** (if targeting different event)
3. **Submit:**
   ```bash
   cd /glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES
   python3 experiments/submit_aires.py --exp-name <your_exp_name>
   ```

The script auto-increments if experiment name exists.

## Prevent Duplicate Submissions

Duplicate submissions usually happen when automation treats a transient PBS state as a failed submit:
- `qsub` succeeds and returns a job id, but an immediate `qstat -f <job_id>` can briefly return "Unknown Job Id".
- Agents then submit again, creating another `exp_AIRES` job.
- The selected submit script (`submit_job_derecho_AIRES.pbs` for CPU or `submit_job_derecho_AIRES_gpu.pbs` for GPU) auto-increments experiment index, so duplicates become separate experiment folders.

Use this idempotent workflow:

1. Submit once and capture `JOB_ID`:
   ```bash
   cd /glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES
   JOB_ID=$(python3 experiments/submit_aires.py --exp-name <your_exp_name>)
   echo "$JOB_ID"
   ```
2. Do **not** resubmit if `qstat -f "$JOB_ID"` fails immediately. Poll for up to 2 minutes:
   ```bash
   for _ in {1..8}; do
     qstat -f "$JOB_ID" && break
     sleep 15
   done
   ```
3. Before any resubmit attempt, check for active jobs (`Q/R/E/H`) with the same `EXP_NAME`:
   ```bash
   EXP_NAME=<your_exp_name>
   for jid in $(qstat -x -u "$USER" -w | awk 'NR>5 && $4=="exp_AIRES" && $10 ~ /Q|R|E|H/ {print $1}'); do
     if qstat -f "$jid" 2>/dev/null | grep -q "EXP_NAME=${EXP_NAME}"; then
       state=$(qstat -f "$jid" 2>/dev/null | awk -F' = ' '/job_state/{print $2; exit}')
       echo "Active job for ${EXP_NAME}: ${jid} (state=${state})"
     fi
   done
   ```
4. Resubmit **only if no active job exists** for that `EXP_NAME`.

Notes:
- `experiments/logs/latest_exp_path.txt` is useful after a job starts, but not a reliable pre-submit duplicate guard.
- After `qdel`, wait until the job disappears from `qstat` before submitting again.

---

## Debugging

### Check for errors
```bash
grep -i "traceback\|error\|exception" outerr_forecasts.log
```

### Common failures

| Error | Fix |
|-------|-----|
| `PermissionError` | Update YAML `save_dir` to your scratch |
| `KeyError: 'clim_file'` | Add `clim_file` and `threshold_json_file` to JSON |
| `Restart file not found` | Check EVENT_DATE is in valid range (0006-08-26 to 0116-03-02) |
| `FileNotFoundError` for checkpoint | Verify `DIR_PANGUPLASIM_MODEL` and `run_num` |
| `DistStoreError: Timed out waiting for clients` | See "Pangu-Plasim GPU Distribution Fix" below |
| `Couldn't send RPC launch ... Resource temporarily unavailable` | Transient node launch failure (PBS/RPC). `QDMC.sh` auto-retries; if retries are exhausted, resubmit from scratch. See `../reference/pbs_job_management.md`. |

### Pangu-Plasim GPU Distribution Fix

**Symptom:** Job finishes early with errors like:
```
torch.distributed.DistStoreError: Timed out after 901 seconds waiting for clients. 1/4 clients joined.
FileNotFoundError: .../forecast/PanguPlasimFS_s4_p0_A_NorthAtlantic.npy
IndexError: tuple index out of range (in resampling.py)
```

**Root cause:** Multi-node PyTorch distributed via SSH fails due to inter-node NCCL communication issues.

**Solution (implemented 2026-02-06):** Parallel independent single-node inference across all GPU nodes.

Instead of one multi-node `torchrun` (which requires cross-node NCCL), `run_PanguPlasim()` now:
1. Splits the `N_particles` evenly across all allocated GPU nodes
2. Writes a per-node config JSON (with `particle_start_index` for correct file naming)
3. Launches independent `torchrun --standalone --nproc_per_node=4` on each node via SSH
4. Waits for all nodes to finish, then merges per-node logs into `outerr_forecasts.log`

```bash
# Per-node launcher (environment forwarded from head node)
ssh ${nodes[$ni]} "bash $LAUNCHER_SCRIPT \
    $PATH_CONFIG_NODE $DIR_PANGUPLASIM_MODEL \
    $DIR_PANGUPLASIMFS/PanguPlasimFS.py $NUM_GPUS_PER_NODE $NODE_LOG" &
```

**Why this works (vs. the old multi-node approach):**
- Each node is fully self-contained: `torchrun --standalone` uses only local GPU-GPU communication
- SSH is fire-and-forget (process launch only), not used for ongoing NCCL coordination
- Uses all 16 GPUs across 4 nodes (~4x faster than single-node)
- `particle_start_index` in config ensures output files use global particle indices (matching what `resampling.py` expects)
- PlaSim CPU runs still use all 4 nodes

**Key files:**
- `QDMC.sh`: `run_PanguPlasim()` — particle splitting and SSH launch logic
- `PanguPlasimFS.py`: reads `particle_start_index` from config to offset `_p{i}` in output filenames

---

## Related Skills

- [Plot return curves](plot_return_curve.md) - Generate return period plots from experiment output
- [Config reference](../reference/aires_config.md) - All config parameters
- [PBS job management](../reference/pbs_job_management.md) - Queue limits, canceling jobs, walltime issues
- [Genealogical collapse diagnostics](../diagnostics/genealogical_collapse.md) - Diagnose algorithm issues
