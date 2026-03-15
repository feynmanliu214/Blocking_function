# QDMC.sh Bug Fix + Refactor Design

**Date:** 2026-03-15
**Status:** Approved
**Triggered by:** Job 5385611.desched1 failure (CUDA_VISIBLE_DEVICES UUID mismatch across nodes)

## 1. Bug Fix: CUDA_VISIBLE_DEVICES UUID Propagation

**Problem:** The generated `run_torchrun_node_mpi.sh` launcher doesn't reset `CUDA_VISIBLE_DEVICES`. When `mpiexec` propagates the head node's environment, remote nodes receive GPU UUIDs that don't match their local hardware. PyTorch sees 0 GPUs on those nodes, causing `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!` and cascade failure (rc=143 after 3 retries).

**Fix:** Add `unset CUDA_VISIBLE_DEVICES` at the top of both generated launcher scripts (`run_torchrun_node.sh` and `run_torchrun_node_mpi.sh`). Each node's CUDA runtime auto-discovers its local GPUs. `torchrun --nproc_per_node=$NUM_GPUS` handles per-process assignment.

## 2. File Split: New Module Structure

Current: 1 monolithic file (1,578 lines). New layout:

```
RES/
├── QDMC.sh                    # Main loop only (~120 lines)
│                               # Sources all lib/ files, parses config, runs init + main loop
├── lib/
│   ├── cluster_setup.sh       # PBS/SLURM detection, node counting, GPU detection,
│   │                          # conda activation, module loads (~100 lines)
│   ├── lib_utils.sh           # Shared retry/backoff helpers, validation,
│   │                          # SSH options builder, wait_for_pids, path helpers,
│   │                          # create_step_dir, cleanup, write_used_config (~250 lines)
│   ├── lib_plasim.sh          # run_one_particle_one_step, run_particles_one_step,
│   │                          # ensure_run_dir, pumaburner namelist (~200 lines)
│   └── lib_forecast.sh        # run_forecasts dispatch, run_PanguPlasim,
│                               # run_PFS, postprocess_step_to_panguplasim_inputs,
│                               # launcher script generation, SSH preflight (~400 lines)
└── ...
```

- `QDMC.sh` stays at the same path (no breaking change for configs/PBS scripts)
- `lib/` subdirectory keeps grouping clean

## 3. Duplication Consolidation

### 3a. Retry Infrastructure

- `validate_retry_params()` — single function replacing ~60 lines of duplicated integer validation across 4 locations
- `compute_retry_backoff_delay()` — already exists, stays as shared delay math
- Attempt-loop skeleton centralized as an internal engine (attempt counting + delay math)
- **Domain-specific wrappers remain first-class** — they own preflight checks, per-attempt state repair (e.g., PlaSim's `plasim_restart`/`plasim_output` cleanup), and retryability decisions. Only the attempt counting + delay math is delegated to the shared engine
- SLURM retry path updated to use the same backoff pattern for consistency (low-risk cleanup, not a semantic rewrite)

### 3b. SSH Options

A `build_ssh_opts()` helper function that takes identity file as input and returns the options array. No mutable global.

### 3c. Batch-Wait Pattern

No generic `run_batch_parallel` abstraction. `wait_for_pids_or_fail()` stays as the shared primitive. A small `flush_batch()` helper added if warranted after other refactoring, but `run_particles_one_step()` and `run_PFS()` keep their own loop structures given their different setup/log-merging behavior.

## 4. Launcher Script Hardening

- **Unset `CUDA_VISIBLE_DEVICES`** in both launcher scripts — the immediate fix
- **Activate conda environment inside the launcher** rather than baking in `$PATH`/`$LD_LIBRARY_PATH` — makes the launcher self-contained and correct on any node
- **Move launcher generation** into `lib_forecast.sh` as a `generate_launcher_scripts()` function
- The "generate once, reuse across steps" caching pattern is preserved

## 5. What Gets Removed / Kept

### Removed
- `run_particles_one_step_doing_nothing()` — the fake-run debug execution path
- The conditional branches in the main loop that route to this function
- Redundant `echo "DIR_PANGUPLASIM_MODEL = ..."` outside the Pangu-Plasim block

### Kept (all stable functionality)
- All forecast modes: Pangu-Plasim, PFS, control, control_random, Persistence
- `diverse_initial_conditions` feature
- All scorer validation logic
- Cleanup modes (`delete_output`: true/partial/false)
- Both PBS and SLURM paths
- `initialize_experiment()` and IC generation
- Step 0 special-casing and main loop structure
- `debug_mode` as a config field (preserved for future use)

### Treated as refactorable (not sacred)
- The mpiexec multi-node launch path — being fixed and hardened per Section 4

## 6. Verification Strategy

1. **Mapping check** — every original responsibility is accounted for in the new layout, with only the approved removals and fixes changed. Produce a mapping table (original function -> new location or "removed per design")
2. **Shellcheck** — run `shellcheck` on all new files
3. **Smoke test (multi-node)** — submit `EXP15_heatwave_AIRES_smoke.json` on Derecho using the real 4-node GPU path (not a 1-node dev submission)
4. **Output-level verification** after the smoke run:
   - Run reaches step 4 Pangu forecasting
   - No `Traceback` or `ERROR` in experiment logs
   - `World size from Cuda: 4` on all node logs (all 4 nodes report local GPU UUIDs)
   - Forecast outputs exist for all expected particles
   - `validate_panguplasim_forecast_scores` does not fail
5. **Optional regression check** — if the refactor touches shared dispatch or main-loop code heavily, run one cheap non-Pangu sanity test (control or PFS) to confirm those paths still work
