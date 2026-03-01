# PFS NUMA-Aware Parallelism Fix

**Date:** 2026-03-01
**Problem:** PBS job 5231305 killed by walltime (43206s > 43200s limit) during step-4 PFS forecast.
**Root cause:** `PFS_run.bash` hardcodes `max_parallel=1`, forcing all PlaSim ensemble members to run serially.

## Context

- **Pipeline:** `QDMC.sh` → `run_PFS()` → `PFS.py` → `PFS_run.bash` (per particle, serially)
- **Failed config:** N_particles≈400, N_members_fcst≈50, 3 PFS steps (4,5,6)
- **Total work:** 400 × 50 × 3 = 60,000 PlaSim runs, each ~11-12s
- **Resources:** 4 nodes × 64 CPUs, ppr=8 cores per PlaSim run
- **Scope:** Fix only `PFS_run.bash` (Level 2 parallelism within a single particle)

## Design

Replace the hardcoded `max_parallel=1` block (lines 57-67) with NUMA-aware auto-tuning:

### Algorithm (two-phase, due to script flow)

**Phase 1 (lines 57-67, before config parsing):**
1. Compute NUMA-optimal parallelism: `numa_domains_per_node × num_nodes_plasim`
2. Apply deployment safety cap (default 16) to the auto-computed default only
3. Apply env override `PFS_MAX_PARALLEL` — this **can exceed** deployment cap (explicit opt-in)
4. Validate all env inputs (positive integers), fallback to defaults on invalid input

**Phase 2 (after line 134, after `num_members` is parsed from config JSON):**
5. Apply workload cap (`max_parallel <= num_members`) to avoid empty batches
6. Finalize `num_parallel` capping and emit diagnostic log

**Override semantics:** `PFS_MAX_PARALLEL` is an explicit opt-in that bypasses the deployment safety cap. Users who set it are taking responsibility for host placement stability. The deployment cap (16) only gates the auto-computed default for first-time/unvalidated deployments.

### What stays unchanged

- Batch/wait loop (lines 270-282): `run_member $run &` + `wait` every `num_parallel` jobs
- `sleep 0.02` launch delay between members (prevents MPI race conditions)
- `run_member()` function and `mpiexec -n $ppr` invocation (no host-binding changes)
- All directory initialization and postprocessing logic

### Risk analysis

| Risk | Mitigation | Residual |
|------|-----------|----------|
| NUMA contention | Default 1 run/NUMA domain; deployment cap=16 | Low: 16 < 24 available slots on 4×64 |
| Memory pressure | 16 × ~2GB = ~32GB << 256GB/node | Negligible |
| MPI launch races | Existing `sleep 0.02` between launches | Low |
| Host placement | Deployment cap=16 limits risk | Medium: mpiexec may cluster on fewer nodes. Future fix: explicit host pinning via `ni` variable |
| Invalid env vars | Input validation with fallback to defaults | None |
| Regression | `PFS_MAX_PARALLEL=1` restores serial behavior | None |

### Expected performance (measured ~11-12s per member)

| Config | max_parallel | Batches/particle | Time/particle | Time/step (400p) | 3 steps |
|--------|-------------|-----------------|--------------|-------------------|---------|
| Before (serial) | 1 | 50 | ~600s | ~67h | ~200h |
| After (auto, 4 nodes) | 16 | 4 | ~48s | ~5.3h | ~16h |
| After (env=32) | 32 | 2 | ~24s | ~2.7h | ~8h |

With deployment cap=16: ~16h (still over 12h). User should set `PFS_MAX_PARALLEL=32` for the production run after pilot benchmark validates host placement stability.

### Required validation

A pilot benchmark (Task 3 in implementation plan) must pass before claiming the walltime issue is resolved:
- Small run (N_particles=10, N_members_fcst=50, same node allocation)
- Acceptance: no launch errors, all outputs present, observed batching at expected parallelism
- Extrapolated full-run time must fit within 12h

### Future work

- Explicit host pinning: use computed `ni` (node index) from `run_member()` in `mpiexec --host ${nodes[$ni]}` to ensure round-robin across nodes, enabling safe removal of deployment cap
- Level 1 parallelism: parallelize particle loop in `QDMC.sh:run_PFS()` for further speedup
