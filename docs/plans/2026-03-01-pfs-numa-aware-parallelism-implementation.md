# PFS NUMA-Aware Parallelism Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `max_parallel=1` in `PFS_run.bash` with NUMA-aware auto-tuned parallelism so PFS experiments with N_particles~400 and N_members_fcst~50 complete within a 12-hour PBS walltime.

**Architecture:** The parallelism logic is split into two phases due to script flow:
- **Phase 1 (lines 57-67, before config parsing):** Compute NUMA-aware `max_parallel` from hardware topology + env overrides + deployment safety cap. This replaces the old `max_parallel=1`.
- **Phase 2 (after line 109, after config parsing):** Apply workload cap (`max_parallel <= num_members`) and emit final diagnostic log.

The env override `PFS_MAX_PARALLEL` intentionally **can exceed** the deployment cap — it is an explicit opt-in for users who have validated host placement stability. The deployment cap only gates the auto-computed default.

**Tech Stack:** Bash (PFS_run.bash), PBS/SLURM job scheduler

**Design doc:** `docs/plans/2026-03-01-pfs-numa-aware-parallelism-design.md`

---

### Task 1: Replace max_parallel block in PFS_run.bash (Phase 1)

**Files:**
- Modify: `AI-RES/RES/forecast_modules/PFS/PFS_run.bash:57-67`

**Step 1: Replace the hardcoded max_parallel=1 block**

Replace lines 57-67 (from `# Cap parallel runs` through `echo "Then number of parallel runs is set to: $num_parallel"`) with:

```bash
# --- NUMA-aware parallelism cap (Phase 1: hardware + env) ---
# Derecho: 2x AMD EPYC 7543 = 8 NUMA domains x 8 physical cores per domain
# Each PlaSim run uses ppr=8 MPI processes = fits exactly 1 NUMA domain
# Optimal: one PlaSim run per NUMA domain per node
# Override via env: PFS_NUMA_DOMAINS (domains/node), PFS_MAX_PARALLEL (hard cap)
# NOTE: PFS_MAX_PARALLEL can exceed deployment cap (explicit opt-in for validated setups)
# Emergency fallback: PFS_MAX_PARALLEL=1 restores serial behavior

# Validate PFS_NUMA_DOMAINS (must be positive integer, default 8)
if [[ -n "$PFS_NUMA_DOMAINS" ]] && [[ "$PFS_NUMA_DOMAINS" =~ ^[1-9][0-9]*$ ]]; then
    numa_domains_per_node=$PFS_NUMA_DOMAINS
else
    [[ -n "$PFS_NUMA_DOMAINS" ]] && echo "WARNING: invalid PFS_NUMA_DOMAINS='$PFS_NUMA_DOMAINS', using default 8"
    numa_domains_per_node=8
fi

# Auto-compute from NUMA topology
max_parallel_auto=$((numa_domains_per_node * num_nodes_plasim))

# Deployment safety cap: without explicit host-pinning, mpiexec may cluster
# runs on fewer nodes. Cap the auto-computed default conservatively.
# This cap does NOT apply to explicit PFS_MAX_PARALLEL overrides.
deployment_cap=16
if [ "$max_parallel_auto" -gt "$deployment_cap" ]; then
    echo "Auto-computed parallelism $max_parallel_auto exceeds deployment cap $deployment_cap; capping auto default"
    max_parallel_auto=$deployment_cap
fi

# Validate and apply env override (PFS_MAX_PARALLEL)
# Explicit override CAN exceed deployment cap — user takes responsibility for host placement
if [[ -n "$PFS_MAX_PARALLEL" ]] && [[ "$PFS_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    max_parallel=$PFS_MAX_PARALLEL
    echo "Using explicit PFS_MAX_PARALLEL=$max_parallel (bypasses deployment cap)"
else
    [[ -n "$PFS_MAX_PARALLEL" ]] && echo "WARNING: invalid PFS_MAX_PARALLEL='$PFS_MAX_PARALLEL', using auto-computed value"
    max_parallel=$max_parallel_auto
fi

echo "NUMA topology: $numa_domains_per_node domains/node x $num_nodes_plasim node(s) = $((numa_domains_per_node * num_nodes_plasim)) (raw auto)"
echo "Deployment cap: $deployment_cap | Effective max_parallel (pre-workload-cap): $max_parallel"
echo "(env PFS_MAX_PARALLEL=${PFS_MAX_PARALLEL:-<unset>}, PFS_NUMA_DOMAINS=${PFS_NUMA_DOMAINS:-<unset>})"
echo "########################################################"
```

Note: The `num_parallel` capping and final echo are deferred to Phase 2 (Task 2).

**Step 2: Verify the script parses correctly**

Run: `bash -n AI-RES/RES/forecast_modules/PFS/PFS_run.bash`
Expected: No output (syntax OK)

**Step 3: Commit**

```bash
git add AI-RES/RES/forecast_modules/PFS/PFS_run.bash
git commit -m "perf(PFS): Phase 1 - NUMA-aware max_parallel replaces hardcoded 1

Auto-compute max_parallel from NUMA topology (8 domains/node x num_nodes).
Apply deployment safety cap (16) to auto default only.
Env override PFS_MAX_PARALLEL can exceed cap (explicit opt-in).
Input validation for PFS_NUMA_DOMAINS and PFS_MAX_PARALLEL.

Phase 2 (workload cap + num_parallel capping) follows after config parsing."
```

---

### Task 2: Add workload cap after config parsing (Phase 2)

**Files:**
- Modify: `AI-RES/RES/forecast_modules/PFS/PFS_run.bash` — insert between the `echo "N_RUN_DAYS = $N_RUN_DAYS"` line and the `if [ -z "$basename_output" ]` fallback block

**Step 1: Insert workload cap and num_parallel finalization**

Find the anchor line `echo "N_RUN_DAYS = $N_RUN_DAYS"` and insert immediately after it (before the `basename_output` fallback check):

```bash
# --- NUMA-aware parallelism cap (Phase 2: workload cap, after config parsing) ---
# Now that num_members is known, cap max_parallel to avoid empty wait cycles
if [ "$max_parallel" -gt "$num_members" ]; then
    echo "Capping max_parallel from $max_parallel to $num_members (workload cap: num_members=$num_members)"
    max_parallel=$num_members
fi
if [ "$num_parallel" -gt "$max_parallel" ]; then
    echo "Capping parallel runs from $num_parallel to $max_parallel (NUMA-aware)"
    num_parallel=$max_parallel
fi
echo "Final number of parallel runs: $num_parallel (max_parallel=$max_parallel, num_members=$num_members)"
```

**Step 2: Verify the script parses correctly**

Run: `bash -n AI-RES/RES/forecast_modules/PFS/PFS_run.bash`
Expected: No output (syntax OK)

**Step 3: Smoke-test the full parallelism computation logic in isolation**

Run:
```bash
# Case 1: 4 nodes, 64 CPUs, no env override → expect 16
num_nodes_plasim=4; num_members=50; num_parallel=24
PFS_NUMA_DOMAINS=""; PFS_MAX_PARALLEL=""
numa_domains_per_node=8
max_parallel_auto=$((8 * 4))  # 32
deployment_cap=16
[[ $max_parallel_auto -gt $deployment_cap ]] && max_parallel_auto=$deployment_cap  # 16
max_parallel=$max_parallel_auto
[[ $max_parallel -gt $num_members ]] && max_parallel=$num_members  # 16 < 50, no change
[[ $num_parallel -gt $max_parallel ]] && num_parallel=$max_parallel  # 24 > 16, cap to 16
echo "Case 1: Expected num_parallel=16, got: $num_parallel"
```
Expected: `Case 1: Expected num_parallel=16, got: 16`

Run:
```bash
# Case 2: env override PFS_MAX_PARALLEL=32 (exceeds deployment cap) → expect 32
num_nodes_plasim=4; num_members=50; num_parallel=24
PFS_MAX_PARALLEL=32
max_parallel=$PFS_MAX_PARALLEL
[[ $max_parallel -gt $num_members ]] && max_parallel=$num_members  # 32 < 50, no change
[[ $num_parallel -gt $max_parallel ]] && num_parallel=$max_parallel  # 24 < 32, no cap
echo "Case 2: Expected num_parallel=24, got: $num_parallel"
```
Expected: `Case 2: Expected num_parallel=24, got: 24`

Run:
```bash
# Case 3: num_members=10, small workload → expect capped to 10
num_nodes_plasim=4; num_members=10; num_parallel=24
PFS_MAX_PARALLEL=""
max_parallel=16  # from auto
[[ $max_parallel -gt $num_members ]] && max_parallel=$num_members  # 16 > 10, cap to 10
[[ $num_parallel -gt $max_parallel ]] && num_parallel=$max_parallel  # 24 > 10, cap to 10
echo "Case 3: Expected num_parallel=10, got: $num_parallel"
```
Expected: `Case 3: Expected num_parallel=10, got: 10`

Run:
```bash
# Case 4: invalid env → expect fallback to auto
PFS_MAX_PARALLEL="abc"
[[ "$PFS_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] && echo "valid" || echo "Case 4: invalid, fallback to auto"
```
Expected: `Case 4: invalid, fallback to auto`

**Step 4: Commit**

```bash
git add AI-RES/RES/forecast_modules/PFS/PFS_run.bash
git commit -m "perf(PFS): Phase 2 - workload cap after config parsing

Apply max_parallel <= num_members cap after num_members is parsed from
config JSON. Finalize num_parallel capping and emit diagnostic log.

Completes the two-phase NUMA-aware parallelism logic."
```

---

### Task 3: Pilot benchmark to validate performance

This task is run manually by the user, not automated. It validates that the change actually works before claiming the fix.

**Step 1: Submit a small pilot job**

Create a test PBS script or modify the experiment config to run a reduced workload:
- `N_particles=10` (not 400)
- `N_members_fcst=50` (same as production)
- `num_steps=6`, `step_first_resampling=4` (same as production)
- Same 4-node allocation (`select=4:ncpus=64`)

This tests 10 × 50 × 3 = 1,500 PlaSim runs with the new parallelism, which should complete in ~15 minutes per PFS step.

**Step 2: Check logs for correctness**

In the job output, verify:
```
NUMA topology: 8 domains/node x 4 node(s) = 32 (raw auto)
Deployment cap: 16 | Effective max_parallel (pre-workload-cap): 16
Final number of parallel runs: 16 (max_parallel=16, num_members=50)
```

**Acceptance criteria:**
- No `mpiexec` launch errors or segfaults in `outerr_forecasts.log`
- All forecast member outputs exist per particle per PFS step:
  ```bash
  # For each PFS-calling step (4, 5, 6) and each particle (0-9):
  # Check postprocessed NetCDF files (50 per particle)
  ls step_${k}/particle_${i}/forecast/PFS_s${k}_p${i}_run.*_plasim_output.nc_postproc.nc | wc -l
  # Expected: 50
  # Check score files (1 per particle per region)
  ls step_${k}/particle_${i}/forecast/PFS_s${k}_p${i}_A_*.npy | wc -l
  # Expected: 1 (or N_regions if multiple regions configured)
  ```
- Observed throughput: ~16 members completing per batch wait cycle (check timestamps in log)
- No `Abort_Message` files in any run directory

**Step 3: Extrapolate to full run**

Measure total wall-clock time across all 3 PFS-calling steps (steps 4, 5, 6) in the pilot.
Each step has a different `lead_time` (10, 5, 0 days), so per-step times will differ.
Extrapolate: `total_pilot_PFS_time × (400 / 10)` to estimate full-run PFS time.
If estimated PFS total > 10 hours, set `PFS_MAX_PARALLEL=32` for the production run.

**Step 4: Update commit message after validation**

After pilot confirms the fix works:
```bash
git commit --allow-empty -m "test: pilot benchmark confirms NUMA-aware PFS parallelism

Pilot with N_particles=10, N_members_fcst=50 on 4 nodes completed in Xm.
Extrapolated full run (400 particles, 3 PFS steps): ~Xh, within 12h limit.
No launch errors, all outputs present, throughput as expected."
```

---

### Task 4: Commit design and plan docs

**Files:**
- `docs/plans/2026-03-01-pfs-numa-aware-parallelism-design.md`
- `docs/plans/2026-03-01-pfs-numa-aware-parallelism-implementation.md`

**Step 1: Commit**

```bash
git add docs/plans/2026-03-01-pfs-numa-aware-parallelism-design.md
git add docs/plans/2026-03-01-pfs-numa-aware-parallelism-implementation.md
git commit -m "docs: add PFS NUMA-aware parallelism design and implementation plan"
```
