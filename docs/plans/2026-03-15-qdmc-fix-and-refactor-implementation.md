# QDMC.sh Bug Fix + Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the CUDA_VISIBLE_DEVICES multi-node bug, then refactor QDMC.sh from a 1,578-line monolith into a sourced-library architecture with consolidated retry logic.

**Architecture:** Split QDMC.sh into 5 files: a thin main driver (`QDMC.sh`) that sources 4 libraries under `RES/lib/` — `cluster_setup.sh`, `lib_utils.sh`, `lib_plasim.sh`, `lib_forecast.sh`. Retry/validation boilerplate consolidated into shared helpers; domain-specific wrappers kept first-class. Launcher scripts hardened with `unset CUDA_VISIBLE_DEVICES` and conda activation.

**Tech Stack:** Bash, shellcheck, PBS/mpiexec, torchrun, conda

**Design doc:** `docs/plans/2026-03-15-qdmc-fix-and-refactor-design.md`

---

## Responsibility Mapping (original → new location)

| Original Function/Section | Lines | New Location | Notes |
|---|---|---|---|
| Section 1: cluster/resource config | 1-91 | `lib/cluster_setup.sh` | |
| `create_pumaburner_namelist()` | 97-140 | `lib/lib_plasim.sh` | |
| Section 3: config parsing | 142-367 | `QDMC.sh` | Stays in main driver |
| `create_step_dir()` | 372-381 | `lib/lib_utils.sh` | |
| `cleanup_step_outputs_keep_lineage()` | 383-398 | `lib/lib_utils.sh` | |
| `write_used_config()` | 400-451 | `lib/lib_utils.sh` | |
| `ensure_used_config()` | 453-459 | `lib/lib_utils.sh` | |
| `initialize_experiment()` | 461-509 | `lib/lib_utils.sh` | |
| `get_output_path()` | 511-516 | `lib/lib_utils.sh` | |
| `get_pangu_plasim_postproc_path()` | 518-524 | `lib/lib_utils.sh` | |
| `get_pangu_plasim_output_dir()` | 526-530 | `lib/lib_utils.sh` | |
| `ensure_run_dir()` | 532-541 | `lib/lib_plasim.sh` | |
| `resolve_ssh_identity_file()` | 543-564 | `lib/lib_utils.sh` | |
| `classify_ssh_preflight_error()` | 566-580 | `lib/lib_utils.sh` | |
| `preflight_ssh_nodes()` | 582-667 | `lib/lib_forecast.sh` | Uses shared SSH/retry helpers |
| `wait_for_pids_or_fail()` | 669-679 | `lib/lib_utils.sh` | |
| `validate_panguplasim_forecast_scores()` | 681-703 | `lib/lib_forecast.sh` | |
| `compute_retry_backoff_delay()` | 705-734 | `lib/lib_utils.sh` | Simplified (validation extracted) |
| `launch_pangu_node_with_retry()` | 736-801 | `lib/lib_forecast.sh` | Uses `validate_retry_params` |
| `launch_pangu_mpiexec_step_with_retry()` | 803-855 | `lib/lib_forecast.sh` | Uses `validate_retry_params` |
| `run_one_particle_one_step()` | 857-963 | `lib/lib_plasim.sh` | Uses `validate_retry_params` |
| `run_particles_one_step()` | 966-1003 | `lib/lib_plasim.sh` | |
| `run_particles_one_step_doing_nothing()` | 1005-1015 | **Removed** | Fake-run debug path |
| `run_PFS()` | 1017-1152 | `lib/lib_forecast.sh` | |
| `postprocess_step_to_panguplasim_inputs()` | 1154-1181 | `lib/lib_forecast.sh` | |
| `run_PanguPlasim()` | 1183-1425 | `lib/lib_forecast.sh` | Launcher generation extracted to `generate_launcher_scripts()` |
| `run_forecasts()` | 1427-1453 | `lib/lib_forecast.sh` | |
| Section 5: main loop | 1455-1578 | `QDMC.sh` | `debug_mode` fake-run branches removed |
| Duplicated SSH opts array (lines 587-597, 1306-1316) | — | `lib/lib_utils.sh` as `build_ssh_opts()` | |
| Duplicated retry validation (lines 605-612, 751-766, 813-828, 889-904) | — | `lib/lib_utils.sh` as `validate_retry_params()` | |
| Redundant echo line 359 | — | **Removed** | |

---

### Task 1: Create lib/ directory and lib_utils.sh with shared helpers

**Files:**
- Create: `AI-RES/RES/lib/lib_utils.sh`

**Step 1: Create lib/ directory**

Run: `mkdir -p AI-RES/RES/lib`

**Step 2: Write lib_utils.sh**

This file contains all shared helper functions. Extract from QDMC.sh:

```bash
#!/bin/bash
#
# lib_utils.sh — Shared utility functions for the QDMC experiment driver.
#
# Sourced by QDMC.sh. Do not run directly.
#

# ── Retry / Backoff Helpers ──────────────────────────────────────────────────

function validate_retry_params() {
    # Validate and sanitize retry-related environment variables.
    # Usage: validate_retry_params <varname> <value> <default> <pattern>
    #   pattern: "positive" (^[1-9][0-9]*$) or "nonneg" (^[0-9]+$)
    local varname=$1 value=$2 default=$3 pattern_type=${4:-nonneg}
    local regex
    if [ "$pattern_type" = "positive" ]; then
        regex='^[1-9][0-9]*$'
    else
        regex='^[0-9]+$'
    fi
    if ! [[ "$value" =~ $regex ]]; then
        echo "WARNING: Invalid $varname=$value; using $default"
        echo "$default"
        return 0
    fi
    echo "$value"
}

function compute_retry_backoff_delay() {
    # Exponential backoff with optional jitter, capped at max_sleep_sec.
    local attempt=$1
    local base_sleep_sec=$2
    local max_sleep_sec=$3
    local jitter_sec=${4:-0}
    local delay

    delay=$(( base_sleep_sec * (1 << (attempt - 1)) ))
    if [ "$jitter_sec" -gt 0 ]; then
        delay=$(( delay + (RANDOM % (jitter_sec + 1)) ))
    fi
    if [ "$max_sleep_sec" -gt 0 ] && [ "$delay" -gt "$max_sleep_sec" ]; then
        delay=$max_sleep_sec
    fi
    echo "$delay"
}

# ── SSH Helpers ──────────────────────────────────────────────────────────────

function resolve_ssh_identity_file() {
    # Pick a stable SSH identity to avoid agent key flooding.
    local candidate
    for candidate in "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa" "$HOME/.ssh/id_ecdsa" "$HOME/.ssh/id_dsa"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    for candidate in "$HOME"/.ssh/id_*; do
        [ -f "$candidate" ] || continue
        case "$candidate" in
            *.pub|*known_hosts*|*authorized_keys*|*config*) continue ;;
        esac
        echo "$candidate"
        return 0
    done
    echo ""
    return 1
}

function classify_ssh_preflight_error() {
    # Classify common SSH preflight failures for faster diagnosis.
    local msg=$1
    if [[ "$msg" == *"Permission denied"* ]]; then
        echo "auth_or_pbs_attach_denied"
    elif [[ "$msg" == *"Connection timed out"* ]] || [[ "$msg" == *"Operation timed out"* ]]; then
        echo "connect_timeout"
    elif [[ "$msg" == *"No route to host"* ]] || [[ "$msg" == *"Connection refused"* ]]; then
        echo "network_unreachable"
    elif [[ "$msg" == *"Host key verification failed"* ]]; then
        echo "host_key_verification"
    else
        echo "ssh_unknown"
    fi
}

function build_ssh_opts() {
    # Build SSH options array from an identity file path.
    # Usage: local -a opts; build_ssh_opts opts "/path/to/key"
    local -n _opts_ref=$1
    local identity_file=$2
    _opts_ref=(
        -o BatchMode=yes
        -o IdentitiesOnly=yes
        -o PreferredAuthentications=hostbased,publickey
        -o PasswordAuthentication=no
        -o KbdInteractiveAuthentication=no
        -o ConnectTimeout=20
        -o IdentityAgent=none
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
    )
    if [ -n "$identity_file" ]; then
        _opts_ref+=(-i "$identity_file")
    fi
}

# ── Process Helpers ──────────────────────────────────────────────────────────

function wait_for_pids_or_fail() {
    # Wait all PIDs and fail if any child process failed.
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    [ "$failed" -eq 0 ]
}

# ── Path Helpers ─────────────────────────────────────────────────────────────

function get_output_path() {
    i=$1; k=$2
    OUTPUT_NAME="plasim_out.step_$k.particle_$i.nc"
    OUTPUT_PATH=$PATH_EXP/step_$k/particle_$i/output/$OUTPUT_NAME
}

function get_pangu_plasim_postproc_path() {
    i=$1; k=$2
    PANGUPLASIM_IN_DIR=$PATH_EXP/step_$k/particle_$i/output
    PANGUPLASIM_IN_NAME="panguplasim_in.step_$k.particle_$i.nc"
    PANGUPLASIM_IN_PATH=$PANGUPLASIM_IN_DIR/$PANGUPLASIM_IN_NAME
}

function get_pangu_plasim_output_dir() {
    i=$1; k=$2
    PANGUPLASIM_OUT_DIR=$PATH_EXP/step_$k/particle_$i/forecast
}

# ── Directory / Experiment Helpers ───────────────────────────────────────────

function create_step_dir() {
    next_k=$1
    echo "Creating step directory for k = $next_k"
    for i in $(seq 0 $(($N_particles - 1))); do
        mkdir -p $PATH_EXP/step_$next_k/particle_$i/restart
        mkdir -p $PATH_EXP/step_$next_k/particle_$i/output
        mkdir -p $PATH_EXP/step_$next_k/particle_$i/forecast
    done
}

function cleanup_step_outputs_keep_lineage() {
    cleanup_k=$1
    step_dir=$PATH_EXP/step_$cleanup_k
    if [ ! -d "$step_dir" ]; then
        return
    fi
    echo "Lineage-safe cleanup for step $cleanup_k (keeping restart lineage)"
    for i in $(seq 0 $(($N_particles - 1))); do
        output_dir="$step_dir/particle_$i/output"
        if [ -d "$output_dir" ]; then
            rm -f "$output_dir"/*.nc
            rm -f "$output_dir"/plasim_output
        fi
    done
}

function write_used_config() {
    local pangu_fields=""
    if [ "$forecast_method" = "Pangu-Plasim" ]; then
        pangu_fields=$(cat <<'INNER'
    "clim_file": "__CLIM_FILE__",
    "threshold_json_file": "__THRESHOLD_JSON_FILE__",
    "scorer": __SCORER_JSON__,
INNER
)
        pangu_fields="${pangu_fields//__CLIM_FILE__/$clim_file}"
        pangu_fields="${pangu_fields//__THRESHOLD_JSON_FILE__/$threshold_json_file}"
        pangu_fields="${pangu_fields//__SCORER_JSON__/$scorer_json}"
    fi

    cat <<EOF > $PATH_EXP/$experiment_name"_used_config.json"
  {
    "experiment_name": "$experiment_name",
    "region": "$region",
    "N_particles": $N_particles,
    "dtau": $dtau,
    "num_steps": $num_steps,
    "target_duration": $target_duration,
    "lower_tail": $lower_tail,
    "EPS_perturb": $EPS_perturb,
    "create_ICs_script": "$create_ICs_script",
    "diverse_initial_conditions": $diverse_initial_conditions,
    "perturb_from_past": $perturb_from_past,
    "resampling_method": "$resampling_method",
    "use_quantile": $use_quantile,
    "step_first_resampling": $step_first_resampling,
    "resampling_last_step": $resampling_last_step,
    "theta_type": "$theta_type",
    "splitting_constant": $splitting_constant,
    "alpha": $alpha,
    "divide_by_parent_weights": $divide_by_parent_weights,
    "forecast_method": "$forecast_method",
    "N_members_fcst": $N_members_fcst,
    "PATH_CODE": "$PATH_CODE",
    "PATH_SCRATCH": "$PATH_SCRATCH",
    "PATH_EXP": "$PATH_EXP",
    "PATH_REFERENCE_RUN_DIR": "$PATH_REFERENCE_RUN_DIR",
    "PATH_POSTPROC_NL": "$PATH_POSTPROC_NL",
    "POSTPROC_SCRIPT": "$POSTPROC_SCRIPT",
    "job_manager": "$job_manager",
    "path_config_file": "$CONFIG_FILE",
    "debug_mode": $debug_mode,
    "delete_output": $delete_output_json,
    ${pangu_fields}"save_srv_output": $save_srv_output
  }
EOF
}

function ensure_used_config() {
    local used_cfg="$PATH_EXP/${experiment_name}_used_config.json"
    if [ ! -f "$used_cfg" ]; then
        echo "WARNING: used_config missing; recreating $used_cfg"
        write_used_config
    fi
}

function initialize_experiment() {
    rm -rf $PATH_EXP && mkdir -p $PATH_EXP
    write_used_config

    cd $PATH_CODE

    create_ICs_script_abs="$PATH_CODE/$create_ICs_script"

    if [[ "$diverse_initial_conditions" == "true" ]]; then
        caps=$(bash "$create_ICs_script_abs" --capabilities 2>/dev/null || true)
        if ! grep -qw "diverse_initial_conditions" <<< "$caps"; then
            echo "ERROR: create_ICs_script '$create_ICs_script' does not support" \
                 "diverse_initial_conditions (missing capability token)" >&2
            exit 1
        fi
    fi

    echo "Loading the ICs"
    list_ICs=$(bash "$create_ICs_script_abs" \
        "$N_particles" "$region" "$length_simu" "$PATH_SCRATCH" \
        "$diverse_initial_conditions")
    list_ICs=($list_ICs)
    echo "List of ICs: ${list_ICs[@]}"

    if [[ ${#list_ICs[@]} -ne $N_particles ]]; then
        echo "ERROR: IC script returned ${#list_ICs[@]} paths, expected $N_particles" >&2
        exit 1
    fi

    echo "Creating the working tree"
    python3 $PATH_RESAMPLING/create_working_tree.py $PATH_EXP $N_particles ${list_ICs[@]}
    mkdir -p $PATH_EXP/resampling

    create_step_dir 0

    echo "Creating the run directory"
    mkdir -p $PATH_EXP/run
    for i in $(seq 0 $(($N_particles - 1))); do
        mkdir -p $PATH_EXP/run/run_$i
        cp -r $PATH_REFERENCE_RUN_DIR/. $PATH_EXP/run/run_$i
        IC_i="${list_ICs[$i]}"
        cp $IC_i $PATH_EXP/step_0/particle_$i/restart/restart_end
        sed -i '/EPSRESTART/s/=.*/=     '"$EPS_perturb"'/' $PATH_EXP/run/run_$i/plasim_namelist
    done
}
```

**Step 3: Verify syntax**

Run: `bash -n AI-RES/RES/lib/lib_utils.sh`
Expected: No output (clean parse)

**Step 4: Commit**

```bash
git add AI-RES/RES/lib/lib_utils.sh
git commit -m "refactor: extract lib_utils.sh with shared helpers from QDMC.sh"
```

---

### Task 2: Create cluster_setup.sh

**Files:**
- Create: `AI-RES/RES/lib/cluster_setup.sh`

**Step 1: Write cluster_setup.sh**

Extract lines 1-91 of QDMC.sh (cluster detection, node counting, GPU detection, conda activation). This becomes a sourceable file that sets global variables (`job_manager`, `num_cpus`, `num_nodes`, `nodes`, `NNODES`, `MASTER_ADDR`, `MASTER_PORT`, `cores_per_node`, `num_nodes_plasim`, `ppn`, `ppr`, `num_parallel`).

```bash
#!/bin/bash
#
# cluster_setup.sh — Cluster detection, resource enumeration, environment activation.
#
# Sourced by QDMC.sh. Do not run directly.
# Expects: CONFIG_FILE is set before sourcing.
#

# Activate Python environment
ml conda
conda activate aires

job_manager=$(jq -r '.job_manager' "$CONFIG_FILE")
if [ "$job_manager" = "PBS" ]; then
    num_cpus=$(qstat -f $PBS_JOBID | grep "Resource_List.ncpus" | awk -F= '{print $2}' | tr -d '[:space:]')
    num_nodes=$(qstat -f $PBS_JOBID | grep "Resource_List.nodect" | awk -F= '{print $2}' | tr -d '[:space:]')
    nodes=($(sort -u $PBS_NODEFILE))
elif [ "$job_manager" = "SLURM" ]; then
    num_cpus=$SLURM_NTASKS
    num_nodes=$SLURM_NNODES
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
else
    echo "Job manager not supported"
    exit 1
fi
NNODES=${#nodes[@]}
MASTER_ADDR=${nodes[0]}
MASTER_PORT=29500
cores_per_node=$((num_cpus / num_nodes))
num_nodes_plasim=$num_nodes
if [ "$job_manager" = "SLURM" ]; then
    num_nodes_plasim=$((num_nodes - 1))
fi
if [ "$num_nodes_plasim" -eq 0 ]; then
    num_nodes_plasim=1
fi
ppn=$((cores_per_node * 6 / 7))
ppr=8
num_parallel=$((ppn * num_nodes_plasim / ppr))

echo "########################################################"
echo "Computational parameters"
echo "########################################################"
date
echo "Available number of nodes: $num_nodes"
echo "Setting number of nodes for PlaSim to: $num_nodes_plasim"
echo "Available number of cores per node: $cores_per_node"
echo "Setting number of processes running per node (ppn) for PlaSim to: $ppn"
echo "Using Plasim compiled with $ppr cores per run"
echo "Then number of parallel runs is set to: $num_parallel"
echo "########################################################"
```

**Step 2: Verify syntax**

Run: `bash -n AI-RES/RES/lib/cluster_setup.sh`
Expected: No output (clean parse)

**Step 3: Commit**

```bash
git add AI-RES/RES/lib/cluster_setup.sh
git commit -m "refactor: extract cluster_setup.sh from QDMC.sh"
```

---

### Task 3: Create lib_plasim.sh

**Files:**
- Create: `AI-RES/RES/lib/lib_plasim.sh`

**Step 1: Write lib_plasim.sh**

Contains: `create_pumaburner_namelist()`, `ensure_run_dir()`, `run_one_particle_one_step()` (with consolidated retry using `validate_retry_params`), `run_particles_one_step()`.

Key changes from original:
- `run_one_particle_one_step()` PBS retry path uses `validate_retry_params()` instead of inline validation (removes ~16 lines of duplication)
- SLURM retry path updated to use `compute_retry_backoff_delay()` for consistency (low-risk cleanup)
- `run_particles_one_step_doing_nothing()` NOT included (removed per design)

```bash
#!/bin/bash
#
# lib_plasim.sh — PlaSim execution and postprocessing functions.
#
# Sourced by QDMC.sh. Do not run directly.
# Depends on: lib_utils.sh (validate_retry_params, compute_retry_backoff_delay,
#             wait_for_pids_or_fail, ensure_run_dir, get_output_path)
#

function create_pumaburner_namelist() {
    PATH_YAML_CONFIG=$1
    PATH_POSTPROC_NL_DIR=$2
    experiment_name=$3
    postprocess_precip=false
    echo "Creating pumaburner namelist for model with config file: $PATH_YAML_CONFIG"
    postproc_variables=""
    for upper_air_variable in $(yq -r '.PLASIM.upper_air_variables[]' "$PATH_YAML_CONFIG"); do
        if [ "$upper_air_variable" != "zg" ]; then
            postproc_variables+="${upper_air_variable},"
        fi
    done
    for surface_variable in $(yq -r '.PLASIM.surface_variables[]' "$PATH_YAML_CONFIG"); do
        postproc_variables+="$surface_variable,"
    done
    for land_variable in $(yq -r '.PLASIM.land_variables[]' "$PATH_YAML_CONFIG"); do
        if [ "$land_variable" = "snd" ]; then
            postproc_variables+="sndc,"
        else
            postproc_variables+="$land_variable,"
        fi
    done
    for ocean_variable in $(yq -r '.PLASIM.ocean_variables[]' "$PATH_YAML_CONFIG"); do
        postproc_variables+="$ocean_variable,"
    done
    for diagnostic_variable in $(yq -r '.PLASIM.diagnostic_variables[]' "$PATH_YAML_CONFIG"); do
        if [[ "$diagnostic_variable" == pr_*h ]]; then
            if [ "$postprocess_precip" = false ]; then
                postprocess_precip=true
                postproc_variables+="pr,"
            fi
        else
            postproc_variables+="$diagnostic_variable,"
        fi
    done
    echo "Postproc variables: $postproc_variables"
    PATH_POSTPROC_NL="$PATH_POSTPROC_NL_DIR/$experiment_name"
    PATH_POSTPROC_NL+=".nl"
    cat <<EOF > $PATH_POSTPROC_NL
        code=${postproc_variables%?},
        MODLEV = 10,9,8,7,6,5,4,3,2,1,0
        vtype=sigma,htype=g,mean=0,netcdf=1
EOF
}

function ensure_run_dir() {
    i=$1
    run_dir="$PATH_EXP/run/run_$i"
    if [ ! -d "$run_dir" ]; then
        echo "WARNING: missing run dir $run_dir; recreating from $PATH_REFERENCE_RUN_DIR"
        mkdir -p "$run_dir"
        cp -r "$PATH_REFERENCE_RUN_DIR/." "$run_dir"
        sed -i '/EPSRESTART/s/=.*/=     '"$EPS_perturb"'/' "$run_dir/plasim_namelist"
    fi
}

function run_one_particle_one_step() {
    i=$1
    k=$2
    duration=$3
    index_run_par=$(( i % $num_parallel ))
    ni=$((ppr * index_run_par / ppn))
    sc=$((ppr * (index_run_par) % ppn))
    ec=$((sc + ppr - 1))
    echo "Run particle $i at step $k: Node index $ni, cores $sc-$ec. "
    ensure_run_dir $i
    cd $PATH_EXP/run/run_$i
    get_output_path $i $k
    echo "Run particle $i at step $k started for $duration days"
    restart_start="$PATH_EXP/step_$k/particle_$i/restart/restart_start"
    if [ ! -f "$restart_start" ]; then
        echo "ERROR: Missing restart_start for particle $i at step $k: $restart_start"
        return 1
    fi
    if ! cp "$restart_start" plasim_restart; then
        echo "ERROR: Failed to copy restart_start for particle $i at step $k"
        return 1
    fi
    sed -i '/N_RUN_DAYS/s/=.*/=     '"$duration"'/' $PATH_EXP/run/run_$i/plasim_namelist
    [ -e plasim_output ] && rm -f plasim_output
    if [ $job_manager = "PBS" ]; then
        local max_attempts=${AIRES_PBS_MPIEXEC_MAX_ATTEMPTS:-3}
        local retry_base_sec=${AIRES_PBS_MPIEXEC_RETRY_BASE_SEC:-10}
        local retry_max_sec=${AIRES_PBS_MPIEXEC_RETRY_MAX_SEC:-90}
        local retry_jitter_sec=${AIRES_PBS_MPIEXEC_RETRY_JITTER_SEC:-10}
        local attempt rc delay

        max_attempts=$(validate_retry_params "AIRES_PBS_MPIEXEC_MAX_ATTEMPTS" "$max_attempts" 3 positive)
        retry_base_sec=$(validate_retry_params "AIRES_PBS_MPIEXEC_RETRY_BASE_SEC" "$retry_base_sec" 10)
        retry_max_sec=$(validate_retry_params "AIRES_PBS_MPIEXEC_RETRY_MAX_SEC" "$retry_max_sec" 90)
        retry_jitter_sec=$(validate_retry_params "AIRES_PBS_MPIEXEC_RETRY_JITTER_SEC" "$retry_jitter_sec" 10)

        for attempt in $(seq 1 "$max_attempts"); do
            if [ "$attempt" -gt 1 ]; then
                [ -e plasim_output ] && rm -f plasim_output
                if ! cp "$restart_start" plasim_restart; then
                    echo "ERROR: Failed to reset restart_start for retry of particle $i at step $k"
                    return 1
                fi
            fi

            mpiexec -host ${nodes[$ni]} -n $ppr --cpu-bind list:$sc-$ec ./most_plasim_t42_l10_p8.x
            rc=$?
            if [ "$rc" -eq 0 ]; then
                break
            fi

            if [ "$attempt" -ge "$max_attempts" ]; then
                echo "ERROR: PlaSim run failed for particle $i at step $k on node ${nodes[$ni]} after $max_attempts attempt(s); last rc=$rc"
                return 1
            fi

            delay=$(compute_retry_backoff_delay "$attempt" "$retry_base_sec" "$retry_max_sec" "$retry_jitter_sec")
            echo "WARNING: PlaSim launch/run failed for particle $i at step $k on node ${nodes[$ni]} (attempt $attempt/$max_attempts, rc=$rc). Retrying in ${delay}s..."
            sleep "$delay"
        done
    else
        # SLURM path — updated to use shared backoff for consistency
        local max_attempts=${AIRES_SLURM_MPIEXEC_MAX_ATTEMPTS:-10}
        local retry_base_sec=${AIRES_SLURM_MPIEXEC_RETRY_BASE_SEC:-5}
        local retry_max_sec=${AIRES_SLURM_MPIEXEC_RETRY_MAX_SEC:-30}
        local retry_jitter_sec=${AIRES_SLURM_MPIEXEC_RETRY_JITTER_SEC:-15}
        local attempt rc delay

        max_attempts=$(validate_retry_params "AIRES_SLURM_MPIEXEC_MAX_ATTEMPTS" "$max_attempts" 10 positive)
        retry_base_sec=$(validate_retry_params "AIRES_SLURM_MPIEXEC_RETRY_BASE_SEC" "$retry_base_sec" 5)
        retry_max_sec=$(validate_retry_params "AIRES_SLURM_MPIEXEC_RETRY_MAX_SEC" "$retry_max_sec" 30)
        retry_jitter_sec=$(validate_retry_params "AIRES_SLURM_MPIEXEC_RETRY_JITTER_SEC" "$retry_jitter_sec" 15)

        for attempt in $(seq 1 "$max_attempts"); do
            mpiexec -host ${nodes[$ni]} -n $ppr --membind list:$sc-$ec ./most_plasim_t42_l10_p8.x
            rc=$?
            if [ "$rc" -eq 0 ]; then
                break
            fi

            if [ "$attempt" -ge "$max_attempts" ]; then
                echo "ERROR: All attempts failed for particle $i at step $k."
                return 1
            fi

            delay=$(compute_retry_backoff_delay "$attempt" "$retry_base_sec" "$retry_max_sec" "$retry_jitter_sec")
            echo "WARNING: PlaSim run failed for particle $i at step $k (attempt $attempt/$max_attempts, rc=$rc). Retrying in ${delay}s..."
            sleep "$delay"
        done
    fi
    echo "run particle $i at step $k finished"
    echo "OUTPUT_PATH = $OUTPUT_PATH"
    echo "postprocessing started"
    [ -e plasim_output ] && bash $POSTPROC_SCRIPT $PATH_POSTPROC_NL $PATH_EXP/run/run_$i/plasim_output $PATH_EXP/step_$k/particle_$i/output/$OUTPUT_NAME &>> $PATH_EXP/outerr_postproc.log
    echo "postprocessing finished"
    if [ $save_srv_output = true ]; then
        cp plasim_output $PATH_EXP/step_$k/particle_$i/output/plasim_output
    fi
    [ -e plasim_status ] && cp plasim_status plasim_restart
    if [ ! -e plasim_restart ]; then
        echo "ERROR: Missing plasim_restart after run for particle $i at step $k"
        return 1
    fi
    if ! cp plasim_restart $PATH_EXP/step_$k/particle_$i/restart/restart_end; then
        echo "ERROR: Failed to write restart_end for particle $i at step $k"
        return 1
    fi
    return 0
}

function run_particles_one_step() {
    next_k=$1
    duration=$2
    echo "Running particles for step $next_k"
    pids=()
    for i in $(seq 0 $(($N_particles - 1))); do
        if [ $((i % num_parallel)) -eq 0 ] && [ $i -ne 0 ]; then
            echo "Waiting all runs to finish (at run = $i)"
            if ! wait_for_pids_or_fail "${pids[@]}"; then
                echo "ERROR: One or more particle runs failed at step $next_k"
                return 1
            fi
            pids=()
        fi
        echo "run_one_particle_one_step $i $next_k $duration"
        ensure_run_dir $i
        default_epsilon_kick="0.003"
        if (( $(echo "$EPS_perturb == 0" | bc -l) )) && [ "$next_k" -eq 1 ]; then
            sed -i '/EPSRESTART/s/=.*/=     '"$default_epsilon_kick"'/' $PATH_EXP/run/run_$i/plasim_namelist
            echo "EPSRESTART set to $default_epsilon_kick for the first run"
        elif (( $(echo "$EPS_perturb == 0" | bc -l) )) && [ "$next_k" -eq 2 ]; then
            sed -i '/EPSRESTART/s/=.*/=     '"$EPS_perturb"'/' "$PATH_EXP/run/run_$i/plasim_namelist"
            echo "EPSRESTART set back to $EPS_perturb for the rest of the runs"
        fi
        run_one_particle_one_step $i $next_k $duration &
        pids+=($!)
        sleep 0.5
    done
    if [ "${#pids[@]}" -gt 0 ]; then
        if ! wait_for_pids_or_fail "${pids[@]}"; then
            echo "ERROR: One or more particle runs failed at step $next_k"
            return 1
        fi
    fi
    return 0
}
```

**Step 2: Verify syntax**

Run: `bash -n AI-RES/RES/lib/lib_plasim.sh`
Expected: No output

**Step 3: Commit**

```bash
git add AI-RES/RES/lib/lib_plasim.sh
git commit -m "refactor: extract lib_plasim.sh with PlaSim execution from QDMC.sh"
```

---

### Task 4: Create lib_forecast.sh with hardened launcher generation

**Files:**
- Create: `AI-RES/RES/lib/lib_forecast.sh`

**Step 1: Write lib_forecast.sh**

Contains: `generate_launcher_scripts()` (with CUDA fix + conda activation), `preflight_ssh_nodes()`, `validate_panguplasim_forecast_scores()`, `launch_pangu_node_with_retry()`, `launch_pangu_mpiexec_step_with_retry()`, `run_PFS()`, `postprocess_step_to_panguplasim_inputs()`, `run_PanguPlasim()`, `run_forecasts()`.

Key changes:
- `generate_launcher_scripts()` is a new function extracted from `run_PanguPlasim()` lines 1220-1269
- Both launcher scripts now: (1) `unset CUDA_VISIBLE_DEVICES`, (2) activate conda instead of baking PATH/LD_LIBRARY_PATH
- `preflight_ssh_nodes()` uses `validate_retry_params()` instead of inline validation
- `launch_pangu_node_with_retry()` uses `validate_retry_params()`
- `launch_pangu_mpiexec_step_with_retry()` uses `validate_retry_params()`
- SSH opts built via `build_ssh_opts()` instead of duplicated array

```bash
#!/bin/bash
#
# lib_forecast.sh — Forecast orchestration (Pangu-Plasim, PFS, control paths).
#
# Sourced by QDMC.sh. Do not run directly.
# Depends on: lib_utils.sh (validate_retry_params, compute_retry_backoff_delay,
#             build_ssh_opts, resolve_ssh_identity_file, classify_ssh_preflight_error,
#             wait_for_pids_or_fail, get_pangu_plasim_postproc_path, get_pangu_plasim_output_dir,
#             get_output_path)
#

function generate_launcher_scripts() {
    # Generate torchrun launcher scripts for Pangu-Plasim GPU inference.
    # Called once per experiment; scripts are cached in PATH_EXP.
    LAUNCHER_SCRIPT=$PATH_EXP/run_torchrun_node.sh
    if [ ! -f "$LAUNCHER_SCRIPT" ]; then
        cat > "$LAUNCHER_SCRIPT" <<'EOF_LAUNCHER'
#!/bin/bash
# Pangu-Plasim single-node GPU inference launcher.
# Self-contained: activates conda, discovers local GPUs.
unset CUDA_VISIBLE_DEVICES
ml conda
conda activate aires
export OMP_NUM_THREADS=4
CONFIG=$1; MODEL_DIR=$2; SCRIPT=$3; NUM_GPUS=$4; LOG_FILE=$5
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    $SCRIPT \
    --config $CONFIG \
    --panguplasim_model_dir $MODEL_DIR \
    >> $LOG_FILE 2>&1
EOF_LAUNCHER
        chmod +x "$LAUNCHER_SCRIPT"
    fi

    LAUNCHER_SCRIPT_MPI=$PATH_EXP/run_torchrun_node_mpi.sh
    if [ ! -f "$LAUNCHER_SCRIPT_MPI" ]; then
        cat > "$LAUNCHER_SCRIPT_MPI" <<'EOF_LAUNCHER_MPI'
#!/bin/bash
# Pangu-Plasim multi-node launcher entrypoint for one mpiexec rank per node.
# Self-contained: activates conda, discovers local GPUs.
unset CUDA_VISIBLE_DEVICES
ml conda
conda activate aires
export OMP_NUM_THREADS=4
STEP_DIR=$1; MODEL_DIR=$2; SCRIPT=$3; NUM_GPUS=$4; EXP_DIR=$5; STEP_NUMBER=$6
RANK=${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
CONFIG="$STEP_DIR/config_PanguPlasim_node_${RANK}.json"
LOG_FILE="$EXP_DIR/outerr_forecasts_node_${RANK}_step_${STEP_NUMBER}.log"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Missing node config for rank $RANK: $CONFIG" >> "$LOG_FILE"
    exit 1
fi
{
    echo "Launcher rank=$RANK host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
} >> "$LOG_FILE" 2>&1
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    $SCRIPT \
    --config $CONFIG \
    --panguplasim_model_dir $MODEL_DIR \
    >> $LOG_FILE 2>&1
EOF_LAUNCHER_MPI
        chmod +x "$LAUNCHER_SCRIPT_MPI"
    fi
}

function preflight_ssh_nodes() {
    local identity_file=$1
    local max_attempts=${AIRES_SSH_PREFLIGHT_MAX_ATTEMPTS:-3}
    local retry_sleep_sec=${AIRES_SSH_PREFLIGHT_RETRY_SLEEP_SEC:-15}

    max_attempts=$(validate_retry_params "AIRES_SSH_PREFLIGHT_MAX_ATTEMPTS" "$max_attempts" 3 positive)
    retry_sleep_sec=$(validate_retry_params "AIRES_SSH_PREFLIGHT_RETRY_SLEEP_SEC" "$retry_sleep_sec" 15)

    local -a base_opts
    build_ssh_opts base_opts "$identity_file"

    local -a pending_nodes=("${nodes[@]}")
    local -a failed_nodes=()
    local -a failed_summaries=()
    local -a next_pending_nodes=()
    local attempt node out rc last_line err_class

    for attempt in $(seq 1 "$max_attempts"); do
        [ "${#pending_nodes[@]}" -eq 0 ] && break
        echo "SSH preflight attempt $attempt/$max_attempts on ${#pending_nodes[@]} node(s)"
        next_pending_nodes=()
        failed_nodes=()
        failed_summaries=()

        for node in "${pending_nodes[@]}"; do
            echo "SSH preflight check on node $node"
            out=$(ssh "${base_opts[@]}" "$node" "hostname >/dev/null" 2>&1 < /dev/null)
            rc=$?
            if [ "$rc" -ne 0 ]; then
                last_line=$(printf '%s\n' "$out" | tail -n 1)
                [ -n "$last_line" ] || last_line="<no ssh stderr>"
                err_class=$(classify_ssh_preflight_error "$out")
                failed_nodes+=("$node")
                failed_summaries+=("$err_class | rc=$rc | $last_line")
                next_pending_nodes+=("$node")
                if [ "$attempt" -lt "$max_attempts" ]; then
                    echo "WARNING: SSH preflight failed on $node ($err_class, rc=$rc). Will retry."
                fi
            fi
        done

        if [ "${#next_pending_nodes[@]}" -eq 0 ]; then
            echo "SSH preflight passed on all ${#nodes[@]} nodes."
            return 0
        fi

        if [ "$attempt" -lt "$max_attempts" ]; then
            echo "SSH preflight still failing on ${#next_pending_nodes[@]} node(s): ${next_pending_nodes[*]}"
            echo "Retrying SSH preflight in ${retry_sleep_sec}s..."
            sleep "$retry_sleep_sec"
        fi

        pending_nodes=("${next_pending_nodes[@]}")
    done

    if [ "${#failed_nodes[@]}" -ne 0 ]; then
        local idx
        for idx in "${!failed_nodes[@]}"; do
            echo "ERROR: SSH preflight final failure on ${failed_nodes[$idx]}: ${failed_summaries[$idx]}"
        done
        echo "ERROR: SSH preflight failed on ${#failed_nodes[@]} node(s) after ${max_attempts} attempt(s): ${failed_nodes[*]}"
        return 1
    fi

    echo "ERROR: SSH preflight failed for unknown reason (no failed node list captured)."
    return 1
}

function validate_panguplasim_forecast_scores() {
    local step_number=$1
    local missing_count=0
    local missing_examples=""
    local i score_file
    for i in $(seq 0 $((N_particles - 1))); do
        score_file="$PATH_EXP/step_${step_number}/particle_${i}/forecast/PanguPlasimFS_s${step_number}_p${i}_A_${region}.npy"
        if [ ! -f "$score_file" ]; then
            if [ "$missing_count" -lt 10 ]; then
                missing_examples="${missing_examples}\n  - $score_file"
            fi
            missing_count=$((missing_count + 1))
        fi
    done
    if [ "$missing_count" -ne 0 ]; then
        echo "ERROR: Missing $missing_count Pangu-Plasim score files at step $step_number."
        echo -e "Examples:${missing_examples}"
        return 1
    fi
    echo "Validated Pangu-Plasim score files for step $step_number: $N_particles/$N_particles present."
    return 0
}

function launch_pangu_node_with_retry() {
    local node_name=$1
    local step_number=$2
    local p_start=$3
    local p_end=$4
    local path_config_node=$5
    local node_log=$6
    local launch_backend=${7:-ssh}
    local max_attempts=${AIRES_SSH_LAUNCH_MAX_ATTEMPTS:-3}
    local retry_base_sec=${AIRES_SSH_LAUNCH_RETRY_BASE_SEC:-10}
    local retry_max_sec=${AIRES_SSH_LAUNCH_RETRY_MAX_SEC:-60}
    local retry_jitter_sec=${AIRES_SSH_LAUNCH_RETRY_JITTER_SEC:-10}
    local attempt rc delay

    max_attempts=$(validate_retry_params "AIRES_SSH_LAUNCH_MAX_ATTEMPTS" "$max_attempts" 3 positive)
    retry_base_sec=$(validate_retry_params "AIRES_SSH_LAUNCH_RETRY_BASE_SEC" "$retry_base_sec" 10)
    retry_max_sec=$(validate_retry_params "AIRES_SSH_LAUNCH_RETRY_MAX_SEC" "$retry_max_sec" 60)
    retry_jitter_sec=$(validate_retry_params "AIRES_SSH_LAUNCH_RETRY_JITTER_SEC" "$retry_jitter_sec" 10)

    for attempt in $(seq 1 "$max_attempts"); do
        if [ "$launch_backend" = "mpiexec" ]; then
            mpiexec -host "$node_name" -n 1 bash "$LAUNCHER_SCRIPT" \
                "$path_config_node" \
                "$DIR_PANGUPLASIM_MODEL" \
                "$DIR_PANGUPLASIMFS/PanguPlasimFS.py" \
                "$NUM_GPUS_PER_NODE" \
                "$node_log"
        else
            ssh "${ssh_opts[@]}" "${node_name}" "bash $LAUNCHER_SCRIPT \
                $path_config_node \
                $DIR_PANGUPLASIM_MODEL \
                $DIR_PANGUPLASIMFS/PanguPlasimFS.py \
                $NUM_GPUS_PER_NODE \
                $node_log"
        fi
        rc=$?
        if [ "$rc" -eq 0 ]; then
            return 0
        fi

        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "ERROR: $launch_backend launch failed on node $node_name for step $step_number (particles $p_start-$p_end) after $max_attempts attempt(s); rc=$rc"
            return "$rc"
        fi

        delay=$(compute_retry_backoff_delay "$attempt" "$retry_base_sec" "$retry_max_sec" "$retry_jitter_sec")
        echo "WARNING: $launch_backend launch failed on node $node_name for step $step_number (particles $p_start-$p_end), attempt $attempt/$max_attempts (rc=$rc). Retrying in ${delay}s..."
        sleep "$delay"
    done

    echo "ERROR: $launch_backend launch retry loop fell through on node $node_name for step $step_number."
    return 1
}

function launch_pangu_mpiexec_step_with_retry() {
    local step_number=$1
    local launch_node_count=$2
    local max_attempts=${AIRES_PANGU_MPIEXEC_MAX_ATTEMPTS:-3}
    local retry_base_sec=${AIRES_PANGU_MPIEXEC_RETRY_BASE_SEC:-10}
    local retry_max_sec=${AIRES_PANGU_MPIEXEC_RETRY_MAX_SEC:-60}
    local retry_jitter_sec=${AIRES_PANGU_MPIEXEC_RETRY_JITTER_SEC:-10}
    local attempt rc delay

    max_attempts=$(validate_retry_params "AIRES_PANGU_MPIEXEC_MAX_ATTEMPTS" "$max_attempts" 3 positive)
    retry_base_sec=$(validate_retry_params "AIRES_PANGU_MPIEXEC_RETRY_BASE_SEC" "$retry_base_sec" 10)
    retry_max_sec=$(validate_retry_params "AIRES_PANGU_MPIEXEC_RETRY_MAX_SEC" "$retry_max_sec" 60)
    retry_jitter_sec=$(validate_retry_params "AIRES_PANGU_MPIEXEC_RETRY_JITTER_SEC" "$retry_jitter_sec" 10)

    for attempt in $(seq 1 "$max_attempts"); do
        mpiexec -n "$launch_node_count" --ppn 1 bash "$LAUNCHER_SCRIPT_MPI" \
            "$PATH_EXP/step_$step_number" \
            "$DIR_PANGUPLASIM_MODEL" \
            "$DIR_PANGUPLASIMFS/PanguPlasimFS.py" \
            "$NUM_GPUS_PER_NODE" \
            "$PATH_EXP" \
            "$step_number"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            return 0
        fi

        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "ERROR: mpiexec multi-node launch failed at step $step_number after $max_attempts attempt(s); rc=$rc"
            return "$rc"
        fi

        delay=$(compute_retry_backoff_delay "$attempt" "$retry_base_sec" "$retry_max_sec" "$retry_jitter_sec")
        echo "WARNING: mpiexec multi-node launch failed at step $step_number (attempt $attempt/$max_attempts, rc=$rc). Retrying in ${delay}s..."
        sleep "$delay"
    done

    echo "ERROR: mpiexec multi-node retry loop fell through at step $step_number."
    return 1
}

function postprocess_step_to_panguplasim_inputs() {
    step_number=$1
    log_file=$2
    POSTPROC_INPUT_FILES=""
    PANGUPLASIM_INPUT_FILES=""

    echo "Preparing Pangu-Plasim input files at step $step_number"
    for i in $(seq 0 $(($N_particles - 1))); do
        get_output_path $i $step_number
        get_pangu_plasim_postproc_path $i $step_number
        POSTPROC_INPUT_FILES+="$OUTPUT_PATH,"
        PANGUPLASIM_INPUT_FILES+="$PANGUPLASIM_IN_PATH,"
    done

    cmd="python3 $PATH_CODE/postprocessor2.0/postprocess_data.py --config $PATH_YAML_CONFIG \
    --input_file ${POSTPROC_INPUT_FILES%?} \
    --output_file ${PANGUPLASIM_INPUT_FILES%?} \
    --use_multiprocessing"
    echo "$cmd"
    eval "$cmd" &>> "$log_file"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: postprocess_data.py failed for step $step_number (see $log_file)"
        return $status
    fi
    echo "Prepared Pangu-Plasim input files at step $step_number"
}

function run_PanguPlasim() {
    step_number=$1
    echo "Num members forecast = $N_members_fcst"
    num_members=$N_members_fcst
    lead_time=$((dtau * (num_steps - step_number)))
    echo "lead_time = $lead_time"
    target_duration=$target_duration
    regions=($region)
    var="tas"
    PANGUPLASIM_OUTPUT_DIRS=""
    if ! postprocess_step_to_panguplasim_inputs $step_number $PATH_EXP/outerr_forecasts.log; then
        return 1
    fi

    # Get init_datetimes for all particles
    PANGUPLASIM_INPUT_FILES=""
    for i in $(seq 0 $(($N_particles - 1))); do
        get_pangu_plasim_postproc_path $i $step_number
        get_pangu_plasim_output_dir $i $step_number
        PANGUPLASIM_INPUT_FILES+="$PANGUPLASIM_IN_PATH,"
        PANGUPLASIM_OUTPUT_DIRS+="$PANGUPLASIM_OUT_DIR,"
    done
    IFS=',' read -ra input_files_array <<< "${PANGUPLASIM_INPUT_FILES%?}"
    declare -a init_datetimes_raw=()
    for input_file in "${input_files_array[@]}"; do
        dt=$(python $DIR_PANGUPLASIMFS/get_init_datetime.py "$input_file" | tail -n 1)
        init_datetimes_raw+=("$dt")
    done

    IFS=',' read -ra all_input_files <<< "${PANGUPLASIM_INPUT_FILES%,}"
    IFS=',' read -ra all_output_dirs <<< "${PANGUPLASIM_OUTPUT_DIRS%,}"

    basename_output="PanguPlasimFS_s"$step_number

    # Generate launcher scripts (once per experiment)
    generate_launcher_scripts

    # Distribute particles across all GPU nodes
    particles_per_node=$(( (N_particles + NNODES - 1) / NNODES ))
    echo "Distributing $N_particles particles across $NNODES GPU nodes (~$particles_per_node per node)"
    local launch_backend=${AIRES_PANGU_LAUNCH_BACKEND:-auto}
    if [ "$launch_backend" = "auto" ]; then
        if [ "$job_manager" = "PBS" ]; then
            launch_backend="mpiexec"
        else
            launch_backend="ssh"
        fi
    fi
    if [ "$launch_backend" != "ssh" ] && [ "$launch_backend" != "mpiexec" ]; then
        echo "WARNING: Invalid AIRES_PANGU_LAUNCH_BACKEND=$launch_backend; falling back to ssh"
        launch_backend="ssh"
    fi
    echo "Pangu launch backend: $launch_backend"

    if [ "$launch_backend" = "mpiexec" ]; then
        if ! command -v mpiexec >/dev/null 2>&1; then
            echo "ERROR: launch_backend=mpiexec but mpiexec was not found in PATH."
            return 1
        fi
        echo "Using PBS mpiexec backend for Pangu node launches; skipping SSH preflight."
    else
        ssh_identity_file=$(resolve_ssh_identity_file || true)
        if [ -n "$ssh_identity_file" ]; then
            echo "Using SSH identity file for node launches: $ssh_identity_file"
        else
            echo "WARNING: No SSH private key file found under ~/.ssh/id_*; relying on SSH defaults."
        fi
        if ! preflight_ssh_nodes "$ssh_identity_file"; then
            echo "ERROR: SSH preflight failed; skipping Pangu-Plasim launches for step $step_number."
            return 1
        fi

        build_ssh_opts ssh_opts "$ssh_identity_file"
    fi
    launch_pids=()
    launch_nodes=()
    launch_node_count=0

    for ni in $(seq 0 $((NNODES - 1))); do
        p_start=$((ni * particles_per_node))
        p_end=$(( (ni + 1) * particles_per_node ))
        [ $p_end -gt $N_particles ] && p_end=$N_particles
        [ $p_start -ge $N_particles ] && continue
        launch_node_count=$((launch_node_count + 1))

        NODE_INPUT_FILES=""
        NODE_OUTPUT_DIRS=""
        NODE_INIT_DATETIMES=""
        for j in $(seq $p_start $(($p_end - 1))); do
            NODE_INPUT_FILES+="${all_input_files[$j]},"
            NODE_OUTPUT_DIRS+="${all_output_dirs[$j]},"
            NODE_INIT_DATETIMES+="\"${init_datetimes_raw[$j]}\","
        done
        NODE_INPUT_FILES=${NODE_INPUT_FILES%,}
        NODE_OUTPUT_DIRS=${NODE_OUTPUT_DIRS%,}
        NODE_INIT_DATETIMES=${NODE_INIT_DATETIMES%,}

        PATH_CONFIG_NODE=$PATH_EXP/step_$step_number/config_PanguPlasim_node_$ni.json
        cat <<EOF > $PATH_CONFIG_NODE
        {
            "input_files": "$NODE_INPUT_FILES",
            "dirs_output": "$NODE_OUTPUT_DIRS",
            "basename_output": "$basename_output",
            "init_datetimes": [$NODE_INIT_DATETIMES],
            "num_members": $num_members,
            "lead_time": $lead_time,
            "target_duration": $target_duration,
            "PanguPlasim_perturb": $PanguPlasim_perturb,
            "var": "$var",
            "regions": [$(printf '"%s",' "${regions[@]}" | sed 's/,$//')],
            "job_manager": "$job_manager",
            "PATH_SCRATCH": "$PATH_SCRATCH",
            "PATH_REFERENCE_RUN_DIR": "$PATH_REFERENCE_RUN_DIR",
            "PATH_YAML_CONFIG": "$PATH_YAML_CONFIG",
            "PATH_REGIONS": "$PATH_REGIONS",
            "DIR_PANGUPLASIMFS": "$DIR_PANGUPLASIMFS",
            "run_num": "$run_num",
            "clim_file": "$clim_file",
            "threshold_json_file": "$threshold_json_file",
            "scorer": $scorer_json,
            "particle_start_index": $p_start
        }
EOF

        NODE_LOG=$PATH_EXP/outerr_forecasts_node_${ni}_step_${step_number}.log
        echo "Prepared Pangu-Plasim config on node slot ${nodes[$ni]} for particles $p_start-$((p_end-1))"
        launch_nodes+=("${nodes[$ni]}")
        if [ "$launch_backend" != "mpiexec" ]; then
            echo "Launching Pangu-Plasim on node ${nodes[$ni]} for particles $p_start-$((p_end-1))"
            launch_pangu_node_with_retry "${nodes[$ni]}" "$step_number" "$p_start" "$((p_end-1))" "$PATH_CONFIG_NODE" "$NODE_LOG" "$launch_backend" &
            launch_pids+=($!)
        fi
    done

    if [ "$launch_backend" = "mpiexec" ]; then
        if [ "$launch_node_count" -le 0 ]; then
            echo "ERROR: No launch nodes were prepared for step $step_number."
            return 1
        fi
        echo "Launching Pangu-Plasim via one mpiexec step across $launch_node_count node(s)"
        if ! launch_pangu_mpiexec_step_with_retry "$step_number" "$launch_node_count"; then
            echo "ERROR: One or more mpiexec launches failed at step $step_number."
            return 1
        fi
    else
        echo "Waiting for all GPU nodes to finish Pangu-Plasim forecasts..."
        launch_failed=0
        for idx in "${!launch_pids[@]}"; do
            pid="${launch_pids[$idx]}"
            node_name="${launch_nodes[$idx]}"
            if ! wait "$pid"; then
                echo "ERROR: $launch_backend launch failed on node $node_name for step $step_number."
                launch_failed=1
            fi
        done
        if [ "$launch_failed" -ne 0 ]; then
            echo "ERROR: One or more node launches failed at step $step_number."
            return 1
        fi
    fi
    echo "All GPU nodes finished Pangu-Plasim forecasts"

    for ni in $(seq 0 $((NNODES - 1))); do
        NODE_LOG=$PATH_EXP/outerr_forecasts_node_${ni}_step_${step_number}.log
        if [ -f "$NODE_LOG" ]; then
            echo "=== Node $ni (${nodes[$ni]}) step $step_number ===" >> $PATH_EXP/outerr_forecasts.log
            cat "$NODE_LOG" >> $PATH_EXP/outerr_forecasts.log
        fi
    done

    if ! validate_panguplasim_forecast_scores $step_number; then
        echo "ERROR: Forecast output validation failed at step $step_number."
        return 1
    fi
    return 0
}

function run_PFS() {
    step_number=$1
    echo "Num members forecast = $N_members_fcst"
    num_members=$N_members_fcst
    lead_time=$((dtau * (num_steps - step_number)))
    target_duration=$target_duration
    regions=($region)
    var="tas"

    local pfs_particle_parallel=${PFS_PARTICLE_PARALLEL:-8}
    if [ "$pfs_particle_parallel" -gt "$N_particles" ]; then
        pfs_particle_parallel=$N_particles
    fi
    if [ "$pfs_particle_parallel" -gt "$num_parallel" ]; then
        pfs_particle_parallel=$num_parallel
    fi
    local pfs_member_parallel=$((num_parallel / pfs_particle_parallel))
    if [ "$pfs_member_parallel" -lt 1 ]; then
        pfs_member_parallel=1
    fi
    echo "PFS particle parallelism: $pfs_particle_parallel concurrent, $pfs_member_parallel members each (total slots: $((pfs_particle_parallel * pfs_member_parallel)))"

    local pfs_postproc_script="$POSTPROC_SCRIPT"
    local pfs_postproc_config=""
    local pfs_postproc_nl="$PATH_POSTPROC_NL"
    local pfs_use_blocking_score_cfg
    pfs_use_blocking_score_cfg=$(jq -r '.use_blocking_score // "auto"' "$CONFIG_FILE")
    local pfs_use_blocking_score=false
    local _has_clim=false
    [ "$clim_file" != "null" ] && [ -n "$clim_file" ] && \
        [ "$threshold_json_file" != "null" ] && [ -n "$threshold_json_file" ] && _has_clim=true

    if [ "$pfs_use_blocking_score_cfg" = "false" ]; then
        echo "PFS: blocking scoring explicitly disabled (use_blocking_score=false)"
    elif [ "$_has_clim" = "true" ]; then
        pfs_postproc_script="$PATH_CODE/postprocessor2.0/postprocess_data.py"
        pfs_postproc_config="$PATH_CODE/forecast_modules/PFS/postproc_config/PFS_blocking_postproc.yaml"
        pfs_use_blocking_score=true
        echo "PFS: using postprocess_data.py for Z500 extraction (blocking scoring enabled)"
    elif [ "$pfs_use_blocking_score_cfg" = "true" ]; then
        echo "ERROR: use_blocking_score=true but clim_file or threshold_json_file is missing."
        echo "  clim_file=$clim_file"
        echo "  threshold_json_file=$threshold_json_file"
        echo "  Set use_blocking_score=false in config or provide both files."
        return 1
    else
        echo "PFS: using burn7 for legacy temperature scoring (no clim_file/threshold_json_file)"
    fi

    local pids=()
    for i in $(seq 0 $(($N_particles - 1))); do
        if [ $((i % pfs_particle_parallel)) -eq 0 ] && [ $i -ne 0 ]; then
            echo "Waiting for PFS particle batch to finish (at particle $i)"
            if ! wait_for_pids_or_fail "${pids[@]}"; then
                echo "ERROR: One or more PFS particles failed at step $step_number"
                return 1
            fi
            for logf in $PATH_EXP/step_$step_number/particle_*/forecast/pfs_particle.log; do
                if [ -f "$logf" ]; then
                    echo "=== PFS particle log: $logf ===" >> $PATH_EXP/outerr_forecasts.log
                    cat "$logf" >> $PATH_EXP/outerr_forecasts.log
                    rm "$logf"
                fi
            done
            pids=()
        fi

        local particle_dir=$PATH_EXP/step_$step_number/particle_$i
        local dir_output=$particle_dir/forecast
        local path_restart=$particle_dir/restart/restart_end
        local basename_output="PFS_s${step_number}_p${i}"
        local PATH_CONFIG_PFS=$dir_output/config_PFS.json
        local _clim_json="null"
        local _thresh_json="null"
        [ "$clim_file" != "null" ] && [ -n "$clim_file" ] && _clim_json="\"$clim_file\""
        [ "$threshold_json_file" != "null" ] && [ -n "$threshold_json_file" ] && _thresh_json="\"$threshold_json_file\""

        cat <<EOF > $PATH_CONFIG_PFS
{
    "path_restart": "$path_restart",
    "dir_output": "$dir_output",
    "basename_output": "$basename_output",
    "num_members": $num_members,
    "lead_time": $lead_time,
    "target_duration": $target_duration,
    "var": "$var",
    "regions": [$(printf '"%s",' "${regions[@]}" | sed 's/,$//')],
    "job_manager": "$job_manager",
    "PATH_SCRATCH": "$PATH_SCRATCH",
    "PATH_REFERENCE_RUN_DIR": "$PATH_REFERENCE_RUN_DIR",
    "PATH_POSTPROC_NL": "$pfs_postproc_nl",
    "POSTPROC_SCRIPT": "$pfs_postproc_script",
    "POSTPROC_CONFIG": "$pfs_postproc_config",
    "PATH_REGIONS": "$PATH_REGIONS",
    "DIR_PFS": "$DIR_PFS",
    "use_blocking_score": $pfs_use_blocking_score,
    "clim_file": $_clim_json,
    "threshold_json_file": $_thresh_json,
    "scorer": $scorer_json
}
EOF
        echo "PFS particle $i"
        local slot_offset=$(( (i % pfs_particle_parallel) * pfs_member_parallel ))
        PFS_MAX_PARALLEL=$pfs_member_parallel PFS_SLOT_OFFSET=$slot_offset python3 $DIR_PFS/PFS.py \
            --config $PATH_CONFIG_PFS &> $dir_output/pfs_particle.log &
        pids+=($!)
        sleep 0.5
    done
    if [ "${#pids[@]}" -gt 0 ]; then
        echo "Waiting for final PFS particle batch to finish"
        if ! wait_for_pids_or_fail "${pids[@]}"; then
            echo "ERROR: One or more PFS particles failed at step $step_number"
            return 1
        fi
        for logf in $PATH_EXP/step_$step_number/particle_*/forecast/pfs_particle.log; do
            if [ -f "$logf" ]; then
                echo "=== PFS particle log: $logf ===" >> $PATH_EXP/outerr_forecasts.log
                cat "$logf" >> $PATH_EXP/outerr_forecasts.log
                rm "$logf"
            fi
        done
    fi
}

function run_forecasts() {
    step_number=$1
    echo "Running forecasts for step $step_number"
    if [ $forecast_method = "PFS" ]; then
        echo "Running PFS forecasts"
        run_PFS $step_number
        echo "PFS forecasts finished"
    elif [ $forecast_method = "Pangu-Plasim" ]; then
        echo "Running Pangu-Plasim forecasts"
        if ! run_PanguPlasim $step_number; then
            echo "ERROR: Pangu-Plasim forecast stage failed at step $step_number"
            return 1
        fi
        echo "Pangu-Plasim forecasts finished"
    fi
    if [ $forecast_method = "control" ]; then
        echo "Control forecast method selected, doing nothing"
    fi
    if [ $forecast_method = "control_random" ]; then
        echo "Control random forecast method selected, doing nothing"
    fi
    if [ $forecast_method = "Persistence" ]; then
        echo "Persistence forecast method selected, doing nothing"
    fi
    return 0
}
```

**Step 2: Verify syntax**

Run: `bash -n AI-RES/RES/lib/lib_forecast.sh`
Expected: No output

**Step 3: Commit**

```bash
git add AI-RES/RES/lib/lib_forecast.sh
git commit -m "refactor: extract lib_forecast.sh with hardened launcher scripts

Fixes CUDA_VISIBLE_DEVICES UUID propagation bug by unsetting it in
generated launcher scripts and activating conda locally on each node."
```

---

### Task 5: Rewrite QDMC.sh as thin main driver

**Files:**
- Modify: `AI-RES/RES/QDMC.sh`

**Step 1: Replace QDMC.sh with the thin orchestrator**

The new QDMC.sh sources the 4 lib files, parses the config (section 3 from original), and runs the main loop (section 5 from original). Changes from original:
- Sources `lib/cluster_setup.sh`, `lib/lib_utils.sh`, `lib/lib_plasim.sh`, `lib/lib_forecast.sh`
- Removes all function definitions (now in lib files)
- Removes `run_particles_one_step_doing_nothing` calls from the main loop
- Removes the `debug_mode` fake-run branches (keeps `debug_mode` config parsing for write_used_config)
- Removes redundant `echo "DIR_PANGUPLASIM_MODEL = ..."` at line 359
- All else is preserved verbatim

The new QDMC.sh structure:
1. Header / shebang (same)
2. `CONFIG_FILE=$1`
3. `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`
4. `source "$SCRIPT_DIR/lib/cluster_setup.sh"`
5. Config parsing (lines 142-367 minus redundant echo, minus function defs)
6. `source "$SCRIPT_DIR/lib/lib_utils.sh"`
7. `source "$SCRIPT_DIR/lib/lib_plasim.sh"`
8. `source "$SCRIPT_DIR/lib/lib_forecast.sh"`
9. Main loop (lines 1455-1578 with `debug_mode` fake-run branches removed)

**Step 2: Verify syntax**

Run: `bash -n AI-RES/RES/QDMC.sh`
Expected: No output

**Step 3: Commit**

```bash
git add AI-RES/RES/QDMC.sh
git commit -m "refactor: rewrite QDMC.sh as thin main driver sourcing lib/ modules

Removes fake-run debug path and redundant echo. All functions now
live in lib/{cluster_setup,lib_utils,lib_plasim,lib_forecast}.sh."
```

---

### Task 6: Shellcheck all files

**Files:**
- Check: `AI-RES/RES/QDMC.sh`, `AI-RES/RES/lib/cluster_setup.sh`, `AI-RES/RES/lib/lib_utils.sh`, `AI-RES/RES/lib/lib_plasim.sh`, `AI-RES/RES/lib/lib_forecast.sh`

**Step 1: Run shellcheck**

Run: `shellcheck -x AI-RES/RES/QDMC.sh AI-RES/RES/lib/*.sh`

Note: `-x` follows source directives. Expect some warnings about:
- SC2086 (double quote to prevent globbing) — many existing unquoted variables; fix only new code, don't touch existing behavior
- SC2034 (unused variables) — some variables are set for use by sourced files; add `# shellcheck disable=SC2034` where appropriate

**Step 2: Fix any errors (not warnings) found by shellcheck**

**Step 3: Commit fixes if any**

```bash
git add AI-RES/RES/QDMC.sh AI-RES/RES/lib/*.sh
git commit -m "fix: address shellcheck errors in refactored QDMC files"
```

---

### Task 7: Produce responsibility mapping table and verify completeness

**Step 1: Produce the mapping table**

Compare every function and section from the original 1,578-line QDMC.sh against the new files. Confirm every original responsibility is accounted for with only the approved removals.

**Step 2: Cross-check**

- Verify `run_particles_one_step_doing_nothing()` is absent from all new files
- Verify `debug_mode` fake-run branches are absent from main loop
- Verify `unset CUDA_VISIBLE_DEVICES` is present in both launcher heredocs
- Verify `ml conda && conda activate aires` is present in both launcher heredocs
- Verify `build_ssh_opts` is used everywhere SSH options were previously duplicated
- Verify `validate_retry_params` is used in all 4 retry locations

**Step 3: Commit the mapping as a verification record**

Add a comment block at the top of the design doc or in the commit message confirming the mapping check passed.

---

### Task 8: Smoke test on Derecho (multi-node)

**Step 1: Submit the smoke test**

Run: `qsub` the PBS script that invokes `bash RES/QDMC.sh RES/experiments/configs/derecho/AIRES/EXP15_heatwave_AIRES_smoke.json` with at least 4 GPU nodes.

**Step 2: Monitor job completion**

Wait for the job to complete. Check `qstat` for status.

**Step 3: Output-level verification**

After job completes, check:

```bash
# 1. Run reaches step 4 Pangu forecasting
grep "Running Pangu-Plasim forecasts" <exp_dir>/outerr_forecasts.log

# 2. No Traceback or ERROR in logs
grep -i "traceback\|^ERROR" <exp_dir>/*.log

# 3. World size from Cuda: 4 on all nodes
grep "World size from Cuda" <exp_dir>/outerr_forecasts_node_*_step_4.log

# 4. Each node reports local GPU UUIDs (should differ across nodes)
grep "CUDA_VISIBLE_DEVICES" <exp_dir>/outerr_forecasts_node_*_step_4.log

# 5. Forecast score files exist
ls <exp_dir>/step_4/particle_*/forecast/PanguPlasimFS_s4_p*_A_France.npy | wc -l
# Expected: 40 (N_particles)

# 6. validate_panguplasim_forecast_scores passed
grep "Validated Pangu-Plasim score files" <exp_dir>/outerr_forecasts.log
```

**Step 4: Commit verification results**

If all checks pass, record in commit message or design doc.

---

### Task 9 (optional): Non-Pangu regression check

Only if Task 5 modified shared dispatch or main-loop code beyond removing the fake-run path.

**Step 1: Run a cheap control experiment**

Submit a control-mode experiment with small N_particles (e.g., 10) and few steps to verify the non-GPU path still works.

**Step 2: Verify completion**

Check that the experiment finishes without errors and produces expected output files.
