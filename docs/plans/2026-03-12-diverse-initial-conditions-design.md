# Design: `diverse_initial_conditions` Option

**Date:** 2026-03-12
**Status:** Approved

## Background

In the current AI-RES experiment (EXP15), all walkers start from the same single restart file
`MOST.0069-11-24_00:00:00`. The initialization date is derived as `EVENT_DATE - length_simu`
(e.g., `0069-12-24 - 30 days = 0069-11-24`).

The new `diverse_initial_conditions` option allows walkers to start from 100 different years on
the same calendar date (Nov 24 in EXP15), providing a climatologically diverse ensemble.

## Requirements

- New JSON config boolean field `diverse_initial_conditions`.
- When `true`: walkers initialized from years 0006–0105 on the computed init month-day.
- When `false`: existing single-date behavior preserved.
- `N_particles` must be a multiple of 100 when `true`; error otherwise.
- Default is `true` (code-level); all existing configs must be updated explicitly.

## Config Schema

Add to every experiment config JSON:

```json
"diverse_initial_conditions": false
```

- All existing configs (excluding `old/`) → `false` (preserve current behavior).
- New diverse-IC configs → `true`.
- `old/` configs: intentionally not patched; they are archived and unsupported for new runs.
- Absent field defaults to `true` in code (via `jq // true`); explicit values are required in all
  active configs to avoid ambiguity.
- Must be a JSON boolean (`true`/`false`), not a string. jq type-checks and errors on wrong type.

## Changes to `QDMC.sh`

### 1. Parse and type-check the flag

```bash
diverse_initial_conditions=$(jq -r '
  if (.diverse_initial_conditions != null) and ((.diverse_initial_conditions | type) != "boolean")
  then error("diverse_initial_conditions must be a JSON boolean, got " + (.diverse_initial_conditions | type))
  else (.diverse_initial_conditions // true)
  end' "$CONFIG_FILE")
```

### 2. Validate flag value

```bash
if [[ "$diverse_initial_conditions" != "true" && "$diverse_initial_conditions" != "false" ]]; then
    echo "ERROR: diverse_initial_conditions must be 'true' or 'false', got '$diverse_initial_conditions'" >&2
    exit 1
fi
```

### 3. Validate N_particles when diverse mode is active

```bash
if [[ "$diverse_initial_conditions" == "true" ]] && (( N_particles % 100 != 0 )); then
    echo "ERROR: diverse_initial_conditions=true requires N_particles to be a multiple of 100 (got $N_particles)" >&2
    exit 1
fi
```

### 4. Resolve IC script to canonical absolute path

```bash
create_ICs_script_abs="$PATH_CODE/$create_ICs_script"
```

Used for both capability probe and actual invocation. Avoids cwd-dependent failures.

### 5. Capability probe before calling the IC script

```bash
if [[ "$diverse_initial_conditions" == "true" ]]; then
    caps=$(bash "$create_ICs_script_abs" --capabilities 2>/dev/null || true)
    if ! grep -qw "diverse_initial_conditions" <<< "$caps"; then
        echo "ERROR: create_ICs_script '$create_ICs_script' does not support diverse_initial_conditions" >&2
        exit 1
    fi
fi
```

### 6. Pass flag as 5th argument

```bash
list_ICs=$(bash "$create_ICs_script_abs" "$N_particles" "$region" "$length_simu" "$PATH_SCRATCH" "$diverse_initial_conditions")
list_ICs=($list_ICs)
```

### 7. Assert list length matches N_particles

```bash
if [[ ${#list_ICs[@]} -ne $N_particles ]]; then
    echo "ERROR: IC script returned ${#list_ICs[@]} paths, expected $N_particles" >&2
    exit 1
fi
```

### 8. Include flag in written used-config JSON

Add `diverse_initial_conditions` to the `write_used_config` function output for reproducibility.

### 9. Echo the flag in the experiment parameters log

Add `echo "diverse_initial_conditions = $diverse_initial_conditions"` alongside other parameters.

## Changes to `blocking_1_ICs_0069_12_24.sh`

### Capability probe handler (near top, before argument parsing)

```bash
if [[ "${1:-}" == "--capabilities" ]]; then
    echo "diverse_initial_conditions"
    exit 0
fi
```

Contract: prints one capability token per line, nothing else to stdout.

### Accept and validate 5th argument

```bash
diverse_initial_conditions=${5:-false}

if [[ "$diverse_initial_conditions" != "true" && "$diverse_initial_conditions" != "false" ]]; then
    echo "ERROR: argument 5 (diverse_initial_conditions) must be 'true' or 'false', got '$diverse_initial_conditions'" >&2
    exit 1
fi
```

### Safety guard for N_particles % 100

```bash
if [[ "$diverse_initial_conditions" == "true" ]] && (( num_particles % 100 != 0 )); then
    echo "ERROR: diverse_initial_conditions=true requires num_particles % 100 == 0 (got $num_particles)" >&2
    exit 1
fi
```

### Diverse IC generation logic

After computing `init_month_fmt` and `init_day_fmt` (unchanged from current arithmetic):

```bash
if [[ "$diverse_initial_conditions" == "true" ]]; then
    walkers_per_year=$(( num_particles / 100 ))
    list_ICs=()
    for year in $(seq 6 105); do
        year_fmt=$(printf "%04d" $year)
        INIT_DATE="${year_fmt}-${init_month_fmt}-${init_day_fmt}"
        path_IC="${RESTART_DIR}/MOST.${INIT_DATE}_00:00:00"
        if [[ ! -f "$path_IC" ]]; then
            echo "ERROR: Restart file not found: $path_IC" >&2
            exit 1
        fi
        for (( w=0; w<walkers_per_year; w++ )); do
            list_ICs+=("$path_IC")
        done
    done
else
    # existing single-date logic
fi
```

Assignment order: walkers `0..(k-1)` → year 0006, walkers `k..(2k-1)` → year 0007, etc.
where `k = N_particles / 100`.

### Assert list length before output

```bash
if [[ ${#list_ICs[@]} -ne $num_particles ]]; then
    echo "ERROR: built ${#list_ICs[@]} IC paths, expected $num_particles" >&2
    exit 1
fi
```

### Stdout discipline

All `echo` diagnostics go to stderr (`>&2`). Only the final IC list goes to stdout:

```bash
echo "${list_ICs[@]}"
```

## Existing Config Updates

All active configs (excluding `old/`) receive `"diverse_initial_conditions": false`:

- `configs/derecho/AIRES/EXP0_AIRES.json` through `EXP28_AIRES.json`, `EXP33_AIRES.json`,
  `EXP15_AIRES_FALLBACK_NONBLOCKED.json`, `EXP_PARTIAL_DEMO.json`
- `configs/derecho/AI-DNS/EXP15_AI-DNS.json`
- `configs/derecho/PFS/EXP15_PFS.json`, `EXP15_PFS_pilot10.json`, `EXP21_PFS.json`, `EXP3_PFS.json`
- `configs/derecho/Persistence/EXP15_Persistence.json`
- `configs/derecho/BASE0_ctrl.json` through `BASE6_ctrl.json`
- `configs/derecho/TEST_AIRES_amaury.json`, `TEST_PanguPlasim.json`
- `configs/Stampede3/*.json`
- `configs/test_PFS_derecho.json`, `configs/test_PFS_Stampede3.json`

`old/` configs: not updated; archived and unsupported for new runs.

## Year Selection

- Years: 0006–0105 (first 100 years with available Nov-24 restart files out of 110 available).
- Block assignment: walkers `[i*k .. (i+1)*k - 1]` → year `0006 + i`, where `k = N_particles / 100`.
- Init month-day: derived from `EVENT_DATE - length_simu` (same calendar arithmetic as current).
- Year component of `INIT_DATE` is discarded in diverse mode.

## `old/` Configs

`old/` configs are intentionally not patched. They are archived experiments not intended to be
re-run. If invoked with new code, the `// true` default would apply and they would likely fail the
`N_particles % 100` validation.
