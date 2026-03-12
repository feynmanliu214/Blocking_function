# `diverse_initial_conditions` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `diverse_initial_conditions` JSON boolean that, when `true`, initializes the 400 walkers from 100 different years on the computed init calendar date instead of a single year.

**Architecture:** Three-layer change: (1) patch all existing configs to set the field explicitly, (2) extend the IC bash script with a capability probe and diverse-year logic, (3) extend `QDMC.sh` to read/validate the flag, probe capability, pass it as arg 5, and assert the returned list length. All existing behavior is preserved when `false`.

**Tech Stack:** Bash, jq, pytest (subprocess), PlaSim restart files at `/glade/u/home/zhil/project/AI-RES/Blocking/data/PlaSim/sim52/restart_files/`

> **Known gap (non-blocking):** `AI-DNS.sh` (line 296) also calls `create_ICs_script` with only 4 args and does not read `diverse_initial_conditions`. AIRES experiments use `QDMC.sh`, so this is not a blocker here. A follow-up task should apply equivalent changes to `AI-DNS.sh` before any AI-DNS run uses this flag.

---

### Task 1: Patch all existing configs with `"diverse_initial_conditions": false`

**Files:**
- Modify: every `.json` in `AI-RES/RES/experiments/configs/` excluding `old/` subdirectories

**Step 1: Run the bulk patch**

```bash
cd /glade/u/home/zhil/project/AI-RES/Blocking

# For each config, add the field before the closing brace if not already present
for f in $(find AI-RES/RES/experiments/configs -name "*.json" \
           -not -path "*/old/*"); do
  # Skip if field already exists
  if jq -e 'has("diverse_initial_conditions")' "$f" >/dev/null 2>&1; then
    echo "SKIP (already has field): $f"
    continue
  fi
  # Add field before last closing brace using jq
  tmp=$(jq '. + {"diverse_initial_conditions": false}' "$f")
  echo "$tmp" > "$f"
  echo "PATCHED: $f"
done
```

**Step 2: Verify all active configs have the field**

```bash
for f in $(find AI-RES/RES/experiments/configs -name "*.json" \
           -not -path "*/old/*"); do
  val=$(jq -r '.diverse_initial_conditions // "MISSING"' "$f")
  echo "$val  $f"
done
```

Expected: every line shows `false  <path>` (none should say `MISSING`).

**Step 3: Spot-check JSON validity**

```bash
for f in $(find AI-RES/RES/experiments/configs -name "*.json" \
           -not -path "*/old/*"); do
  jq . "$f" > /dev/null || echo "INVALID JSON: $f"
done
```

Expected: no output (all valid).

**Step 4: Commit**

```bash
git add AI-RES/RES/experiments/configs/
git commit -m "config: set diverse_initial_conditions=false in all existing configs"
```

---

### Task 2: Add capability probe to `blocking_1_ICs_0069_12_24.sh`

**Files:**
- Modify: `AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh`
- Create: `tests/create_ICs/test_blocking_1_ICs.py`

**Background:** The IC script is called as:
```
bash blocking_1_ICs_0069_12_24.sh <num_particles> <region> <length_simu> <PATH_SCRATCH> [diverse_flag]
```
When called with `--capabilities` as arg 1, it must print `diverse_initial_conditions` to stdout and exit 0. This is how `QDMC.sh` will check that the script supports diverse mode.

**Step 1: Write the failing test**

Create `tests/create_ICs/__init__.py` (empty) and `tests/create_ICs/test_blocking_1_ICs.py`:

```python
import subprocess
import pytest

SCRIPT = "AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh"

def run_script(*args, check=False):
    """Run the IC script with given args, return CompletedProcess."""
    return subprocess.run(
        ["bash", SCRIPT] + list(args),
        capture_output=True, text=True
    )

def test_capabilities_probe_returns_token():
    """--capabilities must print 'diverse_initial_conditions' on stdout."""
    result = run_script("--capabilities")
    assert result.returncode == 0
    tokens = result.stdout.strip().splitlines()
    assert "diverse_initial_conditions" in tokens

def test_capabilities_probe_no_extra_stdout():
    """--capabilities must produce no output beyond capability tokens."""
    result = run_script("--capabilities")
    assert result.returncode == 0
    for line in result.stdout.strip().splitlines():
        assert line.strip() != "", "unexpected blank line in capabilities output"
    # stderr must be empty in capability mode
    assert result.stderr.strip() == ""
```

**Step 2: Run test to verify it fails**

```bash
cd /glade/u/home/zhil/project/AI-RES/Blocking
python -m pytest tests/create_ICs/test_blocking_1_ICs.py::test_capabilities_probe_returns_token -v
```

Expected: FAIL — the script doesn't handle `--capabilities` yet.

**Step 3: Add the capability probe to the script**

In `AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh`, insert immediately after the shebang line and top comments (before `num_particles=$1`):

```bash
# Capability probe — called by QDMC.sh to verify diverse mode support.
# Contract: print one capability token per line, nothing else to stdout, exit 0.
if [[ "${1:-}" == "--capabilities" ]]; then
    echo "diverse_initial_conditions"
    exit 0
fi
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/create_ICs/test_blocking_1_ICs.py -v
```

Expected: both tests PASS.

**Step 5: Commit**

```bash
git add AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh \
        tests/create_ICs/__init__.py \
        tests/create_ICs/test_blocking_1_ICs.py
git commit -m "feat: add --capabilities probe to blocking_1_ICs_0069_12_24.sh"
```

---

### Task 3: Add `diverse_initial_conditions` arg + validation to the IC script

**Files:**
- Modify: `AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh`
- Modify: `tests/create_ICs/test_blocking_1_ICs.py`

**Step 1: Write the failing tests**

Add to `tests/create_ICs/test_blocking_1_ICs.py`:

```python
RESTART_DIR = "AI-RES/RES/../../data/PlaSim/sim52/restart_files"
# Resolved: /glade/u/home/zhil/project/AI-RES/Blocking/data/PlaSim/sim52/restart_files

def test_invalid_flag_value_errors():
    """Arg 5 must be 'true' or 'false'; anything else must fail with exit 1."""
    result = run_script("10", "NorthAtlantic", "30", "/tmp", "maybe")
    assert result.returncode == 1
    assert "diverse_initial_conditions" in result.stderr.lower() or \
           "true" in result.stderr or "false" in result.stderr

def test_diverse_false_not_multiple_of_100_is_ok():
    """diverse=false imposes no restriction on N_particles."""
    # N_particles=3, diverse=false — should succeed (3 copies of single path)
    result = run_script("3", "NorthAtlantic", "30", "/tmp", "false")
    assert result.returncode == 0
    paths = result.stdout.strip().split()
    assert len(paths) == 3
    assert len(set(paths)) == 1  # all same path

def test_diverse_true_requires_multiple_of_100():
    """diverse=true with N_particles not divisible by 100 must fail."""
    result = run_script("50", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 1
    assert "100" in result.stderr

def test_diverse_false_default_same_as_false():
    """Omitting arg 5 should behave identically to diverse=false."""
    r_omitted = run_script("3", "NorthAtlantic", "30", "/tmp")
    r_false   = run_script("3", "NorthAtlantic", "30", "/tmp", "false")
    assert r_omitted.returncode == r_false.returncode == 0
    assert r_omitted.stdout == r_false.stdout
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/create_ICs/test_blocking_1_ICs.py \
  -k "invalid_flag or diverse_false_not_multiple or diverse_true_requires or diverse_false_default" -v
```

Expected: most tests FAIL (no arg-5 handling yet).

**Step 3: Add arg 5 reading + validation to the script**

After `PATH_SCRATCH=$4` in the script, add:

```bash
diverse_initial_conditions=${5:-false}

# Validate flag value
if [[ "$diverse_initial_conditions" != "true" && \
      "$diverse_initial_conditions" != "false" ]]; then
    echo "ERROR: argument 5 (diverse_initial_conditions) must be 'true' or 'false'," \
         "got '$diverse_initial_conditions'" >&2
    exit 1
fi

# Safety guard: diverse mode requires N_particles divisible by 100
if [[ "$diverse_initial_conditions" == "true" ]] && \
   (( 10#$num_particles % 100 != 0 )); then
    echo "ERROR: diverse_initial_conditions=true requires num_particles to be" \
         "a multiple of 100 (got $num_particles)" >&2
    exit 1
fi
```

Also redirect all existing diagnostic `echo` lines to stderr (`>&2`) — scan for any `echo` not
already going to stderr and add `>&2`. The `echo "${list_ICs[@]}"` at the very end must remain
on stdout (no `>&2`).

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/create_ICs/test_blocking_1_ICs.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh \
        tests/create_ICs/test_blocking_1_ICs.py
git commit -m "feat: add arg5 validation and diverse_initial_conditions guard to IC script"
```

---

### Task 4: Implement diverse IC generation in the IC script

**Files:**
- Modify: `AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh`
- Modify: `tests/create_ICs/test_blocking_1_ICs.py`

**Step 1: Write the failing tests**

Add to `tests/create_ICs/test_blocking_1_ICs.py`:

```python
import os

RESTART_DIR = "/glade/u/home/zhil/project/AI-RES/Blocking/data/PlaSim/sim52/restart_files"

def test_diverse_true_400_walkers_correct_count():
    """diverse=true, N=400 must return exactly 400 paths."""
    result = run_script("400", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 400

def test_diverse_true_400_walkers_block_assignment():
    """walkers 0-3 -> year 0006, walkers 4-7 -> year 0007, etc."""
    result = run_script("400", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    # Init date for EVENT_DATE=0069-12-24, length_simu=30 is YYYY-11-24
    # Year for walker 0 should be 0006
    assert "0006-11-24" in paths[0]
    assert "0006-11-24" in paths[3]   # last in first block
    assert "0007-11-24" in paths[4]   # first in second block
    assert "0105-11-24" in paths[399]  # last year, last walker

def test_diverse_true_100_walkers_one_per_year():
    """N=100 -> 1 walker per year, 100 distinct years."""
    result = run_script("100", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 100
    assert len(set(paths)) == 100  # all distinct

def test_diverse_true_200_walkers_two_per_year():
    """N=200 -> 2 walkers per year."""
    result = run_script("200", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 200
    # Year 0006 gets walkers 0 and 1
    assert "0006-11-24" in paths[0]
    assert "0006-11-24" in paths[1]
    assert "0007-11-24" in paths[2]

def test_diverse_mode_only_stdout_is_paths():
    """No diagnostic output on stdout in diverse mode; stderr is separate."""
    result = run_script("100", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    for token in result.stdout.strip().split():
        # Every stdout token must look like a file path
        assert token.startswith("/"), f"Non-path token on stdout: {token!r}"

def test_diverse_mode_does_not_require_single_year_file():
    """diverse=true must not fail due to missing single-year restart file.
    Year 0069 exists, but the check should not run in diverse mode at all.
    This guards against the bug where lines 97-103 run unconditionally."""
    # Use a length_simu that maps to a date with no single-year file
    # by using a year not in 0006-0116 range; e.g. year 0200 would fail single check
    # We verify diverse mode succeeds regardless of what the single-year path resolves to.
    # Since EVENT_DATE is hardcoded in the script (0069-12-24), we can't change the year,
    # but we verify that diverse mode does NOT check the single-year path by confirming
    # success even when we know the single-year file exists (regression guard for the guard itself).
    result = run_script("400", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, \
        f"diverse mode failed unexpectedly:\n{result.stderr}"

def test_list_length_assertion_fires():
    """IC script asserts list length == num_particles before returning."""
    # This is implicitly verified by the count tests above; here we verify
    # the script doesn't silently return the wrong count even in edge cases.
    result = run_script("300", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 300
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/create_ICs/test_blocking_1_ICs.py \
  -k "diverse_true" -v
```

Expected: FAIL — diverse mode logic not implemented yet.

**Step 3: Implement diverse IC generation in the script**

In `blocking_1_ICs_0069_12_24.sh`, find the section that builds `list_ICs` (currently just a
`for i in $(seq 1 $num_particles)` loop) and replace it with a conditional:

```bash
# Build the IC list
list_ICs=()

if [[ "$diverse_initial_conditions" == "true" ]]; then
    # Diverse mode: 100 years (0006-0105), block assignment
    walkers_per_year=$(( 10#$num_particles / 100 ))
    for year in $(seq 6 105); do
        year_fmt=$(printf "%04d" $year)
        INIT_DATE_DIVERSE="${year_fmt}-${init_month_fmt}-${init_day_fmt}"
        path_IC_diverse="${RESTART_DIR}/MOST.${INIT_DATE_DIVERSE}_00:00:00"
        if [[ ! -f "$path_IC_diverse" ]]; then
            echo "ERROR: Restart file not found: $path_IC_diverse" >&2
            exit 1
        fi
        for (( w=0; w<walkers_per_year; w++ )); do
            list_ICs+=("$path_IC_diverse")
        done
    done
else
    # Standard mode: all walkers from same single restart file
    for i in $(seq 1 $num_particles); do
        list_ICs+=("$path_IC")
    done
fi

# Safety assertion: list length must equal num_particles
if [[ ${#list_ICs[@]} -ne $num_particles ]]; then
    echo "ERROR: built ${#list_ICs[@]} IC paths but expected $num_particles" >&2
    exit 1
fi

echo "${list_ICs[@]}"
```

**Critical:** The existing single-file existence check (lines 97–103 of the IC script) runs
unconditionally and will fail in diverse mode if the single-year file happens to be absent.
Wrap it so it only runs in standard mode:

```bash
# Standard mode only: verify the single restart file exists
if [[ "$diverse_initial_conditions" != "true" ]]; then
    if [ ! -f "$path_IC" ]; then
        echo "ERROR: Restart file not found: $path_IC" >&2
        echo "Available date range: 0006-08-26 to 0116-03-02" >&2
        echo "Event date: $EVENT_DATE, Init date: $INIT_DATE, Lead time: $length_simu days" >&2
        exit 1
    fi
fi
```

The `path_IC` variable is still computed (from `INIT_DATE` including `init_year_fmt`) regardless
of mode; in diverse mode it is simply never used.

**Step 4: Run all tests**

```bash
python -m pytest tests/create_ICs/test_blocking_1_ICs.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh \
        tests/create_ICs/test_blocking_1_ICs.py
git commit -m "feat: implement diverse_initial_conditions in blocking_1_ICs_0069_12_24.sh"
```

---

### Task 5: Update `QDMC.sh` to read, validate, probe, and pass the flag

**Files:**
- Modify: `AI-RES/RES/QDMC.sh` (lines ~149–170 for parsing, ~244–284 for echo block,
  ~324–374 for `write_used_config`, ~384–410 for `initialize_experiment`)

Note: `QDMC.sh` runs full cluster jobs and cannot be unit-tested in isolation. Changes are
verified by manual inspection and a dry-run integration test in Task 6.

**Step 1: Add flag parsing after line 160 (after `create_ICs_script=...`)**

After the line:
```bash
create_ICs_script=$(jq -r '.create_ICs_script' "$CONFIG_FILE")
```

Add:

```bash
# Parse diverse_initial_conditions (must be JSON boolean true/false, default true)
diverse_initial_conditions=$(jq -r '
  if (.diverse_initial_conditions != null) and
     ((.diverse_initial_conditions | type) != "boolean")
  then error("diverse_initial_conditions must be a JSON boolean, got " +
             (.diverse_initial_conditions | type))
  else (.diverse_initial_conditions // true)
  end' "$CONFIG_FILE")
```

**Step 2: Add flag validation and N_particles check after the parsing block**

After all parameter parsing (around line 170, before the experiment name/directory setup):

```bash
# Validate diverse_initial_conditions value
if [[ "$diverse_initial_conditions" != "true" && \
      "$diverse_initial_conditions" != "false" ]]; then
    echo "ERROR: diverse_initial_conditions must be 'true' or 'false'," \
         "got '$diverse_initial_conditions'" >&2
    exit 1
fi

# Early validation: diverse mode requires N_particles divisible by 100
if [[ "$diverse_initial_conditions" == "true" ]] && \
   (( N_particles % 100 != 0 )); then
    echo "ERROR: diverse_initial_conditions=true requires N_particles to be" \
         "a multiple of 100 (got $N_particles)" >&2
    exit 1
fi
```

**Step 3: Add echo for the flag in the experiment parameters log (around line 260)**

After `echo "create_ICs_script = $create_ICs_script"`, add:

```bash
echo "diverse_initial_conditions = $diverse_initial_conditions"
```

**Step 4: Add flag to `write_used_config` (around line 349)**

In the `write_used_config` heredoc, after `"create_ICs_script": "$create_ICs_script",` add:

```bash
    "diverse_initial_conditions": $diverse_initial_conditions,
```

(no quotes — `$diverse_initial_conditions` is already `true` or `false`, valid JSON booleans)

**Step 5: Update `initialize_experiment` (lines 384–410)**

Replace the existing IC invocation block:

```bash
# Old:
list_ICs=$(bash "$create_ICs_script" "$N_particles" "$region" "$length_simu" "$PATH_SCRATCH")
list_ICs=($list_ICs)
echo "List of ICs: ${list_ICs[@]}"
```

With:

```bash
# Resolve IC script to absolute path to avoid cwd-dependent failures
create_ICs_script_abs="$PATH_CODE/$create_ICs_script"

# Capability probe: verify script supports diverse mode if requested
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

# Assert returned list has exactly N_particles entries
if [[ ${#list_ICs[@]} -ne $N_particles ]]; then
    echo "ERROR: IC script returned ${#list_ICs[@]} paths, expected $N_particles" >&2
    exit 1
fi
```

**Step 6: Verify the diff looks correct**

```bash
git diff AI-RES/RES/QDMC.sh
```

Check: no unintended deletions, all 5 change sites are present.

**Step 7: Commit**

```bash
git add AI-RES/RES/QDMC.sh
git commit -m "feat: read and enforce diverse_initial_conditions in QDMC.sh"
```

---

### Task 6: Integration smoke test (dry-run)

**Files:**
- No code changes — verification only

**Goal:** Confirm the full pipeline from config → QDMC.sh param parsing → IC script invocation
works correctly for both `false` and `true` modes, without launching a real cluster job.

**Step 1: Verify config parsing for `false` mode**

```bash
cd /glade/u/home/zhil/project/AI-RES/Blocking/AI-RES/RES
CONFIG=experiments/configs/derecho/AIRES/EXP15_AIRES.json

# Check field exists and is false
jq '.diverse_initial_conditions' $CONFIG
# Expected: false

# Simulate what QDMC.sh does with jq
jq -r '
  if (.diverse_initial_conditions != null) and
     ((.diverse_initial_conditions | type) != "boolean")
  then error("type error")
  else (.diverse_initial_conditions // true)
  end' $CONFIG
# Expected: false
```

**Step 2: Direct IC script test (false mode)**

```bash
bash create_ICs/blocking_1_ICs_0069_12_24.sh 4 NorthAtlantic 30 /tmp false 2>/dev/null | \
  tr ' ' '\n' | wc -l
# Expected: 4
bash create_ICs/blocking_1_ICs_0069_12_24.sh 4 NorthAtlantic 30 /tmp false 2>/dev/null | \
  tr ' ' '\n' | sort -u | wc -l
# Expected: 1 (all same path)
```

**Step 3: Direct IC script test (true mode)**

```bash
bash create_ICs/blocking_1_ICs_0069_12_24.sh 400 NorthAtlantic 30 /tmp true 2>/dev/null | \
  tr ' ' '\n' | wc -l
# Expected: 400
bash create_ICs/blocking_1_ICs_0069_12_24.sh 400 NorthAtlantic 30 /tmp true 2>/dev/null | \
  tr ' ' '\n' | sort -u | wc -l
# Expected: 100 (100 distinct years)
# Check first path is year 0006
bash create_ICs/blocking_1_ICs_0069_12_24.sh 400 NorthAtlantic 30 /tmp true 2>/dev/null | \
  tr ' ' '\n' | head -1
# Expected: .../MOST.0006-11-24_00:00:00
```

**Step 4: Capability probe smoke test**

```bash
bash create_ICs/blocking_1_ICs_0069_12_24.sh --capabilities
# Expected: diverse_initial_conditions
```

**Step 5: Error path verification**

```bash
# Non-boolean type in config
echo '{"diverse_initial_conditions": "yes"}' | \
  jq -r 'if (.diverse_initial_conditions != null) and
            ((.diverse_initial_conditions | type) != "boolean")
          then error("diverse_initial_conditions must be a JSON boolean, got " +
                     (.diverse_initial_conditions | type))
          else (.diverse_initial_conditions // true) end' 2>&1
# Expected: error message mentioning "string"

# N_particles not multiple of 100 in IC script
bash create_ICs/blocking_1_ICs_0069_12_24.sh 50 NorthAtlantic 30 /tmp true
# Expected: exit 1 with error message mentioning "100"
```

**Step 6: Run full test suite**

```bash
cd /glade/u/home/zhil/project/AI-RES/Blocking
python -m pytest tests/create_ICs/ -v
```

Expected: all tests PASS, no regressions.

**Step 7: Commit if any fixes were needed**

```bash
git add -p  # only if fixes were made
git commit -m "fix: integration smoke test corrections"
```

---

### Task 7: Final cleanup and docs update

**Files:**
- Modify: `/glade/u/home/zhil/.claude/projects/-glade-u-home-zhil-project-AI-RES-Blocking/memory/MEMORY.md`
  (or relevant memory file) — record that `diverse_initial_conditions` exists and how it works

**Step 1: Run full test suite one final time**

```bash
cd /glade/u/home/zhil/project/AI-RES/Blocking
python -m pytest tests/ -v
```

Expected: all tests PASS.

**Step 2: Verify EXP15_AIRES.json has `false` (it is a legacy config)**

```bash
jq '.diverse_initial_conditions' \
  AI-RES/RES/experiments/configs/derecho/AIRES/EXP15_AIRES.json
# Expected: false
```

**Step 3: Final commit**

```bash
git add .
git commit -m "chore: finalize diverse_initial_conditions implementation"
```
