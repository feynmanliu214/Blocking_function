import subprocess
import pathlib
import pytest

# Anchor the script path relative to this test file so tests can be run
# from any working directory.
_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
SCRIPT = str(_REPO_ROOT / "AI-RES/RES/create_ICs/blocking_1_ICs_0069_12_24.sh")

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
    assert tokens == ["diverse_initial_conditions"]

def test_capabilities_probe_no_extra_stdout():
    """--capabilities must produce no output beyond capability tokens."""
    result = run_script("--capabilities")
    assert result.returncode == 0
    for line in result.stdout.strip().splitlines():
        assert line.strip() != "", "unexpected blank line in capabilities output"
    # stderr must be empty in capability mode
    assert result.stderr.strip() == ""


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

def test_diverse_true_400_walkers_correct_count():
    """diverse=true, N=400 must return exactly 400 paths."""
    result = run_script("400", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 400

def test_diverse_true_400_walkers_block_assignment():
    """walkers 0-3 -> year 0006, walkers 4-7 -> year 0007, ..., walkers 396-399 -> year 0105."""
    result = run_script("400", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    # EVENT_DATE=0069-12-24, length_simu=30 -> init date is YYYY-11-24
    assert "0006-11-24" in paths[0]
    assert "0006-11-24" in paths[3]   # last in first block of 4
    assert "0007-11-24" in paths[4]   # first in second block
    assert "0105-11-24" in paths[399]  # last year, last walker

def test_diverse_true_100_walkers_one_per_year():
    """N=100 -> 1 walker per year, 100 distinct paths."""
    result = run_script("100", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 100
    assert len(set(paths)) == 100  # all distinct

def test_diverse_true_200_walkers_two_per_year():
    """N=200 -> 2 walkers per year, block assignment."""
    result = run_script("200", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 200
    assert "0006-11-24" in paths[0]
    assert "0006-11-24" in paths[1]
    assert "0007-11-24" in paths[2]

def test_diverse_mode_only_stdout_is_paths():
    """No diagnostic output on stdout in diverse mode; stderr is separate."""
    result = run_script("100", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    for token in result.stdout.strip().split():
        assert token.startswith("/"), f"Non-path token on stdout: {token!r}"

def test_diverse_mode_does_not_require_single_year_file():
    """diverse=true must succeed without checking the single-year restart file.
    Guards against the bug where the standard-mode existence check (for MOST.0069-11-24)
    runs unconditionally and may fail in diverse mode."""
    result = run_script("400", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, \
        f"diverse mode failed unexpectedly:\n{result.stderr}"

def test_list_length_assertion_300_walkers():
    """Script must return exactly N_particles paths (300 in this case)."""
    result = run_script("300", "NorthAtlantic", "30", "/tmp", "true")
    assert result.returncode == 0, result.stderr
    paths = result.stdout.strip().split()
    assert len(paths) == 300
