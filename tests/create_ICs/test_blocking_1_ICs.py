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
