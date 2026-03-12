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
    assert tokens == ["diverse_initial_conditions"]

def test_capabilities_probe_no_extra_stdout():
    """--capabilities must produce no output beyond capability tokens."""
    result = run_script("--capabilities")
    assert result.returncode == 0
    for line in result.stdout.strip().splitlines():
        assert line.strip() != "", "unexpected blank line in capabilities output"
    # stderr must be empty in capability mode
    assert result.stderr.strip() == ""
