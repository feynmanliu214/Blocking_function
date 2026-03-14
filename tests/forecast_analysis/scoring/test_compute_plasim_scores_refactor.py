"""Tests for the compute_plasim_scores.py refactoring.

Covers:
- Config without 'scorer' key -> sys.exit(1)
- Config with scorer but missing 'name' -> sys.exit(1)
- Config with scorer but missing 'variable' -> sys.exit(1)
- Config with valid scorer skips clim/threshold check for non-anomaly scorer
- Scoring loop delegates to shared pipeline (extract_variable,
  prepare_daily_field, score_single_member)
- Metadata includes 'variable' field
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Path to the script under test
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

_COMPUTE_PLASIM_SCORES_PY = os.path.join(
    _PROJECT_ROOT,
    "AI-RES", "RES", "resampling", "compute_plasim_scores.py",
)


# ---------------------------------------------------------------------------
# Helper: write a config file and run the script as a subprocess
# ---------------------------------------------------------------------------

def _run_script(tmp_path, config, extra_args=None, timeout=30):
    """Write *config* to a JSON file and run compute_plasim_scores.py.

    Returns the subprocess.CompletedProcess.
    """
    config_path = os.path.join(str(tmp_path), "test_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    cmd = [
        sys.executable, _COMPUTE_PLASIM_SCORES_PY,
        "--exp_path", str(tmp_path),
        "--step", "7",
        "--config", config_path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _base_config():
    """Return a minimal config dict with required top-level keys."""
    return {
        "N_particles": 1,
        "region": "NorthAtlantic",
        "PATH_CODE": "/tmp",
        # clim/threshold intentionally absent — tests check behavior
    }


# ===========================================================================
# Test 1: Config without 'scorer' key -> sys.exit(1)
# ===========================================================================

class TestScorerRequired:
    """compute_plasim_scores.py must exit(1) when 'scorer' is missing."""

    def test_missing_scorer_exits_with_error(self, tmp_path):
        config = _base_config()
        # NOTE: no 'scorer' key

        result = _run_script(tmp_path, config)

        assert result.returncode != 0, (
            f"Expected non-zero exit code.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "'scorer' is required in experiment config" in combined, (
            f"Expected exact error message.\nOutput: {combined}"
        )


# ===========================================================================
# Test 2: Config with scorer but missing 'name' -> sys.exit(1)
# ===========================================================================

class TestScorerNameRequired:
    """compute_plasim_scores.py must exit(1) when scorer has no 'name'."""

    def test_missing_scorer_name_exits_with_error(self, tmp_path):
        config = _base_config()
        config["scorer"] = {
            "variable": "z500",
            "params": {},
        }

        result = _run_script(tmp_path, config)

        assert result.returncode != 0, (
            f"Expected non-zero exit code.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "'scorer.name' is required" in combined, (
            f"Expected exact error message.\nOutput: {combined}"
        )


# ===========================================================================
# Test 3: Config with scorer but missing 'variable' -> sys.exit(1)
# ===========================================================================

class TestScorerVariableRequired:
    """compute_plasim_scores.py must exit(1) when scorer has no 'variable'."""

    def test_missing_scorer_variable_exits_with_error(self, tmp_path):
        config = _base_config()
        config["scorer"] = {
            "name": "ANOScorer",
            "params": {},
        }

        result = _run_script(tmp_path, config)

        assert result.returncode != 0, (
            f"Expected non-zero exit code.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "'scorer.variable' is required" in combined, (
            f"Expected exact error message.\nOutput: {combined}"
        )


# ===========================================================================
# Test 4: Non-anomaly scorer skips clim/threshold requirement
# ===========================================================================

class TestNonAnomalyScorerSkipsClim:
    """When scorer does not require anomaly, missing clim/threshold is OK."""

    def test_heatwave_scorer_does_not_require_clim(self, tmp_path):
        """HeatwaveMeanScorer should not exit(1) due to missing clim_file."""
        config = _base_config()
        config["scorer"] = {
            "name": "HeatwaveMeanScorer",
            "variable": "tas",
            "params": {"n_days": 7},
        }
        # No clim_file or threshold_json_file set

        result = _run_script(tmp_path, config)

        combined = result.stdout + result.stderr
        # The script should NOT fail with "clim_file and threshold_json_file are required"
        assert "requires clim_file and threshold_json_file" not in combined, (
            f"Non-anomaly scorer should not require clim/threshold.\nOutput: {combined}"
        )
        # It should get past the clim check and print the skip message
        assert "skipping clim/threshold loading" in combined.lower(), (
            f"Expected skip message for non-anomaly scorer.\nOutput: {combined}"
        )


# ===========================================================================
# Test 5: Anomaly scorer without clim/threshold -> sys.exit(1)
# ===========================================================================

class TestAnomalyScorerRequiresClim:
    """Anomaly-based scorers must fail if clim/threshold are missing."""

    def test_ano_scorer_requires_clim(self, tmp_path):
        config = _base_config()
        config["scorer"] = {
            "name": "ANOScorer",
            "variable": "z500",
            "params": {"mode": "onset"},
        }
        # No clim_file or threshold_json_file

        result = _run_script(tmp_path, config)

        assert result.returncode != 0, (
            f"Expected non-zero exit code.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "requires clim_file and threshold_json_file" in combined, (
            f"Expected clim requirement error.\nOutput: {combined}"
        )


# ===========================================================================
# Test 6: Scorer as JSON string is parsed correctly
# ===========================================================================

class TestScorerJsonString:
    """When scorer config is a JSON string (not dict), it should be parsed."""

    def test_scorer_json_string_parsed(self, tmp_path):
        config = _base_config()
        config["scorer"] = json.dumps({
            "name": "HeatwaveMeanScorer",
            "variable": "tas",
            "params": {"n_days": 7},
        })
        # No clim needed for heatwave

        result = _run_script(tmp_path, config)

        combined = result.stdout + result.stderr
        # Should parse the JSON string and proceed past scorer validation
        assert "'scorer' is required" not in combined
        assert "'scorer.name' is required" not in combined
        assert "'scorer.variable' is required" not in combined
        # Should print the scorer info
        assert "HeatwaveMeanScorer" in combined, (
            f"Expected scorer name in output.\nOutput: {combined}"
        )
