"""Tests for the PFS.py refactoring to use the shared scoring pipeline.

Covers:
- Config without 'scorer' key raises ValueError at module level
- Config with scorer but missing 'variable' key raises KeyError in
  compute_B_int_single_file
- compute_B_int_single_file delegates to the shared pipeline functions
"""

import json
import os
import subprocess
import sys
import textwrap

from unittest import mock

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helper: path to PFS.py
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

_PFS_PY = os.path.join(
    _PROJECT_ROOT,
    "AI-RES", "RES", "forecast_modules", "PFS", "PFS.py",
)


# ===========================================================================
# Test 1: Config without 'scorer' key raises ValueError
# ===========================================================================


class TestScorerRequired:
    """PFS.py must raise ValueError when 'scorer' is missing from config."""

    def test_missing_scorer_raises_valueerror(self, tmp_path):
        """Running PFS.py with a config that has no 'scorer' key must fail
        with a ValueError containing the expected message.
        """
        # Build a minimal config that has all required keys EXCEPT 'scorer'
        config = {
            "path_restart": "/tmp/fake_restart",
            "dir_output": str(tmp_path),
            "basename_output": "test",
            "num_members": 2,
            "lead_time": 0,
            "target_duration": 5,
            "var": "zg",
            "regions": ["NorthAtlantic"],
            "job_manager": "pbs",
            "PATH_SCRATCH": "/tmp",
            "PATH_REFERENCE_RUN_DIR": "/tmp",
            "PATH_POSTPROC_NL": "/tmp/nl",
            "POSTPROC_SCRIPT": "/tmp/postproc.sh",
            "PATH_REGIONS": "/tmp/regions.json",
            "DIR_PFS": "/tmp/pfs",
            # NOTE: no 'scorer' key
        }
        config_path = tmp_path / "test_config.json"
        config_path.write_text(json.dumps(config))

        result = subprocess.run(
            [sys.executable, _PFS_PY, "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, (
            f"Expected non-zero exit code.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "scorer" in result.stderr.lower() or "scorer" in result.stdout.lower(), (
            f"Expected error message about 'scorer'.\nstderr: {result.stderr}\nstdout: {result.stdout}"
        )

    def test_missing_scorer_message_text(self, tmp_path):
        """The ValueError message must contain the exact expected text."""
        config = {
            "path_restart": "/tmp/fake",
            "dir_output": str(tmp_path),
            "basename_output": "test",
            "num_members": 2,
            "lead_time": 0,
            "target_duration": 5,
            "var": "zg",
            "regions": ["NorthAtlantic"],
            "job_manager": "pbs",
            "PATH_SCRATCH": "/tmp",
            "PATH_REFERENCE_RUN_DIR": "/tmp",
            "PATH_POSTPROC_NL": "/tmp/nl",
            "POSTPROC_SCRIPT": "/tmp/postproc.sh",
            "PATH_REGIONS": "/tmp/regions.json",
            "DIR_PFS": "/tmp/pfs",
        }
        config_path = tmp_path / "test_config.json"
        config_path.write_text(json.dumps(config))

        result = subprocess.run(
            [sys.executable, _PFS_PY, "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        combined = result.stdout + result.stderr
        assert "'scorer' is required in experiment config" in combined, (
            f"Expected exact error message.\nOutput: {combined}"
        )


# ===========================================================================
# Test 2: Config with scorer but missing 'variable' raises KeyError
# ===========================================================================


class TestScorerVariableRequired:
    """Variable validation now happens in build_scorer_context (tested in
    test_build_scorer_context.py).  This class is retained for the subprocess
    smoke test: PFS.py must fail when scorer has no 'variable' key.
    """

    def test_missing_variable_in_subprocess(self, tmp_path):
        """Running PFS.py with scorer missing 'variable' must fail with
        a clear error message about 'scorer.variable'.
        """
        regions_json = tmp_path / "regions.json"
        regions_json.write_text(json.dumps({
            "NorthAtlantic": {"lon": [-60.0, 0.0], "lat": [55.0, 75.0]},
        }))
        config = {
            "path_restart": "/tmp/fake",
            "dir_output": str(tmp_path),
            "basename_output": "test",
            "num_members": 2,
            "lead_time": 0,
            "target_duration": 5,
            "var": "zg",
            "regions": ["NorthAtlantic"],
            "job_manager": "pbs",
            "PATH_SCRATCH": "/tmp",
            "PATH_REFERENCE_RUN_DIR": "/tmp",
            "PATH_POSTPROC_NL": "/tmp/nl",
            "POSTPROC_SCRIPT": "/tmp/postproc.sh",
            "PATH_REGIONS": str(regions_json),
            "DIR_PFS": "/tmp/pfs",
            "scorer": {"name": "GridpointPersistenceScorer", "params": {}},
            # NOTE: scorer has no 'variable' key
        }
        config_path = tmp_path / "test_config.json"
        config_path.write_text(json.dumps(config))

        result = subprocess.run(
            [sys.executable, _PFS_PY, "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "scorer.variable" in combined or "variable" in combined.lower(), (
            f"Expected error about missing variable.\nOutput: {combined}"
        )


# ===========================================================================
# Test 3: compute_B_int_single_file delegates to pipeline
# ===========================================================================


class TestComputeBIntDelegatesToPipeline:
    """compute_B_int_single_file should delegate to pipeline functions."""

    def test_calls_pipeline_functions(self, tmp_path):
        """Verify that extract_variable, prepare_daily_field, and
        score_single_member are called when compute_B_int_single_file runs.
        """
        config = {
            "path_restart": "/tmp/fake",
            "dir_output": str(tmp_path),
            "basename_output": "test",
            "num_members": 2,
            "lead_time": 0,
            "target_duration": 5,
            "var": "zg",
            "regions": ["NorthAtlantic"],
            "job_manager": "pbs",
            "PATH_SCRATCH": "/tmp",
            "PATH_REFERENCE_RUN_DIR": "/tmp",
            "PATH_POSTPROC_NL": "/tmp/nl",
            "POSTPROC_SCRIPT": "/tmp/postproc.sh",
            "PATH_REGIONS": "/tmp/regions.json",
            "DIR_PFS": "/tmp/pfs",
            "scorer": {
                "name": "GridpointPersistenceScorer",
                "params": {"min_persistence": 5},
                "variable": "z500",
            },
        }
        config_path = tmp_path / "test_config.json"
        config_path.write_text(json.dumps(config))

        # Create a dummy netCDF file
        nc_path = tmp_path / "dummy.nc"
        ds = xr.Dataset({
            "zg": xr.DataArray(
                np.ones((10, 5, 10)) * 5500,
                dims=["time", "lat", "lon"],
                coords={
                    "time": np.arange(10),
                    "lat": np.linspace(30, 80, 5),
                    "lon": np.linspace(0, 350, 10),
                },
            )
        })
        ds.to_netcdf(str(nc_path))

        scorer_cfg = {
            "name": "GridpointPersistenceScorer",
            "params": {"min_persistence": 5},
            "variable": "z500",
        }

        # Build a real ScorerContext so region_bounds carries ctx objects
        regions_json = tmp_path / "regions.json"
        regions_json.write_text(json.dumps({
            "NorthAtlantic": {"lon": [-60.0, 0.0], "lat": [55.0, 75.0]},
        }))
        from forecast_analysis.scoring.context import build_scorer_context, ScorerContext
        ctx = build_scorer_context(scorer_cfg, "NorthAtlantic", regions_json, onset_time_idx=0)
        # region_bounds is repurposed to carry {region: ScorerContext}
        region_bounds = {"NorthAtlantic": ctx}

        args_tuple = (
            str(nc_path),
            5,
            0,
            "NorthAtlantic",
            region_bounds,       # carries ScorerContext objects now
            xr.DataArray([0]),   # z500_clim dummy
            {1: 100.0},          # threshold_90 dummy
            scorer_cfg,          # still in tuple (unused inside function)
        )

        # Import the module
        import importlib.util
        with mock.patch("sys.argv", ["PFS.py", "--config", str(config_path)]):
            spec = importlib.util.spec_from_file_location("pfs_module2", _PFS_PY)
            pfs_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pfs_module)

        # Mock the pipeline functions
        fake_field = xr.DataArray(
            np.ones((10, 5, 10)) * 5500,
            dims=["time", "lat", "lon"],
            name="z500",
        )
        expected_score = 42.0

        with (
            mock.patch(
                "forecast_analysis.scoring.pipeline.extract_variable",
                return_value=fake_field,
            ) as mock_extract,
            mock.patch(
                "forecast_analysis.scoring.pipeline.prepare_daily_field",
                return_value=fake_field,
            ) as mock_prepare,
            mock.patch(
                "forecast_analysis.scoring.pipeline.score_single_member",
                return_value=expected_score,
            ) as mock_score,
        ):
            result = pfs_module.compute_B_int_single_file(args_tuple)

        assert result == expected_score
        mock_extract.assert_called_once()
        mock_prepare.assert_called_once()
        mock_score.assert_called_once()

        # Verify score_single_member received ScorerContext with right attributes
        call_kwargs = mock_score.call_args.kwargs
        assert isinstance(call_kwargs["ctx"], ScorerContext)
        assert call_kwargs["ctx"].variable == "z500"
