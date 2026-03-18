"""Tests for the PanguPlasimFS.py refactoring to use the shared scoring pipeline.

Covers:
- Config without 'scorer' key raises ValueError at module level
- Config with scorer but missing 'variable' key raises KeyError in
  compute_A_ensemble
- compute_A_ensemble delegates to the shared pipeline functions
  (extract_variable, prepare_daily_field, score_single_member)
"""

import json
import os
import sys
from unittest import mock

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helper: path to PanguPlasimFS.py
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

_PANGUPLASIMFS_PY = os.path.join(
    _PROJECT_ROOT,
    "AI-RES", "RES", "forecast_modules", "PanguPlasim", "PanguPlasimFS.py",
)


# ---------------------------------------------------------------------------
# Helpers: build args tuple for compute_A_ensemble
# ---------------------------------------------------------------------------

def _make_ensemble_dataset(var_name="zg", n_time=40, n_lat=5, n_lon=10,
                           n_ensemble=2):
    """Build a minimal xr.Dataset resembling PanguPlasim emulator output.

    Returns a dataset with an ``ensemble_idx`` dimension when *n_ensemble* > 1.
    """
    rng = np.random.RandomState(42)
    times = xr.cftime_range("0069-11-24", periods=n_time, freq="6h",
                            calendar="proleptic_gregorian")
    lat = np.linspace(0, 80, n_lat)
    lon = np.linspace(0, 350, n_lon)

    if n_ensemble > 1:
        data = rng.randn(n_time, n_ensemble, n_lat, n_lon) * 50 + 5500
        da = xr.DataArray(
            data,
            dims=["time", "ensemble_idx", "lat", "lon"],
            coords={
                "time": times,
                "ensemble_idx": np.arange(n_ensemble),
                "lat": lat,
                "lon": lon,
            },
            name=var_name,
        )
    else:
        data = rng.randn(n_time, n_lat, n_lon) * 50 + 5500
        da = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lat, "lon": lon},
            name=var_name,
        )

    return xr.Dataset({var_name: da})


def _make_regions_json(tmp_path):
    """Create a regions.json file and return its path."""
    regions = {
        "NorthAtlantic": {
            "lon": [-60, 0],
            "lat": [55, 75],
        },
        "France": {
            "lon": [0.0, 2.8125, 5.625],
            "lat": [48.8, 46.0, 43.2],
        },
    }
    path = os.path.join(str(tmp_path), "regions.json")
    with open(path, "w") as f:
        json.dump(regions, f)
    return path


def _build_args_tuple(ds_list, tmp_path, scorer_config,
                      particle_idxs_list=None,
                      ensemble_start=0, ensemble_end=2,
                      lead_time=0, target_duration=5,
                      regions=None):
    """Build the args tuple expected by compute_A_ensemble."""
    save_dir = str(tmp_path)
    save_basenames = [os.path.join(save_dir, "test_p0")]
    if particle_idxs_list is None:
        particle_idxs_list = list(range(len(ds_list)))
    if regions is None:
        regions = ["NorthAtlantic"]
    regions_json = _make_regions_json(tmp_path)

    return (
        ds_list,
        particle_idxs_list,
        ensemble_start,
        ensemble_end,
        save_basenames,
        target_duration,
        lead_time,
        "zg",                       # var (legacy config field)
        regions,
        regions_json,               # PATH_REGIONS
        "/tmp/fake_clim.nc",        # clim_file
        "/tmp/fake_thresh.json",    # threshold_json_file
        scorer_config,
    )


def _import_compute_A_ensemble():
    """Import compute_A_ensemble from PanguPlasimFS.py without executing __main__.

    PanguPlasimFS.py has heavy module-level imports (torch, etc.) guarded
    inside ``if __name__ == '__main__'``.  The function ``compute_A_ensemble``
    is defined at module scope, so we can load just that function by importing
    the module as a spec.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "panguplasim_module", _PANGUPLASIMFS_PY,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)

    # Prevent __main__ execution — the file guards it already, but to be safe
    # we set __name__ to something other than '__main__'.
    mod.__name__ = "panguplasim_module"

    # The module imports torch at the top level, which should be available.
    spec.loader.exec_module(mod)
    return mod.compute_A_ensemble


# ===========================================================================
# Test 1: Config without 'scorer' key raises ValueError
# ===========================================================================


class TestScorerRequired:
    """PanguPlasimFS.py must raise ValueError when 'scorer' is missing."""

    def test_missing_scorer_raises_valueerror(self, tmp_path):
        """Running the __main__ block with a config that has no 'scorer' key
        must fail with a ValueError containing the expected message.

        We run PanguPlasimFS.py as a subprocess to test the __main__ guard.
        The script will fail before reaching GPU/torch setup because the
        ValueError is raised first.
        """
        import subprocess

        config = {
            "input_files": "/tmp/fake.nc",
            "dirs_output": str(tmp_path),
            "basename_output": "test",
            "num_members": 2,
            "lead_time": 0,
            "target_duration": 5,
            "init_datetimes": ["0069-11-24 00:00:00"],
            "var": "zg",
            "regions": ["NorthAtlantic"],
            "job_manager": "pbs",
            "PATH_SCRATCH": "/tmp",
            "PATH_REFERENCE_RUN_DIR": "/tmp",
            "PATH_REGIONS": "/tmp/regions.json",
            "DIR_PANGUPLASIMFS": "/tmp",
            "PATH_YAML_CONFIG": "/tmp/fake.yaml",
            "run_num": "0001",
            "PanguPlasim_perturb": 0.0,
            "clim_file": "/tmp/clim.nc",
            "threshold_json_file": "/tmp/thresh.json",
            # NOTE: no 'scorer' key
        }
        config_path = os.path.join(str(tmp_path), "test_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = subprocess.run(
            [sys.executable, _PANGUPLASIMFS_PY,
             "--config", config_path,
             "--panguplasim_model_dir", "/tmp"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, (
            f"Expected non-zero exit code.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "'scorer' is required in experiment config" in combined, (
            f"Expected exact error message.\nOutput: {combined}"
        )


# ===========================================================================
# Test 2: Config with scorer but missing 'variable' raises KeyError
# ===========================================================================


class TestScorerVariableRequired:
    """compute_A_ensemble must raise KeyError when scorer_config has no
    'variable' key.
    """

    def test_missing_variable_raises(self, tmp_path):
        """scorer_config without 'variable' must produce a ValueError."""
        compute_A_ensemble = _import_compute_A_ensemble()

        ds = _make_ensemble_dataset(n_ensemble=1)
        scorer_config_no_var = {"name": "GridpointPersistenceScorer", "params": {}}

        args = _build_args_tuple(
            ds_list=[ds],
            tmp_path=tmp_path,
            scorer_config=scorer_config_no_var,
        )

        with pytest.raises((KeyError, ValueError)):
            compute_A_ensemble(args)


# ===========================================================================
# Test 3: compute_A_ensemble delegates to pipeline functions
# ===========================================================================


class TestComputeAEnsembleDelegatesToPipeline:
    """compute_A_ensemble should delegate to shared pipeline functions."""

    def test_calls_pipeline_functions(self, tmp_path):
        """Verify that extract_variable, prepare_daily_field, and
        score_single_member are called when compute_A_ensemble runs.
        """
        compute_A_ensemble = _import_compute_A_ensemble()

        ds = _make_ensemble_dataset(var_name="zg", n_ensemble=2)

        scorer_cfg = {
            "name": "GridpointPersistenceScorer",
            "params": {"min_persistence": 5},
            "variable": "z500",
        }

        args = _build_args_tuple(
            ds_list=[ds],
            tmp_path=tmp_path,
            scorer_config=scorer_cfg,
        )

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
            mock.patch(
                "forecast_analysis.data_loading.load_climatology_and_thresholds",
                return_value=(xr.DataArray([0]), {1: 100.0}),
            ),
        ):
            compute_A_ensemble(args)

        # With 2 ensemble members, extract_variable and prepare_daily_field
        # should each be called once per member (2 times).
        assert mock_extract.call_count == 2
        assert mock_prepare.call_count == 2

        # score_single_member is called per member per region:
        # 2 members * 1 region = 2 calls
        assert mock_score.call_count == 2

        # Verify score_single_member received a ScorerContext with right attributes
        from forecast_analysis.scoring.context import ScorerContext
        call_kwargs = mock_score.call_args.kwargs
        ctx = call_kwargs["ctx"]
        assert isinstance(ctx, ScorerContext)
        assert ctx.variable == "z500"

        # Verify the output file was saved
        expected_file = os.path.join(
            str(tmp_path), "test_p0_0000-0002_A_NorthAtlantic.npy"
        )
        assert os.path.exists(expected_file), (
            f"Expected output file not found: {expected_file}"
        )
        A = np.load(expected_file)
        assert len(A) == 2  # 2 ensemble members
        np.testing.assert_allclose(A, expected_score)

    def test_no_ensemble_dim(self, tmp_path):
        """When dataset has no ensemble_idx dim, pipeline is called once."""
        compute_A_ensemble = _import_compute_A_ensemble()

        ds = _make_ensemble_dataset(var_name="zg", n_ensemble=1)

        scorer_cfg = {
            "name": "GridpointPersistenceScorer",
            "params": {"min_persistence": 5},
            "variable": "z500",
        }

        args = _build_args_tuple(
            ds_list=[ds],
            tmp_path=tmp_path,
            scorer_config=scorer_cfg,
        )

        fake_field = xr.DataArray(
            np.ones((10, 5, 10)) * 5500,
            dims=["time", "lat", "lon"],
            name="z500",
        )

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
                return_value=10.0,
            ) as mock_score,
            mock.patch(
                "forecast_analysis.data_loading.load_climatology_and_thresholds",
                return_value=(xr.DataArray([0]), {1: 100.0}),
            ),
        ):
            compute_A_ensemble(args)

        # No ensemble_idx dim -> 1 call per function
        assert mock_extract.call_count == 1
        assert mock_prepare.call_count == 1
        assert mock_score.call_count == 1

    def test_skips_clim_loading_for_non_anomaly_scorer(self, tmp_path):
        """When scorer does not require anomalies, clim/thresholds are not loaded."""
        compute_A_ensemble = _import_compute_A_ensemble()

        ds = _make_ensemble_dataset(var_name="tas", n_ensemble=1)

        scorer_cfg = {
            "name": "HeatwaveMeanScorer",
            "params": {"n_days": 7},
            "variable": "tas",
        }

        # HeatwaveMeanScorer only allows "France" or "Chicago" regions
        args = _build_args_tuple(
            ds_list=[ds],
            tmp_path=tmp_path,
            scorer_config=scorer_cfg,
            regions=["France"],
        )

        fake_field = xr.DataArray(
            np.ones((10, 5, 10)) * 290.0,
            dims=["time", "lat", "lon"],
            name="tas",
        )

        with (
            mock.patch(
                "forecast_analysis.data_loading.load_climatology_and_thresholds",
            ) as mock_load_clim,
            mock.patch(
                "forecast_analysis.scoring.pipeline.extract_variable",
                return_value=fake_field,
            ),
            mock.patch(
                "forecast_analysis.scoring.pipeline.prepare_daily_field",
                return_value=fake_field,
            ),
            mock.patch(
                "forecast_analysis.scoring.pipeline.score_single_member",
                return_value=290.0,
            ) as mock_score,
        ):
            compute_A_ensemble(args)

        # load_climatology_and_thresholds should NOT be called (non-anomaly scorer)
        mock_load_clim.assert_not_called()

        # score_single_member should receive z500_clim=None, threshold_90=None
        call_kwargs = mock_score.call_args.kwargs
        assert call_kwargs["z500_clim"] is None
        assert call_kwargs["threshold_90"] is None
