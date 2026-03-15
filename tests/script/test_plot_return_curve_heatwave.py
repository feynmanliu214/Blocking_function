"""Tests for script/plot_return_curve_heatwave.py."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


class _FakeNode:
    """Pickle-friendly stand-in for a working-tree node."""

    def __init__(self, name, parent=None, V=None, norm_constant=None):
        self.name = name
        self.parent = parent
        self.V = V
        self.norm_constant = norm_constant
        self.descendants = []


def _make_fake_experiment(tmp_path, scorer_name="HeatwaveMeanScorer", K=7, n_particles=10):
    """Create a minimal experiment directory with working_tree.pkl,
    used_config.json, and plasim_scores_step_K.npy."""
    exp_name = "FakeExp_0"
    exp_dir = tmp_path / exp_name
    exp_dir.mkdir()

    config = {
        "scorer": {"name": scorer_name, "params": {}},
        "region": "France",
        "divide_by_parent_weights": True,
    }
    (exp_dir / f"{exp_name}_used_config.json").write_text(json.dumps(config))

    root = _FakeNode("root")
    children = []
    for i in range(n_particles):
        parent_node = _FakeNode(
            f"parent_{K}_{i}", parent=root, V=0.0, norm_constant=1.0,
        )
        child = _FakeNode(f"step_{K}_particle_{i}", parent=parent_node)
        children.append(child)
    root.descendants = children
    root.parent = None

    with (exp_dir / "working_tree.pkl").open("wb") as f:
        pickle.dump(root, f)

    scores = np.linspace(299.0, 307.0, n_particles)
    resampling_dir = exp_dir / "resampling"
    resampling_dir.mkdir()
    np.save(resampling_dir / f"plasim_scores_step_{K}.npy", scores)

    return exp_dir


class TestHeatwaveCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve_heatwave.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "dns-data-path" in result.stdout
        assert "exp_path" in result.stdout

    def test_missing_exp_path_errors(self):
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve_heatwave.py",
             "--exp_path", "/nonexistent/path",
             "--confirm-scorer", "detected"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestScorerEnforcement:
    def test_rejects_blocking_scorer_via_cli(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve_heatwave.py",
             "--exp_path", str(tmp_path),
             "--confirm-scorer", "GridpointIntensityScorer"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_rejects_non_heatwave_detected_scorer(self, tmp_path):
        exp_dir = _make_fake_experiment(tmp_path, scorer_name="GridpointIntensityScorer")
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve_heatwave.py",
             "--exp_path", str(exp_dir),
             "--confirm-scorer", "detected"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "HeatwaveMeanScorer" in result.stderr


class TestHeatwaveControlFlow:
    def test_full_pipeline_with_mock_experiment(self, tmp_path):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve_heatwave import plot_heatwave_return_curve
        from forecast_analysis.scoring.detection import resolve_scorer_profile

        K = 7
        n_particles = 10
        exp_dir = _make_fake_experiment(tmp_path, K=K, n_particles=n_particles)

        dns_path = tmp_path / "dns_scores.npy"
        np.save(dns_path, np.random.default_rng(42).normal(28.0, 2.0, size=500))

        scorer_profile = resolve_scorer_profile("HeatwaveMeanScorer", {})
        output_dir = tmp_path / "figures"

        result = plot_heatwave_return_curve(
            exp_path=exp_dir, output_dir=output_dir, K=K,
            scorer_name="HeatwaveMeanScorer", scorer_params={},
            scorer_profile=scorer_profile, region="France",
            divide_by_parent_weights=True, show_ground_truth=True,
            dns_data_path=dns_path, dns_subset_n=100,
            save_fig=True, show_fig=False,
        )

        assert result["scores_celsius"].max() < 40.0
        assert result["scores_celsius"].min() > 20.0
        assert len(result["rv_exp"]) == n_particles
        assert len(result["rp_exp"]) == n_particles
        assert all(rp > 0 for rp in result["rp_exp"])
        assert Path(result["output_path"]).exists()

    def test_missing_saved_scores_raises(self, tmp_path):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve_heatwave import plot_heatwave_return_curve
        from forecast_analysis.scoring.detection import resolve_scorer_profile

        K = 7
        exp_dir = _make_fake_experiment(tmp_path, K=K, n_particles=5)
        scores_file = exp_dir / "resampling" / f"plasim_scores_step_{K}.npy"
        scores_file.unlink()

        scorer_profile = resolve_scorer_profile("HeatwaveMeanScorer", {})

        with pytest.raises(FileNotFoundError, match="plasim_scores"):
            plot_heatwave_return_curve(
                exp_path=exp_dir, output_dir=tmp_path / "figures", K=K,
                scorer_name="HeatwaveMeanScorer", scorer_params={},
                scorer_profile=scorer_profile, region="France",
                divide_by_parent_weights=True, save_fig=False, show_fig=False,
            )
