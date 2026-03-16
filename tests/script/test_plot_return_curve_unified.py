"""Tests for script/plot_return_curve.py (unified dispatcher)."""

import json
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np


class _FakeNode:
    """Pickle-friendly stand-in for a working-tree node."""

    def __init__(self, name, parent=None, V=None, norm_constant=None):
        self.name = name
        self.parent = parent
        self.V = V
        self.norm_constant = norm_constant
        self.descendants = []


def _make_fake_experiment(tmp_path, scorer_name="HeatwaveMeanScorer", K=7, n_particles=10):
    """Create a minimal experiment directory."""
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


class TestUnifiedCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "exp_path" in result.stdout
        assert "confirm-scorer" in result.stdout

    def test_missing_exp_path_errors(self):
        result = subprocess.run(
            [sys.executable, "script/plot_return_curve.py",
             "--exp_path", "/nonexistent/path",
             "--confirm-scorer", "detected"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestDefaultExpPath:
    def test_resolve_default_exp_path_reads_latest(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import resolve_default_exp_path

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        latest = logs_dir / "latest_exp_path.txt"
        latest.write_text(
            "job_id: 123\n"
            "exp_name: TestExp_0\n"
            "output_path: /some/experiment/path\n"
            "submitted: Mon Jan 01 12:00:00 PM MDT 2026\n"
        )

        result = resolve_default_exp_path(latest)
        assert result == "/some/experiment/path"

    def test_resolve_default_exp_path_missing_file(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import resolve_default_exp_path

        result = resolve_default_exp_path(tmp_path / "nonexistent.txt")
        assert result is None


class TestScorerDispatch:
    def test_dispatches_heatwave(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import detect_and_dispatch

        K = 7
        n_particles = 10
        exp_dir = _make_fake_experiment(
            tmp_path, scorer_name="HeatwaveMeanScorer", K=K, n_particles=n_particles,
        )

        dns_path = tmp_path / "dns_scores.npy"
        np.save(dns_path, np.random.default_rng(42).normal(28.0, 2.0, size=500))

        result = detect_and_dispatch(
            exp_path=exp_dir, K=K, output_dir=tmp_path / "figures",
            confirm_scorer="detected", show_ground_truth=True,
            dns_data_path=str(dns_path), dns_subset_n=100,
            save_fig=True, show_fig=False,
        )

        assert result["scorer_type"] == "heatwave"
        assert "scores_celsius" in result["plot_result"]
        assert Path(result["plot_result"]["output_path"]).exists()

    def test_dispatches_blocking(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

        from script.plot_return_curve import detect_and_dispatch

        K = 7
        n_particles = 10
        exp_dir = _make_fake_experiment(
            tmp_path, scorer_name="GridpointIntensityScorer", K=K, n_particles=n_particles,
        )

        result = detect_and_dispatch(
            exp_path=exp_dir, K=K, output_dir=tmp_path / "figures",
            confirm_scorer="detected", show_ground_truth=False,
            save_fig=True, show_fig=False,
        )

        assert result["scorer_type"] == "blocking"
        assert "scores" in result["plot_result"]
        assert Path(result["plot_result"]["output_path"]).exists()
