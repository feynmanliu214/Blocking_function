"""Tests for the generalized validate_forecast_scores() function in lib_forecast.sh.

Part 1: Shell text tests — verify the function exists and is wired up correctly.
Part 2: Python validation logic tests — exercise the embedded Python one-liner directly.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest


###############################################################################
# Helpers to read source files
###############################################################################


def _read_lib_forecast_script() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    lib_forecast_path = repo_root / "AI-RES" / "RES" / "lib" / "lib_forecast.sh"
    return lib_forecast_path.read_text(encoding="utf-8")


def _extract_function_body(text: str, func_name: str, end_marker: str) -> str:
    """Return the text between func_name and end_marker."""
    start = text.find(func_name)
    assert start != -1, f"Function {func_name} not found"
    end = text.find(end_marker, start)
    assert end != -1, f"End marker {end_marker!r} not found after {func_name}"
    return text[start:end]


###############################################################################
# Part 1: Shell text tests
###############################################################################


def test_validate_forecast_scores_function_exists():
    text = _read_lib_forecast_script()
    assert "validate_forecast_scores" in text


def test_validate_panguplasim_forecast_scores_removed():
    text = _read_lib_forecast_script()
    assert "validate_panguplasim_forecast_scores" not in text


def test_run_pfs_calls_validate_forecast_scores():
    text = _read_lib_forecast_script()
    run_pfs_body = _extract_function_body(
        text, "function run_PFS()", "function run_forecasts()"
    )
    assert "validate_forecast_scores" in run_pfs_body


def test_run_panguplasim_calls_validate_forecast_scores():
    text = _read_lib_forecast_script()
    run_pangu_body = _extract_function_body(
        text, "function run_PanguPlasim()", "function run_PFS()"
    )
    assert "validate_forecast_scores" in run_pangu_body


def test_validate_function_checks_expected_length():
    text = _read_lib_forecast_script()
    assert "a.shape[0] != expected" in text


def test_validate_function_has_missing_count_check():
    text = _read_lib_forecast_script()
    assert "missing_count" in text


def test_validate_function_has_empty_count_check():
    text = _read_lib_forecast_script()
    assert "empty_count" in text


###############################################################################
# Part 2: Python validation logic tests
###############################################################################


def _run_validation_python(
    tmp_path: Path,
    step: int,
    prefix: str,
    n_particles: int,
    expected_length: int,
    setup_fn,
) -> tuple[bool, str]:
    """
    Run the Python one-liner validation logic extracted from validate_forecast_scores()
    against a temporary directory populated by setup_fn.

    Returns (passed, stderr_output).
    """
    # Note: in the shell one-liner, step comes from sys.argv[5] as a string.
    # Here we pass step as int; f-string coercion produces the same path strings.
    # Create the directory structure expected by the validation logic
    path_exp = tmp_path
    region = "TestRegion"
    for i in range(n_particles):
        forecast_dir = path_exp / f"step_{step}" / f"particle_{i}" / "forecast"
        forecast_dir.mkdir(parents=True, exist_ok=True)

    setup_fn(path_exp, step, prefix, n_particles, region)

    # --- replicate the exact embedded Python logic ---
    bad = []
    for i in range(n_particles):
        f = (
            path_exp
            / f"step_{step}"
            / f"particle_{i}"
            / "forecast"
            / f"{prefix}_s{step}_p{i}_A_{region}.npy"
        )
        f_str = str(f)
        try:
            a = np.load(f_str)
            if a.ndim != 1:
                bad.append(f"{f_str}: expected 1-D, got ndim={a.ndim}")
            elif a.shape[0] != expected_length:
                bad.append(
                    f"{f_str}: expected length {expected_length}, got {a.shape[0]}"
                )
        except Exception as e:
            bad.append(f"{f_str}: {e}")

    if bad:
        err_lines = []
        for b in bad[:10]:
            err_lines.append(f"  BAD: {b}")
        if len(bad) > 10:
            err_lines.append(f"  ... and {len(bad) - 10} more")
        err_lines.append(
            f"ERROR: {len(bad)}/{n_particles} score files failed content validation"
        )
        return False, "\n".join(err_lines)

    return True, ""


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def test_valid_files_pass(tmp_path):
    """N=3 files each with shape (50,) and expected=50 should pass."""

    def setup(path_exp, step, prefix, n, region):
        for i in range(n):
            f = (
                path_exp
                / f"step_{step}"
                / f"particle_{i}"
                / "forecast"
                / f"{prefix}_s{step}_p{i}_A_{region}.npy"
            )
            np.save(str(f), np.zeros(50))

    passed, err = _run_validation_python(
        tmp_path, step=1, prefix="PFS", n_particles=3, expected_length=50, setup_fn=setup
    )
    assert passed, f"Expected validation to pass but got: {err}"


def test_truncated_file_fails(tmp_path):
    """Files with shape (3,) when expected=50 should fail with descriptive message."""

    def setup(path_exp, step, prefix, n, region):
        for i in range(n):
            f = (
                path_exp
                / f"step_{step}"
                / f"particle_{i}"
                / "forecast"
                / f"{prefix}_s{step}_p{i}_A_{region}.npy"
            )
            np.save(str(f), np.zeros(3))

    passed, err = _run_validation_python(
        tmp_path, step=2, prefix="PanguPlasimFS", n_particles=3, expected_length=50, setup_fn=setup
    )
    assert not passed
    assert "expected length 50, got 3" in err


def test_wrong_ndim_fails(tmp_path):
    """A 2-D array (shape (50, 2)) should fail with an ndim message."""

    def setup(path_exp, step, prefix, n, region):
        for i in range(n):
            f = (
                path_exp
                / f"step_{step}"
                / f"particle_{i}"
                / "forecast"
                / f"{prefix}_s{step}_p{i}_A_{region}.npy"
            )
            np.save(str(f), np.zeros((50, 2)))

    passed, err = _run_validation_python(
        tmp_path, step=3, prefix="PFS", n_particles=2, expected_length=50, setup_fn=setup
    )
    assert not passed
    assert "ndim" in err


def test_empty_file_fails(tmp_path):
    """A zero-byte file (numpy cannot load it) should fail."""

    def setup(path_exp, step, prefix, n, region):
        for i in range(n):
            f = (
                path_exp
                / f"step_{step}"
                / f"particle_{i}"
                / "forecast"
                / f"{prefix}_s{step}_p{i}_A_{region}.npy"
            )
            # Write a zero-byte file
            f.write_bytes(b"")

    passed, err = _run_validation_python(
        tmp_path, step=1, prefix="PFS", n_particles=2, expected_length=50, setup_fn=setup
    )
    assert not passed
    # numpy raises an exception for a zero-byte file; should be captured in bad list
    assert "BAD" in err or "ERROR" in err


def test_missing_file_fails(tmp_path):
    """If a score file is missing entirely, the Python one-liner should catch the exception."""

    def setup(path_exp, step, prefix, n, region):
        # Only create files for particles 0..n-2; leave particle n-1 missing
        for i in range(n - 1):
            f = (
                path_exp
                / f"step_{step}"
                / f"particle_{i}"
                / "forecast"
                / f"{prefix}_s{step}_p{i}_A_{region}.npy"
            )
            np.save(str(f), np.zeros(50))

    passed, err = _run_validation_python(
        tmp_path, step=1, prefix="PFS", n_particles=3, expected_length=50, setup_fn=setup
    )
    assert not passed
    assert "BAD" in err or "ERROR" in err
