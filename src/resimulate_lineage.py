#!/usr/bin/env python
"""
Re-simulate PlaSim for each step in an extreme particle's lineage.

Given an AI-RES experiment where delete_output=partial was used (restart files
kept, .nc output deleted), this script regenerates the output by:
  1. Tracing the lineage of a ranked particle
  2. Copying restart files into a temporary PlaSim run directory
  3. Running PlaSim for dtau days per step
  4. Postprocessing with postprocess_data.py to get zg on pressure levels
  5. Saving output as plasim_out.step_k.particle_m.nc

Usage:
    python resimulate_lineage.py --exp_path /path/to/experiment [--rank 1]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Allow importing siblings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_visualization import _auto_detect_K, _get_ranked_particle, _trace_lineage

# Paths to the postprocessor
POSTPROC_SCRIPT = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "AI-RES", "RES", "postprocessor2.0", "postprocess_data.py",
)
POSTPROC_CONFIG = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "AI-RES", "RES", "postprocessor2.0", "config", "test.yaml",
)


def _find_used_config(exp_path):
    """Find and load the *_used_config.json in the experiment directory."""
    pattern = os.path.join(exp_path, "*_used_config.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No *_used_config.json found in {exp_path}"
        )
    with open(matches[0], "r") as f:
        return json.load(f)


def _setup_run_dir(reference_run_dir, work_dir):
    """Create a temporary PlaSim run directory from the reference template."""
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree(reference_run_dir, work_dir)
    # Set EPSRESTART=0 for deterministic re-run (no perturbation noise)
    namelist = os.path.join(work_dir, "plasim_namelist")
    with open(namelist, "r") as f:
        content = f.read()
    import re
    content = re.sub(r"EPSRESTART\s*=\s*[\d.]+", "EPSRESTART  =     0.0", content)
    with open(namelist, "w") as f:
        f.write(content)
    return work_dir


def _set_run_days(work_dir, n_days):
    """Update N_RUN_DAYS in plasim_namelist."""
    namelist = os.path.join(work_dir, "plasim_namelist")
    with open(namelist, "r") as f:
        content = f.read()
    import re
    content = re.sub(
        r"N_RUN_DAYS\s*=\s*\d+",
        f"N_RUN_DAYS  =     {n_days}",
        content,
    )
    with open(namelist, "w") as f:
        f.write(content)


def resimulate_lineage(exp_path, rank=1, K=None, output_prefix="plasim_out",
                       n_mpi=8):
    """
    Re-run PlaSim for each step in the lineage of a ranked particle.

    Parameters
    ----------
    exp_path : str
        Experiment directory.
    rank : int
        Which particle rank to re-simulate (1 = highest score).
    K : int or None
        Final resampling step. Auto-detected if None.
    output_prefix : str
        Naming prefix for regenerated .nc files.
    n_mpi : int
        Number of MPI ranks for PlaSim (default 8).

    Returns
    -------
    list of str
        Paths to regenerated .nc files.
    """
    # Auto-detect K
    if K is None:
        K = _auto_detect_K(exp_path)
    print(f"Using resampling step K={K}")

    # Get particle and lineage
    particle_idx, score = _get_ranked_particle(exp_path, K, rank)
    print(f"Rank {rank}: particle {particle_idx}, score {score:.6f}")

    lineage = _trace_lineage(exp_path, K, particle_idx)
    print(
        f"Lineage ({len(lineage)} steps): "
        + " -> ".join(f"s{e['step']}p{e['particle']}" for e in lineage)
    )

    # Load experiment config for dtau and reference run dir
    config = _find_used_config(exp_path)
    dtau = int(config["dtau"])
    reference_run_dir = config["PATH_REFERENCE_RUN_DIR"]
    print(f"dtau={dtau} days, reference_run_dir={reference_run_dir}")

    if not os.path.isdir(reference_run_dir):
        raise FileNotFoundError(
            f"Reference run directory not found: {reference_run_dir}"
        )

    # Create temporary run directory
    tmp_base = os.path.join(exp_path, "tmp_resim")
    work_dir = os.path.join(tmp_base, "run")
    _setup_run_dir(reference_run_dir, work_dir)
    _set_run_days(work_dir, dtau)
    print(f"Temporary run directory: {work_dir}")

    generated_files = []

    for entry in lineage:
        k, m = entry["step"], entry["particle"]
        output_dir = os.path.join(exp_path, f"step_{k}", f"particle_{m}", "output")
        output_nc = os.path.join(
            output_dir, f"{output_prefix}.step_{k}.particle_{m}.nc"
        )

        # Skip if output already exists
        if os.path.exists(output_nc):
            print(f"  step_{k}/particle_{m}: output already exists, skipping")
            generated_files.append(output_nc)
            continue

        # Find restart file
        restart_src = os.path.join(
            exp_path, f"step_{k}", f"particle_{m}", "restart", "restart_start"
        )
        if not os.path.exists(restart_src):
            print(f"  step_{k}/particle_{m}: WARNING - restart_start not found, skipping")
            continue

        print(f"  step_{k}/particle_{m}: running PlaSim for {dtau} days...")

        # Copy restart -> plasim_restart
        restart_dst = os.path.join(work_dir, "plasim_restart")
        shutil.copy2(restart_src, restart_dst)

        # Remove stale output
        plasim_output = os.path.join(work_dir, "plasim_output")
        if os.path.exists(plasim_output):
            os.remove(plasim_output)

        # Run PlaSim
        binary = os.path.join(work_dir, "most_plasim_t42_l10_p8.x")
        cmd = ["mpiexec", "-n", str(n_mpi), binary]
        result = subprocess.run(
            cmd, cwd=work_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"    PlaSim failed (rc={result.returncode})")
            print(result.stdout.decode(errors="replace")[-500:])
            continue

        if not os.path.exists(plasim_output):
            print(f"    WARNING: plasim_output not created, skipping postprocessing")
            continue

        # Postprocess: binary -> NetCDF with zg on pressure levels
        os.makedirs(output_dir, exist_ok=True)
        postproc_cmd = [
            sys.executable, POSTPROC_SCRIPT,
            "--config", POSTPROC_CONFIG,
            "--input_files", plasim_output,
            "--output_files", output_nc,
            "--save_dir", output_dir,
        ]
        print(f"    Postprocessing -> {os.path.basename(output_nc)}")
        pp_result = subprocess.run(
            postproc_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=300,
        )
        if pp_result.returncode != 0:
            print(f"    Postprocessing failed (rc={pp_result.returncode})")
            print(pp_result.stdout.decode(errors="replace")[-500:])
            continue

        if os.path.exists(output_nc):
            print(f"    Output saved: {output_nc}")
            generated_files.append(output_nc)
        else:
            # postprocess_data.py may write to a tmp subfolder; search for it
            import glob as _glob
            candidates = _glob.glob(
                os.path.join(output_dir, "**", f"*{output_prefix}*postproc.nc"),
                recursive=True,
            )
            if candidates:
                shutil.move(candidates[0], output_nc)
                print(f"    Output moved to: {output_nc}")
                generated_files.append(output_nc)
            else:
                print(f"    WARNING: postprocessed file not found at {output_nc}")

    # Cleanup temporary run directory
    if os.path.exists(tmp_base):
        shutil.rmtree(tmp_base)
        print(f"Cleaned up {tmp_base}")

    print(f"\nDone. Generated {len(generated_files)} files.")
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Re-simulate PlaSim lineage for a partial-cleanup AI-RES experiment"
    )
    parser.add_argument(
        "--exp_path", required=True,
        help="Path to the AI-RES experiment directory",
    )
    parser.add_argument(
        "--rank", type=int, default=1,
        help="Particle rank to re-simulate (1=highest score, default: 1)",
    )
    parser.add_argument(
        "--K", type=int, default=None,
        help="Final resampling step (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output_prefix", default="plasim_out",
        help="Naming prefix for regenerated .nc files (default: plasim_out)",
    )
    parser.add_argument(
        "--n_mpi", type=int, default=8,
        help="Number of MPI ranks for PlaSim (default: 8)",
    )
    args = parser.parse_args()

    resimulate_lineage(
        exp_path=args.exp_path,
        rank=args.rank,
        K=args.K,
        output_prefix=args.output_prefix,
        n_mpi=args.n_mpi,
    )


if __name__ == "__main__":
    main()
