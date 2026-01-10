#!/usr/bin/env python3
"""
compute_DTFib_N_wrapper.py

Orchestrates multiple sliced executions of compute_DTFib_N.py on the same
historical dataset, then merges results using merge_DTFib.py.

Slices applied to number of rows (rounded up):
  1) 10% -> 100%
  2) 25% -> 100%
  3) 10% -> 95%

Design principles:
- DO NOT modify compute_DTFib.py
- DO NOT modify compute_DTFib_N.py
- DO NOT modify merge_DTFib.py
- All scripts are assumed to live in the invocation directory
- Wrapper absorbs all path / staging complexity
- DTFibRuns directory is ALWAYS preserved (manual cleanup only)
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os
import math
import shutil
import pandas as pd
from typing import List


# ---------------------------------------------------------------------
# Invocation directory (where all .py scripts live)
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENGINE_SCRIPTS = [
    "compute_DTFib.py",
    "compute_DTFib_N.py",
]

MERGE_SCRIPT = os.path.join(BASE_DIR, "merge_DTFib.py")

RUN_ROOT = "DTFibRuns"


# ---------------------------------------------------------------------
# Slice definitions
# ---------------------------------------------------------------------

SLICE_SPECS = [
    ("run_10_100", 0.10, 1.00),
    ("run_25_100", 0.25, 1.00),
    ("run_10_95",  0.10, 0.95),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def slice_dataframe(df: pd.DataFrame, start_frac: float, end_frac: float) -> pd.DataFrame:
    """Slice dataframe by fractional row indices (rounded up)."""
    n = len(df)
    start_idx = int(math.ceil(n * start_frac))
    end_idx = int(math.ceil(n * end_frac))
    end_idx = min(end_idx, n)
    return df.iloc[start_idx:end_idx].copy()


def stage_engine_scripts(run_dir: str) -> None:
    """
    Copy required engine scripts into the run directory so that
    downstream subprocess calls using relative paths succeed.
    """
    for script in ENGINE_SCRIPTS:
        src = os.path.join(BASE_DIR, script)
        dst = os.path.join(run_dir, script)
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)


def run_compute_dtfib_n(
    run_dir: str,
    filename: str,
    tolerance: float
) -> None:
    """Invoke compute_DTFib_N.py inside the run directory."""
    cmd = [
        sys.executable,
        "compute_DTFib_N.py",
        "-f", filename,
        "-t", str(tolerance),
    ]
    subprocess.run(cmd, cwd=run_dir, check=True)


def run_merge(
    run_dirs: List[str],
    symbol: str,
    out_dir: str
) -> None:
    """Invoke merge_DTFib.py from the base directory."""
    cmd = [
        sys.executable,
        MERGE_SCRIPT,
        "-d", ",".join(run_dirs),
        "-s", symbol,
        "-o", out_dir
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run compute_DTFib_N.py across multiple data slices and merge results."
    )

    # ---- Required input ----
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Historical CSV filename in historical_data/ (e.g. Roku.csv)"
    )

    # ---- compute_DTFib_N.py CLI passthrough ----
    parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=0.0075,
        help="Tolerance for Fibonacci matching (default: 0.0075)"
    )

    # ---- merge_DTFib.py CLI ----
    parser.add_argument(
        "-o", "--out",
        default=os.path.join(RUN_ROOT, "merged"),
        help="Output directory for merged results (default: DTFibRuns/merged)"
    )

    args = parser.parse_args(argv)

    # -----------------------------------------------------------------
    # Validate input data
    # -----------------------------------------------------------------

    data_path = os.path.join("historical_data", args.file)
    if not os.path.isfile(data_path):
        print(f"[ERROR] File not found: {data_path}", file=sys.stderr)
        sys.exit(2)

    symbol = os.path.splitext(os.path.basename(args.file))[0]
    df = pd.read_csv(data_path)

    # -----------------------------------------------------------------
    # Prepare run root (ALWAYS preserved)
    # -----------------------------------------------------------------

    os.makedirs(RUN_ROOT, exist_ok=True)

    run_dirs: List[str] = []

    # -----------------------------------------------------------------
    # Execute slice runs
    # -----------------------------------------------------------------

    for run_name, start_frac, end_frac in SLICE_SPECS:
        run_dir = os.path.join(RUN_ROOT, run_name)
        hist_dir = os.path.join(run_dir, "historical_data")

        os.makedirs(hist_dir, exist_ok=True)

        # Stage engine scripts
        stage_engine_scripts(run_dir)

        # Slice data
        df_slice = slice_dataframe(df, start_frac, end_frac)
        df_slice.to_csv(os.path.join(hist_dir, args.file), index=False)

        print(f"[INFO] Running {run_name}: rows={len(df_slice)}")

        run_compute_dtfib_n(
            run_dir=run_dir,
            filename=args.file,
            tolerance=args.tolerance
        )

        run_dirs.append(run_dir)

    # -----------------------------------------------------------------
    # Merge results
    # -----------------------------------------------------------------

    run_merge(
        run_dirs=run_dirs,
        symbol=symbol,
        out_dir=args.out
    )

    print("\n[DONE]")
    print("Merged results written to:")
    print(os.path.join(args.out, "DTFib_results", f"DTFib_{symbol}_N.csv"))


if __name__ == "__main__":
    main()