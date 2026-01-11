#!/usr/bin/env python3
"""
compute_DTFib_N_wrapper.py

Wrapper to run compute_DTFib_N.py on multiple temporal slices
and merge results via merge_DTFib.py.

Slicing strategy (conditional by record count):

Always:
  Run 1: 10% -> 100%

Run 2: 20% -> 100%
  - Only if total_rows <= 8000
  - Skipped if 2500 <= total_rows <= 3000

If total_rows >= 2500 (~10y):
  Run 3: 30% -> 100%

If total_rows > 5000 (~20y):
  Run 4: 40% -> 100%

If total_rows > 8000 (~32y):
  Run 5: 50% -> 100%

Design principles:
- DO NOT modify compute_DTFib.py
- DO NOT modify compute_DTFib_N.py
- DO NOT modify merge_DTFib.py
- Wrapper absorbs all path / staging complexity
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
import shutil


# ---------------------------------------------------------------------
# Invocation directory (where all engine scripts live)
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ENGINE_SCRIPTS = [
    "compute_DTFib.py",
    "compute_DTFib_N.py",
]

MERGE_SCRIPT = BASE_DIR / "merge_DTFib.py"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_row_count(csv_path: Path) -> int:
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # exclude header


def slice_csv(src: Path, dst: Path, start_frac: float) -> None:
    with open(src, "r", encoding="utf-8") as fin:
        rows = list(csv.reader(fin))

    header = rows[0]
    data = rows[1:]

    total = len(data)
    start_idx = math.ceil(total * start_frac)
    sliced = data[start_idx:]

    with open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(sliced)


def stage_engine_scripts(run_dir: Path) -> None:
    """Copy engine scripts into run directory (once)."""
    for script in ENGINE_SCRIPTS:
        src = BASE_DIR / script
        dst = run_dir / script
        if not dst.exists():
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="CSV filename (symbol.csv)")
    parser.add_argument("-o", "--out", default="DTFibRuns", help="Output root directory")

    # Passthrough args (DO NOT change defaults)
    parser.add_argument("-t", type=float)
    parser.add_argument("-ma", type=float)
    parser.add_argument("-v", action="store_true")

    args, unknown = parser.parse_known_args()

    data_path = Path("historical_data") / args.file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    symbol = Path(args.file).stem
    total_rows = read_row_count(data_path)

    print(f"[INFO] {symbol}: total rows = {total_rows}")

    # -----------------------------------------------------------------
    # Build slice plan (UNCHANGED logic)
    # -----------------------------------------------------------------

    slice_plan = [("run_10_100", 0.10)]

    if total_rows <= 8000 and not (2500 <= total_rows <= 3000):
        slice_plan.append(("run_20_100", 0.20))

    if total_rows >= 2500:
        slice_plan.append(("run_30_100", 0.30))

    if total_rows > 5000:
        slice_plan.append(("run_40_100", 0.40))

    if total_rows > 8000:
        slice_plan.append(("run_50_100", 0.50))

    # -----------------------------------------------------------------
    # Execute runs
    # -----------------------------------------------------------------

    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True)

    run_dirs = []

    for run_name, frac in slice_plan:
        run_dir = out_root / run_name / symbol
        hist_dir = run_dir / "historical_data"

        hist_dir.mkdir(parents=True, exist_ok=True)
        stage_engine_scripts(run_dir)

        sliced_csv = hist_dir / f"{symbol}.csv"
        slice_csv(data_path, sliced_csv, frac)

        print(f"[INFO] {run_name}: using last {int((1 - frac) * 100)}% of data")

        cmd = [
            sys.executable,
            "compute_DTFib_N.py",
            "-f", f"{symbol}.csv",
        ]

        if args.t is not None:
            cmd += ["-t", str(args.t)]
        if args.ma is not None:
            cmd += ["-ma", str(args.ma)]
        if args.v:
            cmd.append("-v")

        cmd += unknown

        subprocess.check_call(cmd, cwd=run_dir)
        run_dirs.append(run_dir)

    # -----------------------------------------------------------------
    # Merge results
    # -----------------------------------------------------------------

    merge_cmd = [
        sys.executable,
        str(MERGE_SCRIPT),
        "-s", symbol,
        "-o", args.out,
    ]

    subprocess.check_call(merge_cmd)

    print("\n[DONE]")
    print("Merged results written to:")
    print(os.path.join(args.out, "DTFib_results", f"DTFib_{symbol}_N.csv"))


if __name__ == "__main__":
    main()
