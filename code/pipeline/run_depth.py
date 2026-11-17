"""DEPTH over the normalised assembly PDBs.

DEPTH 2.0 (Tan et al.) is third-party software and is not distributed with this code:
place the executable and its water_models/ directory in external/depth/ or point
SADIC_DEPTH_DIR to them. It is invoked as
    DEPTH -i {id}.pdb -o {out} -w water_models/spc216.pdb -survive 5 -n 200
The water box is passed explicitly: if the compiled-in default path does not exist,
DEPTH does not terminate.

Parallel across proteins (each DEPTH process uses ~2 threads). Resume-safe.
Outputs: runs/depth/{id}/output-atom.depth (+ residue files) and a manifest CSV.
"""

import argparse
import concurrent.futures as cf
import csv
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from paths import DEPTH_DIR, METADATA, PDB_DIR, RUNS

OUT_DIR = os.path.join(RUNS, "depth")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")
DEPTH_BIN = os.path.join(DEPTH_DIR, "DEPTH")
WATER = os.path.join(DEPTH_DIR, "water_models", "spc216.pdb")

TIMEOUT = 3600  # seconds per protein


def run_one(pdb_id):
    out_dir = os.path.join(OUT_DIR, pdb_id)
    atom_file = os.path.join(out_dir, "output-atom.depth")
    if os.path.exists(atom_file) and os.path.getsize(atom_file) > 0:
        return pdb_id, "cached", 0.0
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        DEPTH_BIN,
        "-i", os.path.join(PDB_DIR, f"{pdb_id}.pdb"),
        "-o", os.path.join(out_dir, "output"),
        "-w", WATER,
        "-survive", "5",
        "-n", "200",
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=out_dir, timeout=TIMEOUT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return pdb_id, "timeout", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return pdb_id, f"exit_{proc.returncode}", elapsed
    if not (os.path.exists(atom_file) and os.path.getsize(atom_file) > 0):
        return pdb_id, "no_output", elapsed
    return pdb_id, "ok", elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if not os.access(DEPTH_BIN, os.X_OK):
        raise SystemExit(f"DEPTH executable not found at {DEPTH_BIN} (set SADIC_DEPTH_DIR)")
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = pd.read_csv(METADATA)
    ids = meta[meta.status == "ok"].sort_values("n_atoms").pdb_id.tolist()

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(run_one, ids)):
            rows.append(res)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(ids)}", flush=True)

    with open(MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "status", "seconds"])
        writer.writerows(rows)

    statuses = pd.DataFrame(rows, columns=["pdb_id", "status", "seconds"])
    print(statuses.status.value_counts().to_string())
    bad = statuses[~statuses.status.isin(["ok", "cached"])]
    if len(bad):
        print("Failures:", bad.pdb_id.tolist())


if __name__ == "__main__":
    main()
