"""Main experimental runs on the 1000-protein dataset.

Usage: run_experiments.py <config> [--workers K]

Configs (resolution 0.5 A, probe 1.4 A, radii of sadicpipe/vdw.py):
  morph_transph_aligned    MorphPC + TranSph,  OBB-aligned   (SADIC; run with 1 worker for timings)
  infl_reconsph_aligned    InflPC + ReconSph,  OBB-aligned   (baseline; run with 1 worker for timings)
  morph_reconsph_aligned   MorphPC + ReconSph, OBB-aligned
  infl_transph_aligned     InflPC + TranSph,   OBB-aligned
  morph_transph_unaligned  MorphPC + TranSph,  no alignment  (AABB reference)

Proteins are processed in increasing size order (failures surface early).
Resume-safe: already-done proteins are skipped.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from paths import METADATA, NPZ_DIR, RUNS as RUNS_DIR
from sadicpipe.runner import run_batch

BASE = {"resolution": 0.5, "probe_radius": 1.4}
CONFIGS = {
    "morph_transph_aligned": dict(BASE, model="morph", indices="transph", alignment="approximate"),
    "infl_reconsph_aligned": dict(BASE, model="infl", indices="reconsph", alignment="approximate"),
    "morph_reconsph_aligned": dict(BASE, model="morph", indices="reconsph", alignment="approximate"),
    "infl_transph_aligned": dict(BASE, model="infl", indices="transph", alignment="approximate"),
    "morph_transph_unaligned": dict(BASE, model="morph", indices="transph", alignment="none"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    meta = pd.read_csv(METADATA)
    ids = meta[meta.status == "ok"].sort_values("n_atoms").pdb_id.tolist()

    run_dir = os.path.join(RUNS_DIR, args.config)
    run_batch(ids, NPZ_DIR, run_dir, CONFIGS[args.config], workers=args.workers)
    print(f"Run {args.config} complete.")


if __name__ == "__main__":
    main()
