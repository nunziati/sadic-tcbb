"""Sensitivity analysis on the grid resolution (paper, Section "Sensitivity to Resolution").

Resolutions {0.3, 0.5, 0.8, 1.0, 1.5} A with the default probe 1.4 A, MorphPC + TranSph,
OBB-aligned. Other probe radii can be added with --probes (e.g. --probes 1.4 1.52 1.58).

Subset: 5 proteins per atom-count bin (30 bins) = 150 proteins, listed in
data/ablation_subset.csv (drawn deterministically if the file is missing).
Outputs: runs/ablation/res{r}_probe{p}/ in the standard run format.
"""

import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from paths import ABLATION_SUBSET as SUBSET_FILE, METADATA, NPZ_DIR, RUNS, SAMPLE
from sadicpipe.runner import run_batch

RUNS_DIR = os.path.join(RUNS, "ablation")
RESOLUTIONS = [0.3, 0.5, 0.8, 1.0, 1.5]
PER_BIN = 5
SEED = 42


def build_subset():
    if os.path.exists(SUBSET_FILE):
        return pd.read_csv(SUBSET_FILE).pdb_id.tolist()
    rng = random.Random(SEED)
    sample = pd.read_csv(SAMPLE)
    meta = pd.read_csv(METADATA)
    ok = set(meta[meta.status == "ok"].pdb_id)
    chosen = []
    for b, group in sample[sample.pdb_id.isin(ok)].groupby("bin"):
        ids = sorted(group.pdb_id.tolist())
        rng.shuffle(ids)
        chosen.extend(ids[:PER_BIN])
    pd.DataFrame({"pdb_id": sorted(chosen)}).to_csv(SUBSET_FILE, index=False)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--probes", type=float, nargs="+", default=[1.4])
    args = parser.parse_args()

    subset = build_subset()
    meta = pd.read_csv(METADATA).set_index("pdb_id")
    subset = sorted(subset, key=lambda p: meta.loc[p, "n_atoms"])
    print(f"Ablation subset: {len(subset)} proteins")

    for res in RESOLUTIONS:
        for probe in args.probes:
            config = dict(model="morph", indices="transph", alignment="approximate",
                          resolution=res, probe_radius=probe)
            run_dir = os.path.join(RUNS_DIR, f"res{res}_probe{probe}")
            run_batch(subset, NPZ_DIR, run_dir, config, workers=args.workers)


if __name__ == "__main__":
    main()
