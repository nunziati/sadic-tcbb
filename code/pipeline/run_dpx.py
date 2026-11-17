"""DPX depth index (Pintar et al. 2003) over the normalised assembly PDBs.

DPX is the depth metric computed by PSAIA: for every non-hydrogen atom,
dpx = Euclidean distance to the closest solvent-accessible atom (SASA > 0);
dpx = 0 for solvent-accessible atoms themselves. SASA is computed with FreeSASA
(Shrake-Rupley default parameters, probe 1.4 A), matching Pintar's definition.

PSAIA itself is no longer obtainable in a runnable form, so DPX is computed here from
its original definition.

Outputs: runs/dpx/{id}.csv with chain,resseq,icode,atomname,sasa,dpx.
"""

import argparse
import concurrent.futures as cf
import csv
import os

import sys

import freesasa
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from paths import METADATA, PDB_DIR, RUNS

OUT_DIR = os.path.join(RUNS, "dpx")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")

freesasa.setVerbosity(freesasa.silent)


def run_one(pdb_id):
    out_path = os.path.join(OUT_DIR, f"{pdb_id}.csv")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return pdb_id, "cached"
    try:
        structure = freesasa.Structure(os.path.join(PDB_DIR, f"{pdb_id}.pdb"))
        result = freesasa.calc(structure)
        n = structure.nAtoms()
        coords = np.array([structure.coord(i) for i in range(n)]).reshape(n, 3)
        sasa = np.array([result.atomArea(i) for i in range(n)])

        accessible = coords[sasa > 0]
        if len(accessible) == 0:
            return pdb_id, "no_accessible_atoms"
        tree = cKDTree(accessible)
        dpx = tree.query(coords)[0]
        dpx[sasa > 0] = 0.0

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["chain", "resseq", "icode", "atomname", "sasa", "dpx"])
            for i in range(n):
                res_number = structure.residueNumber(i).strip()
                icode = ""
                if res_number and not res_number[-1].isdigit():
                    icode = res_number[-1]
                    res_number = res_number[:-1].strip()
                writer.writerow([
                    structure.chainLabel(i),
                    res_number,
                    icode,
                    structure.atomName(i).strip(),
                    round(float(sasa[i]), 3),
                    round(float(dpx[i]), 3),
                ])
        return pdb_id, "ok"
    except Exception as exc:
        if os.path.exists(out_path):
            os.remove(out_path)
        return pdb_id, f"error: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = pd.read_csv(METADATA)
    ids = meta[meta.status == "ok"].sort_values("n_atoms").pdb_id.tolist()

    rows = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(run_one, ids)):
            rows.append(res)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(ids)}", flush=True)

    df = pd.DataFrame(rows, columns=["pdb_id", "status"])
    df.to_csv(MANIFEST, index=False)
    print(df.status.value_counts().to_string())


if __name__ == "__main__":
    main()
