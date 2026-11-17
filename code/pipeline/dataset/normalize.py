"""Dataset step 2: normalise every assembly and produce the pipeline inputs + dataset metadata.

Per protein:
- data/normalized/{id}.npz        atom arrays (coords, element, chain, resname, resseq, icode, atomname)
- data/normalized_pdb/{id}.pdb    ATOM-only legacy PDB, input of DEPTH and DPX
Global:
- data/protein_metadata.csv       N atoms, chains, Rg, element stats + RCSB entry metadata

Idempotent: proteins with existing npz+pdb are skipped (metadata still recomputed from npz).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import ASSEMBLIES, METADATA as OUT, NPZ_DIR, PDB_DIR, SAMPLE
from sadicpipe.structure import extract_atoms, load_assembly, load_npz, save_npz, write_normalized_pdb
from dataset.rcsb_api import fetch_entry_and_assembly_info

os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)


def radius_of_gyration(coords):
    center = coords.mean(axis=0)
    return float(np.sqrt(((coords - center) ** 2).sum(axis=1).mean()))


def process_one(pdb_id):
    npz_path = os.path.join(NPZ_DIR, f"{pdb_id}.npz")
    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not (os.path.exists(npz_path) and os.path.exists(pdb_path)):
        cif = os.path.join(ASSEMBLIES, f"{pdb_id}-assembly1.cif.gz")
        st = load_assembly(cif)
        atoms = extract_atoms(st)
        if atoms["coords"].shape[0] == 0:
            raise ValueError("no ATOM-record atoms after normalisation")
        save_npz(atoms, npz_path)
        write_normalized_pdb(st, pdb_path)
    atoms = load_npz(npz_path)

    coords = atoms["coords"]
    elements, counts = np.unique(atoms["element"], return_counts=True)
    return {
        "pdb_id": pdb_id,
        "n_atoms": int(coords.shape[0]),
        "n_chains": int(len(np.unique(atoms["chain"]))),
        "radius_of_gyration": round(radius_of_gyration(coords), 2),
        "elements": " ".join(f"{e}:{c}" for e, c in zip(elements, counts)),
        "has_hydrogens": bool((atoms["element"] == "H").any() or (atoms["element"] == "D").any()),
        "status": "ok",
    }


def main():
    sample = pd.read_csv(SAMPLE)
    ids = sample.pdb_id.tolist()

    rows = []
    for i, pdb_id in enumerate(ids):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows.append(process_one(pdb_id))
        except Exception as exc:  # log and continue; failures reviewed afterwards
            rows.append({"pdb_id": pdb_id, "status": f"ERROR: {exc}"})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(ids)}", flush=True)

    meta = pd.DataFrame(rows)

    # enrich with RCSB entry metadata (method, resolution, oligomeric state, ...)
    info = fetch_entry_and_assembly_info(ids)
    rcsb = pd.DataFrame(info.values())
    keep_cols = [
        "pdb_id", "polymer_composition", "method", "resolution", "assembly_count",
        "assembly_polymer_atom_count", "assembly_chains", "oligomeric_details", "oligomeric_count",
    ]
    rcsb = rcsb[[c for c in keep_cols if c in rcsb.columns]]
    meta = meta.merge(rcsb, on="pdb_id", how="left").merge(
        sample[["pdb_id", "source", "bin"]], on="pdb_id", how="left"
    )
    meta.to_csv(OUT, index=False)

    bad = meta[meta.status != "ok"]
    print(f"OK: {len(meta) - len(bad)}/{len(meta)}")
    if len(bad):
        print(bad[["pdb_id", "status"]].to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
