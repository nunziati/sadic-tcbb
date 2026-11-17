"""Dataset step 1: download biological assembly 1 (mmCIF) for every protein in the sample.

Single canonical source: https://files.rcsb.org/download/{ID}-assembly1.cif.gz
mmCIF assemblies materialise symmetry copies as extra chains inside each model,
so the load rule downstream is simply: first model, all chains. A normalised PDB
for DEPTH and DPX is generated afterwards by normalize.py.

Files land in data/assemblies/{id}-assembly1.cif.gz.
Manifest: data/assemblies_manifest.csv. Idempotent.
"""

import concurrent.futures as cf
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import ASSEMBLIES as DEST, DATA, SAMPLE

MANIFEST = os.path.join(DATA, "assemblies_manifest.csv")

os.makedirs(DEST, exist_ok=True)


def download_one(pdb_id):
    path = os.path.join(DEST, f"{pdb_id}-assembly1.cif.gz")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pdb_id, "cached"
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}-assembly1.cif.gz"
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=180)
        except requests.RequestException:
            continue
        if resp.status_code == 404:
            return pdb_id, "NOT_FOUND"
        if resp.ok:
            with open(path, "wb") as f:
                f.write(resp.content)
            return pdb_id, "downloaded"
    return pdb_id, "FAILED"


def main():
    ids = pd.read_csv(SAMPLE).pdb_id.tolist()
    results = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, res in enumerate(pool.map(download_one, ids)):
            results.append(res)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(ids)}", flush=True)

    df = pd.DataFrame(results, columns=["pdb_id", "status"])
    df.to_csv(MANIFEST, index=False)
    print(df.status.value_counts().to_string())
    bad = df[df.status.isin(["FAILED", "NOT_FOUND"])]
    if len(bad):
        print("Problematic:", bad.pdb_id.tolist())
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
