"""Pairwise agreement of SADIC, DEPTH and DPX on the same atoms (paper, Section "Comparison with DPX").

For every protein with both DEPTH and DPX outputs, atoms are matched across the three
measures and the following are computed:
- per protein: Pearson and Spearman correlations SADIC-DPX, SADIC-DEPTH, DEPTH-DPX, the
  same restricted to buried atoms (DPX > 0), and the fraction of atoms with DPX = 0;
- pooled over all matched atoms: Pearson correlations, with DEPTH normalised per protein
  as in the paper (2 * d / max(d)).

Outputs: analysis/dpx_vs_depth.csv, analysis/report_dpx_vs_depth.txt,
analysis/report_pooled_depth_dpx.txt.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_lib import load_depth_array, load_depth_tool, load_dpx, match_tool
from paths import ANALYSIS as OUT, METADATA, RUNS

SADIC_RUN = "morph_transph_aligned"
KEYS = ["chain", "reskey", "atomname"]


def has_outputs(pid):
    return (os.path.exists(os.path.join(RUNS, "depth", pid, "output-atom.depth"))
            and os.path.exists(os.path.join(RUNS, "dpx", f"{pid}.csv")))


def matched_atoms(pid):
    """SADIC, DEPTH and DPX values on the atoms matched by all three, or None."""
    s = load_depth_array(SADIC_RUN, pid)
    m_dpx, _ = match_tool(pid, s, load_dpx(pid))
    m_depth, _ = match_tool(pid, s, load_depth_tool(pid))
    mm = m_dpx.merge(m_depth, on=KEYS, suffixes=("_dpx", "_depth"))
    return mm if len(mm) >= 50 else None


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(METADATA)
    ids = [p for p in sorted(set(meta[meta.status == "ok"].pdb_id)) if has_outputs(p)]

    rows = []
    pooled_s, pooled_depth, pooled_dpx = [], [], []
    for i, pid in enumerate(ids):
        try:
            mm = matched_atoms(pid)
        except Exception as exc:  # report and continue with the other proteins
            print(pid, "ERR", exc)
            continue
        if mm is None:
            continue
        dpx, dep, sad = mm.depth_dpx.values, mm.depth_depth.values, mm.sadic_dpx.values

        r = dict(pdb_id=pid, n=len(mm), frac_dpx_zero=(dpx == 0).mean(),
                 sadic_dpx=pearsonr(dpx, sad)[0], sadic_depth=pearsonr(dep, sad)[0],
                 depth_dpx=pearsonr(dpx, dep)[0],
                 sp_sadic_dpx=spearmanr(dpx, sad)[0], sp_sadic_depth=spearmanr(dep, sad)[0],
                 sp_depth_dpx=spearmanr(dpx, dep)[0])
        buried = dpx > 0
        if buried.sum() > 50:
            r.update(sadic_dpx_buried=pearsonr(dpx[buried], sad[buried])[0],
                     sadic_depth_buried=pearsonr(dep[buried], sad[buried])[0],
                     depth_dpx_buried=pearsonr(dpx[buried], dep[buried])[0])
        rows.append(r)

        pooled_s.append(sad)
        pooled_dpx.append(dpx)
        pooled_depth.append(2.0 * dep / dep.max())
        if i % 100 == 0:
            print(f"{i}/{len(ids)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dpx_vs_depth.csv"), index=False)
    per_protein = (f"proteins: {len(df)}\n"
                   + df.drop(columns="pdb_id").agg(["mean", "std", "median"]).round(3).to_string()
                   + "\n")
    with open(os.path.join(OUT, "report_dpx_vs_depth.txt"), "w") as f:
        f.write(per_protein)
    print(per_protein)

    S, D, P = map(np.concatenate, (pooled_s, pooled_depth, pooled_dpx))
    pooled = (f"atoms: {len(S)}\n"
              f"pooled SADIC-DPX: {np.corrcoef(S, P)[0, 1]:.3f}\n"
              f"pooled SADIC-DEPTH(norm): {np.corrcoef(S, D)[0, 1]:.3f}\n"
              f"pooled DEPTH(norm)-DPX: {np.corrcoef(D, P)[0, 1]:.3f}\n"
              f"frac DPX==0: {(P == 0).mean():.3f}\n")
    with open(os.path.join(OUT, "report_pooled_depth_dpx.txt"), "w") as f:
        f.write(pooled)
    print(pooled)


if __name__ == "__main__":
    main()
