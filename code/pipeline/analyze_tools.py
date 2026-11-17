"""Comparison of SADIC depth indices with DEPTH and DPX.

Protocol (as in the paper): per-protein normalisation d_hat = 2*d/max(d) of the tool's
depth values, overall and per-protein Pearson (and Spearman) correlations, histogram of
the per-protein correlations.

Figures (outputs/figures/):
  histogram_correlation.pdf            DEPTH per-protein correlations
  histogram_correlation_dpx.pdf        DPX per-protein correlations
Numbers in analysis/report_tools.txt; per-protein tables in analysis/per_protein_correlation_*.csv.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle
plotstyle.apply()
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis_lib import load_depth_array, load_depth_tool, load_dpx, match_tool, protein_ids
from paths import ANALYSIS as OUT, FIGURES
from plotstyle import labels as nice_axes, save

os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

SADIC_RUN = "morph_transph_aligned"

report = []


def compare_tool(tool_name, loader):
    ids = protein_ids()
    per_protein = []
    all_sadic, all_tool = [], []
    skipped = []

    for pid in ids:
        try:
            sadic = load_depth_array(SADIC_RUN, pid)
            tool_df = loader(pid)
        except FileNotFoundError:
            skipped.append((pid, "missing output"))
            continue
        merged, stats = match_tool(pid, sadic, tool_df)
        if stats["n_matched"] < 50 or stats["match_rate"] < 0.5:
            skipped.append((pid, f"low match ({stats['n_matched']}, {stats['match_rate']:.2f})"))
            continue

        dmax = merged["depth"].max()
        if dmax <= 0:
            skipped.append((pid, "degenerate tool depths"))
            continue
        merged["tool_norm"] = 2.0 * merged["depth"] / dmax

        pear, pear_p = pearsonr(merged.tool_norm, merged.sadic)
        spear, spear_p = spearmanr(merged.tool_norm, merged.sadic)
        per_protein.append({
            "pdb_id": pid, "n_matched": stats["n_matched"],
            "match_rate": round(stats["match_rate"], 4),
            "pearson": pear, "pearson_p": pear_p,
            "spearman": spear, "spearman_p": spear_p,
        })
        all_sadic.append(merged.sadic.to_numpy())
        all_tool.append(merged.tool_norm.to_numpy())

    pp = pd.DataFrame(per_protein)
    pp.to_csv(os.path.join(OUT, f"per_protein_correlation_{tool_name}.csv"), index=False)

    sadic_cat = np.concatenate(all_sadic)
    tool_cat = np.concatenate(all_tool)
    pear_all, p_all = pearsonr(tool_cat, sadic_cat)
    spear_all, sp_all = spearmanr(tool_cat, sadic_cat)

    report.append(f"== {tool_name.upper()} comparison ==")
    report.append(f"proteins compared: {len(pp)} / {len(ids)}  (skipped: {len(skipped)})")
    report.append(f"matched atoms: {len(sadic_cat)}")
    report.append(f"overall pearson: {pear_all:.4f} (p={p_all:.2e})  "
                  f"spearman: {spear_all:.4f} (p={sp_all:.2e})")
    report.append(f"per-protein pearson: mean={pp.pearson.mean():.4f} std={pp.pearson.std():.4f} "
                  f"min={pp.pearson.min():.4f} max={pp.pearson.max():.4f}")
    report.append(f"strongest anticorrelation: {pp.loc[pp.pearson.idxmin(), 'pdb_id']} "
                  f"({pp.pearson.min():.4f})")
    report.append(f"weakest correlation: {pp.loc[pp.pearson.idxmax(), 'pdb_id']} "
                  f"({pp.pearson.max():.4f})")
    if skipped:
        report.append(f"skipped: {skipped[:15]}{' ...' if len(skipped) > 15 else ''}")
    report.append("")

    fig, ax = plt.subplots()
    ax.hist(pp.pearson, bins=50, **plotstyle.HIST)
    nice_axes(ax, "Pearson correlation coefficient", "Frequency")
    fname = ("histogram_correlation.pdf" if tool_name == "depth"
             else f"histogram_correlation_{tool_name}.pdf")
    save(fig, os.path.join(FIGURES, fname))
    return pp


def main():
    compare_tool("depth", load_depth_tool)
    compare_tool("dpx", load_dpx)

    with open(os.path.join(OUT, "report_tools.txt"), "w") as f:
        f.write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
