"""Sensitivity analysis on the grid resolution (paper, Section "Sensitivity to Resolution").

For every configuration found in runs/ablation/ (res{r}_probe{p}):
- mean and total execution time per protein;
- depth-index discrepancy (MAD) with respect to the finest resolution (0.3 A), same probe;
- mean per-protein Pearson correlation with DEPTH (normalised as in the paper) and with DPX.

Outputs:
- analysis/ablation_table.csv, analysis/report_ablation.txt
- outputs/figures/experiments/ablation_resolution.pdf   (paper figure, probe 1.4 A)
- outputs/supplementary/fig_ablation_mad.pdf            (supplementary figure, probe 1.4 A)
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle
plotstyle.apply()
from analysis_lib import load_depth_tool, load_dpx, match_tool
from paths import ABLATION_SUBSET, ANALYSIS as OUT, FIGURES, RUNS, SUPPLEMENTARY

ABLATION_RUNS = os.path.join(RUNS, "ablation")
FINEST = 0.3
DEFAULT_PROBE = 1.4
SUBSET = set(pd.read_csv(ABLATION_SUBSET).pdb_id)


def run_name(res, probe):
    return f"res{res}_probe{probe}"


def configurations():
    """(probe, resolution) pairs available in runs/ablation/."""
    found = []
    for name in os.listdir(ABLATION_RUNS):
        m = re.fullmatch(r"res([\d.]+)_probe([\d.]+)", name)
        if m:
            found.append((float(m.group(2)), float(m.group(1))))
    return sorted(found)


def load_run(res, probe):
    df = pd.read_csv(os.path.join(ABLATION_RUNS, run_name(res, probe), "summary.csv"))
    df = df[(df.status == "ok") & df.pdb_id.isin(SUBSET)]
    return df.drop_duplicates(subset="pdb_id", keep="last").copy()


def depths_of(res, probe, pid):
    return np.load(os.path.join(ABLATION_RUNS, run_name(res, probe), "depth_indexes", f"{pid}.npy"))


def cached(cache, loader, pid):
    if pid not in cache:
        cache[pid] = loader(pid)
    return cache[pid]


def build_table():
    rows = []
    depth_cache, dpx_cache = {}, {}
    for probe, res in configurations():
        df = load_run(res, probe)
        tcols = [c for c in df.columns if c.startswith("t_")]
        df["total_t"] = df[tcols].sum(axis=1)
        ref_ids = set(load_run(FINEST, probe).pdb_id)

        mads, depth_corrs, dpx_corrs = [], [], []
        for pid in sorted(set(df.pdb_id) & ref_ids):
            d = depths_of(res, probe, pid)
            dref = depths_of(FINEST, probe, pid)
            if len(d) == len(dref):
                mads.append(float(np.abs(d - dref).mean()))
            try:
                merged, stats = match_tool(pid, d, cached(depth_cache, load_depth_tool, pid))
                if stats["n_matched"] >= 50:
                    norm = 2.0 * merged["depth"] / merged["depth"].max()
                    depth_corrs.append(pearsonr(norm, merged.sadic)[0])
            except FileNotFoundError:
                pass
            try:
                merged, _ = match_tool(pid, d, cached(dpx_cache, load_dpx, pid))
                if len(merged) > 50:
                    dpx_corrs.append(pearsonr(merged.depth, merged.sadic)[0])
            except FileNotFoundError:
                pass

        rows.append({
            "probe": probe, "resolution": res,
            "n_proteins": len(df),
            "mean_time_s": df.total_t.mean(),
            "total_time_s": df.total_t.sum(),
            "mad_vs_finest": np.mean(mads) if mads else np.nan,
            "mean_depth_pearson": np.mean(depth_corrs) if depth_corrs else np.nan,
            "mean_dpx_pearson": np.mean(dpx_corrs) if dpx_corrs else np.nan,
            "std_dpx_pearson": np.std(dpx_corrs) if dpx_corrs else np.nan,
        })
    return pd.DataFrame(rows)


def paper_figure(t):
    """Top: correlation with DEPTH and DPX; bottom: execution time (log scale)."""
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(plotstyle.W, 3.6))
    a0.plot(t.resolution, t.mean_depth_pearson, marker="o", color=plotstyle.C0, label="DEPTH")
    a0.plot(t.resolution, t.mean_dpx_pearson, marker="s", color=plotstyle.C2, label="DPX")
    a0.set_ylabel("Mean Pearson correlation")
    a0.set_ylim(-1.0, 0.0)
    a0.legend(loc="upper right", ncol=2)
    a0.grid(alpha=0.3)
    a1.plot(t.resolution, t.mean_time_s, marker="o", color=plotstyle.C1)
    a1.set_yscale("log")
    a1.set_ylabel("Mean execution time (s)")
    a1.grid(alpha=0.3, which="major")
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_yticks([1, 10, 100])
    a1.set_yticklabels(["1", "10", "100"])
    for ax in (a0, a1):
        ax.set_xlabel("Resolution $\\alpha$ (Å)")
        ax.set_xticks(t.resolution)
    path = os.path.join(FIGURES, "experiments", "ablation_resolution.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plotstyle.save(fig, path)


def supplementary_figure(t):
    fig, ax = plt.subplots(figsize=(plotstyle.W, plotstyle.H))
    ax.plot(t.resolution, t.mad_vs_finest, marker="o", color=plotstyle.C0)
    ax.set_xlabel("Resolution $\\alpha$ (Å)")
    ax.set_ylabel("MAD vs 0.3 Å")
    ax.set_xticks(t.resolution)
    ax.grid(alpha=0.3)
    os.makedirs(SUPPLEMENTARY, exist_ok=True)
    plotstyle.save(fig, os.path.join(SUPPLEMENTARY, "fig_ablation_mad.pdf"))


def main():
    os.makedirs(OUT, exist_ok=True)
    table = build_table()
    table.to_csv(os.path.join(OUT, "ablation_table.csv"), index=False)
    text = "== Sensitivity: resolution x probe ==\n" + table.to_string(
        index=False, float_format=lambda v: f"{v:.4f}")
    with open(os.path.join(OUT, "report_ablation.txt"), "w") as f:
        f.write(text + "\n")
    print(text)

    default = table[table.probe == DEFAULT_PROBE].sort_values("resolution")
    paper_figure(default)
    supplementary_figure(default)


if __name__ == "__main__":
    main()
