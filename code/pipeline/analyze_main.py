"""Main analysis: the numbers and figures of the paper's Results section.

Figures are written to outputs/figures/ with the file names used in the paper.
All numbers are collected in analysis/report_main.txt.

Sections:
  1. AABB vs OBB alignment (volume reduction)
  2. TranSph vs ReconSph discrepancies (MorphPC and InflPC)
  3. MorphPC vs InflPC (volume decrease, depth indices, reference radii)
  4. Execution time + N-vs-n relationship
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

from analysis_lib import load_summary, load_depth_array, protein_ids
from paths import ANALYSIS as OUT, FIGURES
from plotstyle import labels as nice_axes, save

os.makedirs(OUT, exist_ok=True)

report = []


def fig_path(*parts):
    path = os.path.join(FIGURES, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def aligned_depth_frames(run_a, run_b, ids):
    """Long depth-index vectors for two runs, restricted to proteins in both."""
    da, db, pid_index = [], [], []
    for pid in ids:
        try:
            a = load_depth_array(run_a, pid)
            b = load_depth_array(run_b, pid)
        except FileNotFoundError:
            continue
        if len(a) != len(b):
            continue
        da.append(a)
        db.append(b)
        pid_index.extend([pid] * len(a))
    return np.concatenate(da), np.concatenate(db), np.array(pid_index)


def main():
    ids = protein_ids()
    id_set = set(ids)

    def load(name):
        # restrict to the dataset in protein_metadata.csv (a summary may hold extra rows)
        df = load_summary(name).set_index("pdb_id")
        df = df[df.index.isin(id_set)]
        df = df[~df.index.duplicated(keep="last")]
        return df

    ms_tr = load("morph_transph_aligned")
    load("morph_reconsph_aligned")
    in_tr = load("infl_transph_aligned")
    in_re = load("infl_reconsph_aligned")
    unal = load("morph_transph_unaligned")

    # ---------- 1. AABB vs OBB ----------
    common = ms_tr.index.intersection(unal.index)
    n_al = ms_tr.loc[common, "n"]
    n_un = unal.loc[common, "n"]
    reduction = np.maximum((n_un - n_al) / n_un * 100, 0)
    report.append("== 1. Bounding box (AABB vs OBB) ==")
    report.append(f"proteins: {len(common)}")
    report.append(f"volume reduction %: mean={reduction.mean():.2f} std={reduction.std():.2f} "
                  f"min={reduction.min():.2f} max={reduction.max():.2f}")
    report.append(f"max-reduction protein: {reduction.idxmax()} ({reduction.max():.2f}%)")
    report.append("")

    N = ms_tr.loc[common, "N"]
    fig, ax = plt.subplots()
    ax.scatter(N, n_un, color=plotstyle.C0, label="AABB", s=1.5, lw=0)
    ax.scatter(N, n_al, color=plotstyle.C1, label="OBB", s=1.5, lw=0)
    nice_axes(ax, "Number of atoms $N$", "Number of voxels $n$"); plotstyle.kfmt(ax.xaxis, ax.yaxis)
    ax.legend(markerscale=4)
    save(fig, fig_path("experiments", "protein_rotation", "bounding_box_scatter.pdf"))

    fig, ax = plt.subplots()
    ax.hist(reduction, bins=30, **plotstyle.HIST)
    nice_axes(ax, "Volume reduction (%)", "Frequency")
    save(fig, fig_path("experiments", "protein_rotation", "volume_reduction_histogram.pdf"))

    # ---------- 2. TranSph vs ReconSph ----------
    for model, run_tr, run_re in [("MorphPC", "morph_transph_aligned", "morph_reconsph_aligned"),
                                  ("InflPC", "infl_transph_aligned", "infl_reconsph_aligned")]:
        d_tr, d_re, pid_idx = aligned_depth_frames(run_tr, run_re, ids)
        sd = d_re - d_tr
        ad = np.abs(sd)
        pw = pd.DataFrame({"pid": pid_idx, "sd": sd, "ad": ad}).groupby("pid").mean()
        report.append(f"== 2. TranSph vs ReconSph ({model}) ==")
        report.append(f"atoms: {len(sd)}  proteins: {len(pw)}")
        report.append(f"MSD={sd.mean():.6f} MAD={ad.mean():.6f} "
                      f"PW-MSD={pw.sd.mean():.6f} PW-MAD={pw.ad.mean():.6f}")
        report.append("")

        if model != "MorphPC":  # the paper shows the histograms for MorphPC only
            continue
        folder = "morphpc_using_translated_sphere_metrics_histograms"
        for values, label, xlim in [
            (sd, "MSD", (-0.05, 0.05)), (ad, "MAD", (-0.05, 0.05)),
            (pw.sd, "PW-MSD", (-0.005, 0.005)), (pw.ad, "PW-MAD", (-0.05, 0.05)),
        ]:
            fig, ax = plt.subplots()
            bins = np.linspace(xlim[0], xlim[1], 50)
            ax.hist(np.clip(values, xlim[0], xlim[1]), bins=bins, **plotstyle.HIST)
            nice_axes(ax, label, "Frequency")
            ax.set_xlim(xlim)
            plotstyle.kfmt(ax.yaxis)
            save(fig, fig_path("experiments", "translated_sphere", folder, f"{label}.pdf"))

    # ---------- 3. MorphPC vs InflPC ----------
    common = ms_tr.index.intersection(in_tr.index)
    vol_m = ms_tr.loc[common, "volume_filled"]
    vol_i = in_tr.loc[common, "volume_filled"]
    decrease = (vol_i - vol_m) / vol_i * 100
    report.append("== 3. MorphPC vs InflPC ==")
    report.append(f"volume decrease %: mean={decrease.mean():.2f} std={decrease.std():.2f} "
                  f"min={decrease.min():.2f} max={decrease.max():.2f}")
    report.append(f"max-decrease protein: {decrease.idxmax()} ({decrease.max():.2f}%)")

    fig, ax = plt.subplots()
    ax.hist(decrease, bins=50, **plotstyle.HIST)
    nice_axes(ax, "Volume reduction (%)", "Frequency")
    save(fig, fig_path("experiments", "morphological",
                       "morphpc_volume_decrease_translated_histogram.pdf"))

    d_m, d_i, pid_idx = aligned_depth_frames("morph_transph_aligned", "infl_transph_aligned", ids)
    corr = np.corrcoef(d_m, d_i)[0, 1]
    report.append(f"depth indices correlation (MorphPC vs InflPC, TranSph): {corr:.4f}")

    fig, ax = plt.subplots()
    from matplotlib.colors import LogNorm
    hh = ax.hist2d(d_m, d_i, bins=200, range=[[0, 2], [0, 2]], cmap="Blues", norm=LogNorm(), rasterized=True)
    fig.colorbar(hh[3], ax=ax, label="Number of atoms", pad=0.02)
    ax.set_aspect("equal")
    nice_axes(ax, "Depth index (MorphPC)", "Depth index (InflPC)")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    save(fig, fig_path("experiments", "morphological", "depth_indices_translated_scatter.pdf"))

    fig, ax = plt.subplots()
    ax.hist(d_m, bins=50, alpha=0.55, color=plotstyle.C0, label="MorphPC")
    ax.hist(d_i, bins=50, alpha=0.55, color=plotstyle.C2, label="InflPC")
    nice_axes(ax, "Depth index", "Frequency"); plotstyle.kfmt(ax.yaxis)
    ax.set_xlim(0, 2)
    ax.legend(markerscale=4)
    save(fig, fig_path("experiments", "morphological", "depth_indices_translated_histogram.pdf"))

    rr_reduction = in_tr.loc[common, "reference_radius"] - ms_tr.loc[common, "reference_radius"]
    report.append(f"reference radius reduction (A): mean={rr_reduction.mean():.3f} "
                  f"min={rr_reduction.min():.3f} max={rr_reduction.max():.3f} "
                  f"negative (increase) count={(rr_reduction < 0).sum()}")
    report.append("")

    fig, ax = plt.subplots()
    ax.hist(rr_reduction, bins=70, **plotstyle.HIST)
    nice_axes(ax, "Reference radius reduction (Å)", "Frequency")
    ax.set_xlim(-2.5, 5.0)
    save(fig, fig_path("experiments", "morphological",
                       "reference_radius_diff_translated_histogram.pdf"))

    # ---------- 4. Execution time ----------
    tcols = ["t_alignment", "t_discretization", "t_closing", "t_cavity_fill",
             "t_reference_radius", "t_indices"]
    for df in (ms_tr, in_re):
        df["execution_time"] = df[tcols].sum(axis=1)

    new_total = ms_tr.execution_time.sum() / 3600
    old_total = in_re.execution_time.sum() / 3600
    new_idx = ms_tr.t_indices.sum() / 3600
    old_idx = in_re.t_indices.sum() / 3600
    report.append("== 4. Execution time ==")
    report.append(f"MorphPC+TranSph total: {new_total:.2f} h  (index: {new_idx:.2f} h)")
    report.append(f"InflPC+ReconSph total: {old_total:.2f} h  (index: {old_idx:.2f} h)")
    report.append(f"speedup total: x{old_total / new_total:.2f}  "
                  f"index: x{old_idx / new_idx:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(in_re["N"], in_re.execution_time, label="InflPC + ReconSph", s=3, lw=0, color=plotstyle.C0)
    ax.scatter(ms_tr["N"], ms_tr.execution_time, label="MorphPC + TranSph", s=3, lw=0,
               color=plotstyle.C1, marker="^")
    nice_axes(ax, "Number of atoms $N$", "Execution time (s)"); plotstyle.kfmt(ax.xaxis)
    ax.legend(markerscale=4)
    save(fig, fig_path("experiments", "execution_time", "execution_time_vs_n_atoms_combined.pdf"))

    fig, ax = plt.subplots()
    ax.scatter(in_re["n"], in_re.execution_time, label="InflPC + ReconSph", s=3, lw=0, color=plotstyle.C0)
    ax.scatter(ms_tr["n"], ms_tr.execution_time, label="MorphPC + TranSph", s=3, lw=0,
               color=plotstyle.C1, marker="^")
    nice_axes(ax, "Number of voxels $n$", "Execution time (s)"); plotstyle.kfmt(ax.xaxis)
    ax.legend(markerscale=4)
    save(fig, fig_path("experiments", "execution_time", "execution_time_vs_n_voxels_combined.pdf"))

    # N vs n with linear fit
    m, b = np.polyfit(ms_tr["N"], ms_tr["n"], 1)
    corr_nn = np.corrcoef(ms_tr["N"], ms_tr["n"])[0, 1]
    report.append(f"N vs n (aligned): pearson={corr_nn:.4f}  linear fit n = {m:.1f} N + {b:.1f}")
    fig, ax = plt.subplots()
    ax.scatter(ms_tr["N"], ms_tr["n"], s=1.5, lw=0, color=plotstyle.C0, label="Proteins")
    xs = np.linspace(ms_tr["N"].min(), ms_tr["N"].max(), 100)
    ax.plot(xs, m * xs + b, label="Linear fit", color=plotstyle.C1)
    nice_axes(ax, "Number of atoms $N$", "Number of voxels $n$"); plotstyle.kfmt(ax.xaxis, ax.yaxis)
    ax.legend(markerscale=4)
    save(fig, fig_path("experiments", "execution_time", "aligned_n_vs_n_scatter_with_fit.pdf"))

    with open(os.path.join(OUT, "report_main.txt"), "w") as f:
        f.write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
