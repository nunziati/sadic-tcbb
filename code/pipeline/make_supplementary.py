"""Supplementary Material: full dataset table and dataset summary figures.

Outputs (outputs/supplementary/):
- dataset_table.csv                  full 1000-entry table (make_supplementary_tex.py turns it into LaTeX)
- fig_atoms_hist.pdf, fig_rg_hist.pdf, fig_method_bar.pdf, fig_oligomer_bar.pdf
Also writes the paper's dataset figure: outputs/figures/N_distribution_sample_new.pdf
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

from paths import FIGURES, METADATA, NPZ_DIR, SUPPLEMENTARY as SUPP

os.makedirs(SUPP, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

STYLE = plotstyle.HIST


def nice_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def chain_labels(pdb_id):
    data = np.load(os.path.join(NPZ_DIR, f"{pdb_id}.npz"))
    return " ".join(sorted(set(str(c) for c in np.unique(data["chain"]))))


def main():
    meta = pd.read_csv(METADATA)
    meta = meta[meta.status == "ok"].copy()

    print("Collecting chain labels...")
    meta["chain_labels"] = [chain_labels(p) for p in meta.pdb_id]

    method_map = {
        "X-RAY DIFFRACTION": "X-ray",
        "SOLUTION NMR": "NMR",
        "SOLID-STATE NMR": "NMR",
        "ELECTRON MICROSCOPY": "cryo-EM",
        "ELECTRON CRYSTALLOGRAPHY": "cryo-EM",
        "NEUTRON DIFFRACTION": "Neutron",
    }
    meta["method_short"] = meta.method.map(method_map).fillna("Other")

    table = meta[[
        "pdb_id", "chain_labels", "oligomeric_details", "n_atoms",
        "radius_of_gyration", "method_short", "resolution",
        "polymer_composition", "assembly_count", "source",
    ]].rename(columns={
        "pdb_id": "PDB ID", "chain_labels": "Chains",
        "oligomeric_details": "Oligomeric state", "n_atoms": "N atoms",
        "radius_of_gyration": "Rg (A)", "method_short": "Method",
        "resolution": "Resolution (A)", "polymer_composition": "Composition",
        "assembly_count": "N assemblies", "source": "Provenance",
    })
    table.to_csv(os.path.join(SUPP, "dataset_table.csv"), index=False)

    # figures
    fig, ax = plt.subplots()
    ax.hist(meta.n_atoms, bins=30, **STYLE)
    nice_axes(ax, "Number of atoms", "Frequency"); plotstyle.kfmt(ax.xaxis)
    fig.savefig(os.path.join(SUPP, "fig_atoms_hist.pdf"))
    fig.set_size_inches(plotstyle.W, 1.5)   # paper copy: same width and fonts, less tall
    fig.savefig(os.path.join(FIGURES, "N_distribution_sample_new.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(meta.radius_of_gyration, bins=30, **STYLE)
    nice_axes(ax, "Radius of gyration (Å)", "Frequency")
    fig.savefig(os.path.join(SUPP, "fig_rg_hist.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots()
    counts = meta.method_short.value_counts()
    ax.bar(counts.index, counts.values, **STYLE)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=6.5)
    nice_axes(ax, "Experimental method", "Number of proteins")
    fig.savefig(os.path.join(SUPP, "fig_method_bar.pdf"))
    plt.close(fig)

    def oligo_class(x):
        if not isinstance(x, str):
            return "other"
        for k in ["monomer", "dimer", "trimer", "tetramer", "pentamer", "hexamer",
                  "heptamer", "octamer"]:
            if x.startswith(k[:-2]):  # monomeric, dimeric, ...
                return k
        return "higher/other"
    meta["oligo_class"] = meta.oligomeric_details.map(oligo_class)
    order = ["monomer", "dimer", "trimer", "tetramer", "pentamer", "hexamer",
             "heptamer", "octamer", "higher/other", "other"]
    counts = meta.oligo_class.value_counts().reindex([o for o in order
                                                      if o in meta.oligo_class.unique()])
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, **STYLE)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=6.5)
    nice_axes(ax, "Oligomeric state (assembly 1)", "Number of proteins")
    plt.xticks(rotation=30, ha="right")
    fig.savefig(os.path.join(SUPP, "fig_oligomer_bar.pdf"))
    plt.close(fig)

    print("Method distribution:")
    print(meta.method_short.value_counts().to_string())
    print("Oligomeric classes:")
    print(meta.oligo_class.value_counts().to_string())
    print(f"Rg range: {meta.radius_of_gyration.min():.1f} - {meta.radius_of_gyration.max():.1f} A")
    print(f"Saved table + figures in {SUPP}")


if __name__ == "__main__":
    main()
