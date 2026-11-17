"""Shared IO and atom-matching utilities for the analysis scripts.

Atom identity for cross-tool matching is (chain, resseq, atomname) — icode appended to
resseq when present. SADIC depth arrays are positional (same order as the npz arrays);
DEPTH and DPX outputs carry their own atom identifiers.
"""

import os
import re

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import METADATA, NPZ_DIR, RUNS


def load_summary(name):
    df = pd.read_csv(os.path.join(RUNS, name, "summary.csv"))
    return df[df.status == "ok"].copy()


def load_depth_array(name, pdb_id):
    return np.load(os.path.join(RUNS, name, "depth_indexes", f"{pdb_id}.npy"))


def atom_keys(pdb_id):
    """Positional atom keys from the normalised npz (index-aligned with depth arrays)."""
    data = np.load(os.path.join(NPZ_DIR, f"{pdb_id}.npz"))
    resseq = data["resseq"].astype(str)
    icode = data["icode"].astype(str)
    reskey = np.char.add(resseq, np.char.strip(icode))
    return pd.DataFrame({
        "chain": data["chain"].astype(str),
        "reskey": reskey,
        "atomname": np.char.strip(data["atomname"].astype(str)),
        "element": data["element"].astype(str),
    })


_DEPTH_RES_RE = re.compile(r"^(.+):(-?\d+[A-Za-z]?)$")


def load_depth_tool(pdb_id):
    """Parse runs/depth/{id}/output-atom.depth -> chain, reskey, atomname, depth.

    Rows with non-numeric depth entries (DEPTH prints '?' for unsolvated atoms) are dropped.
    """
    path = os.path.join(RUNS, "depth", pdb_id, "output-atom.depth")
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"mean(depth)": "depth", "atom-type": "atomname"})
    parsed = df["residue"].astype(str).str.extract(_DEPTH_RES_RE)
    df["chain"] = parsed[0]
    df["reskey"] = parsed[1]
    df["atomname"] = df["atomname"].astype(str).str.strip()
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df = df.dropna(subset=["chain", "reskey", "depth"])
    return df[["chain", "reskey", "atomname", "depth"]]


def load_dpx(pdb_id):
    path = os.path.join(RUNS, "dpx", f"{pdb_id}.csv")
    df = pd.read_csv(path, dtype={"chain": str, "resseq": str, "icode": str})
    df["icode"] = df["icode"].fillna("")
    df["reskey"] = df["resseq"].str.strip() + df["icode"].str.strip()
    df = df.rename(columns={"dpx": "depth"})
    df["atomname"] = df["atomname"].astype(str).str.strip()
    return df[["chain", "reskey", "atomname", "depth", "sasa"]]


def match_tool(pdb_id, sadic_depth, tool_df):
    """Merge SADIC positional depths with a tool's output on (chain, reskey, atomname).

    Returns (merged_df, match_stats). Duplicate keys on either side are dropped
    (altloc remnants / duplicated atom names) to keep the match unambiguous.
    """
    keys = atom_keys(pdb_id)
    keys = keys.assign(sadic=sadic_depth)
    keys = keys.drop_duplicates(subset=["chain", "reskey", "atomname"], keep=False)
    tool = tool_df.drop_duplicates(subset=["chain", "reskey", "atomname"], keep=False)
    merged = keys.merge(tool, on=["chain", "reskey", "atomname"], how="inner")
    stats = {
        "pdb_id": pdb_id,
        "n_sadic": len(sadic_depth),
        "n_tool": len(tool_df),
        "n_matched": len(merged),
        "match_rate": len(merged) / max(1, min(len(keys), len(tool))),
    }
    return merged, stats


def protein_ids(status="ok"):
    meta = pd.read_csv(METADATA)
    return meta[meta.status == status].sort_values("n_atoms").pdb_id.tolist()
