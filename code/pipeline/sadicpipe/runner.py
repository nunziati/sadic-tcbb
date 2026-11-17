"""Per-protein pipeline orchestration and batch running with resume support.

A run configuration mirrors the paper's experimental setups:
- model: 'morph' (VdW radii + closing + cavity fill)  or  'infl' (radii inflated by the
  probe, no closing, cavity fill only);
- indices: 'transph' or 'reconsph';
- alignment: 'approximate' or 'none';
- resolution (A), probe radius (A), VdW table.

Outputs per protein (in <run_dir>/):
- depth_indexes/{id}.npy      float32 depth index per atom (same order as the npz arrays)
- summary.csv                 one row per protein: timings, volumes, radii, sizes, status
"""

import csv
import os
import time
import zlib

import numpy as np

from . import vdw
from .align import align
from .structure import load_npz
from .voxel import (
    compute_extremes,
    depth_indices_reconstructed,
    depth_indices_translated,
    discretize,
    fill_cavities,
    morphological_closing,
    reference_radius,
)

SUMMARY_FIELDS = [
    "pdb_id", "status", "model", "indices", "alignment", "resolution", "probe_radius",
    "N", "n", "obb_volume",
    "t_alignment", "t_discretization", "t_closing", "t_cavity_fill",
    "t_reference_radius", "t_indices",
    "volume_discretization", "volume_closing", "volume_filled",
    "n_background_components", "n_filled_voxels", "reference_radius",
    "error",
]


def protein_seed(pdb_id):
    """Deterministic per-protein seed for the stochastic OBB alignment."""
    return zlib.crc32(pdb_id.lower().encode()) & 0xFFFFFFFF


def process_protein(npz_path, **config):
    """Run SADIC on a normalised protein stored as npz (see structure.save_npz)."""
    atoms = load_npz(npz_path)
    pdb_id = os.path.splitext(os.path.basename(npz_path))[0]
    return process_atoms(atoms["coords"], atoms["element"], pdb_id, **config)


def process_atoms(coords, elements, pdb_id, model="morph", indices="transph",
                  alignment="approximate", resolution=0.5,
                  probe_radius=vdw.PROBE_RADIUS_WATER, vdw_table=None, seed=None):
    """Run SADIC on one protein.

    coords: (N, 3) atom coordinates in Angstrom; elements: N element symbols;
    pdb_id: identifier, also used to derive the seed of the alignment heuristic.
    Returns (depth indices, one per atom in input order; dict of timings and statistics).
    The defaults are the configuration proposed in the paper (MorphPC + TranSph,
    OBB alignment, 0.5 A grid, 1.4 A probe).
    """
    if seed is None:
        seed = protein_seed(pdb_id)

    radii = np.array([vdw.radius_of(e, vdw_table) for e in elements])
    if model == "infl":
        radii = radii + probe_radius

    row = {
        "pdb_id": pdb_id, "model": model, "indices": indices, "alignment": alignment,
        "resolution": resolution, "probe_radius": probe_radius, "N": len(coords),
    }

    t0 = time.perf_counter()
    coords, obb_volume = align(coords, method=alignment, seed=seed)
    t1 = time.perf_counter()
    row["t_alignment"] = t1 - t0
    row["obb_volume"] = obb_volume

    extremes = compute_extremes(coords, radii, resolution)

    t0 = time.perf_counter()
    solid = discretize(coords, radii, extremes, resolution)
    t1 = time.perf_counter()
    row["t_discretization"] = t1 - t0
    row["volume_discretization"] = int(solid.sum())
    row["n"] = int(np.prod(solid.shape))

    if model == "morph":
        t0 = time.perf_counter()
        solid = morphological_closing(solid, resolution, probe_radius)
        t1 = time.perf_counter()
        row["t_closing"] = t1 - t0
        row["volume_closing"] = int(solid.sum())
    else:
        row["t_closing"] = 0.0
        row["volume_closing"] = row["volume_discretization"]

    t0 = time.perf_counter()
    solid, n_components, n_filled = fill_cavities(solid)
    t1 = time.perf_counter()
    row["t_cavity_fill"] = t1 - t0
    row["n_background_components"] = n_components
    row["n_filled_voxels"] = n_filled
    row["volume_filled"] = int(solid.sum())

    t0 = time.perf_counter()
    ref_radius, _ = reference_radius(solid, coords, extremes, resolution)
    t1 = time.perf_counter()
    row["t_reference_radius"] = t1 - t0
    row["reference_radius"] = ref_radius

    depth_fn = depth_indices_translated if indices == "transph" else depth_indices_reconstructed
    t0 = time.perf_counter()
    depth = depth_fn(solid, coords, ref_radius, extremes, resolution)
    t1 = time.perf_counter()
    row["t_indices"] = t1 - t0

    row["status"] = "ok"
    return depth, row


def run_batch(pdb_ids, npz_dir, run_dir, config, workers=1, log=print):
    """Run a configuration over a list of proteins, with resume support.

    config: dict with keys model / indices / alignment / resolution / probe_radius.
    Serial when workers == 1 (clean timings); process pool otherwise.
    """
    os.makedirs(os.path.join(run_dir, "depth_indexes"), exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.csv")

    done = set()
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            done = {r["pdb_id"] for r in csv.DictReader(f) if r.get("status") == "ok"}
    todo = [p for p in pdb_ids if p not in done]
    log(f"[{os.path.basename(run_dir)}] {len(done)} done, {len(todo)} to go")

    write_header = not os.path.exists(summary_path)
    summary_file = open(summary_path, "a", newline="")
    writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    def handle_result(pdb_id, depth, row):
        if depth is not None:
            np.save(os.path.join(run_dir, "depth_indexes", f"{pdb_id}.npy"), depth)
        writer.writerow(row)
        summary_file.flush()

    if workers == 1:
        for i, pdb_id in enumerate(todo):
            try:
                depth, row = process_protein(os.path.join(npz_dir, f"{pdb_id}.npz"), **config)
            except Exception as exc:
                depth, row = None, {"pdb_id": pdb_id, "status": "error", "error": repr(exc)}
            handle_result(pdb_id, depth, row)
            if (i + 1) % 25 == 0:
                log(f"[{os.path.basename(run_dir)}] {i + 1}/{len(todo)}")
    else:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_protein, os.path.join(npz_dir, f"{p}.npz"), **config): p
                for p in todo
            }
            for i, fut in enumerate(cf.as_completed(futures)):
                pdb_id = futures[fut]
                try:
                    depth, row = fut.result()
                except Exception as exc:
                    depth, row = None, {"pdb_id": pdb_id, "status": "error", "error": repr(exc)}
                handle_result(pdb_id, depth, row)
                if (i + 1) % 25 == 0:
                    log(f"[{os.path.basename(run_dir)}] {i + 1}/{len(todo)}")

    summary_file.close()
