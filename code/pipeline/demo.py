"""Compute SADIC depth indices for one or more protein structures.

Usage:
    python pipeline/demo.py [--input DIR_OR_FILES ...] [--output DIR] [--resolution 0.5]

Accepts PDB or mmCIF files (optionally gzipped). Each structure is normalised as in the
paper (first model, first alternative location, ATOM records only) and processed with the
configuration proposed in the paper: MorphPC + TranSph, OBB alignment, 0.5 A grid, 1.4 A probe.

For every structure, writes to the output directory:
- {name}_depth.csv   one row per atom: chain, residue, atom name, element, depth index
- {name}_depth.pdb   the normalised structure with the depth index in the B-factor column
                     (colour by B-factor in PyMOL or ChimeraX to visualise it)
and a summary.csv with timings and grid statistics for all structures.

The depth index ranges in [0, 2]: values close to 0 for exposed atoms, larger for buried ones.
"""

import argparse
import csv
import glob
import os
import sys

import gemmi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sadicpipe.runner import SUMMARY_FIELDS, process_atoms
from sadicpipe.structure import extract_atoms, normalize, write_normalized_pdb

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "..", "..", "data", "examples")
EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".ent.gz", ".cif.gz", ".mmcif.gz")


def structure_name(path):
    name = os.path.basename(path)
    for ext in sorted(EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def input_files(inputs):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files += [f for f in sorted(glob.glob(os.path.join(item, "*")))
                      if f.lower().endswith(EXTENSIONS)]
        else:
            files.append(item)
    return files


def write_outputs(st, atoms, depth, out_dir, name):
    with open(os.path.join(out_dir, f"{name}_depth.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chain", "resname", "resseq", "icode", "atomname", "element", "depth_index"])
        for i in range(len(depth)):
            writer.writerow([atoms["chain"][i], atoms["resname"][i], atoms["resseq"][i],
                             atoms["icode"][i], atoms["atomname"][i], atoms["element"][i],
                             f"{depth[i]:.4f}"])

    # same traversal order as structure.extract_atoms: ATOM records of the first model
    i = 0
    model = st[0]
    for chain in model:
        for res in chain:
            if res.het_flag != "A":
                continue
            for atom in res:
                atom.b_iso = float(depth[i])
                i += 1
    assert i == len(depth)
    write_normalized_pdb(st, os.path.join(out_dir, f"{name}_depth.pdb"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", nargs="+", default=[DEFAULT_INPUT],
                        help="structure files and/or directories (default: data/examples/)")
    parser.add_argument("--output", default="results", help="output directory")
    parser.add_argument("--resolution", type=float, default=0.5, help="voxel size in Angstrom")
    parser.add_argument("--probe", type=float, default=1.4, help="solvent probe radius in Angstrom")
    args = parser.parse_args()

    files = input_files(args.input)
    if not files:
        raise SystemExit(f"no structure files found in {args.input}")
    os.makedirs(args.output, exist_ok=True)

    rows = []
    for path in files:
        name = structure_name(path)
        st = normalize(gemmi.read_structure(path))
        atoms = extract_atoms(st)
        depth, row = process_atoms(atoms["coords"], atoms["element"], name,
                                   resolution=args.resolution, probe_radius=args.probe)
        write_outputs(st, atoms, depth, args.output, name)
        total = sum(v for k, v in row.items() if k.startswith("t_"))
        print(f"{name}: {row['N']} atoms, {row['n']} voxels, reference radius "
              f"{row['reference_radius']:.2f} A, depth index mean {depth.mean():.3f}, {total:.1f} s")
        rows.append(row)

    with open(os.path.join(args.output, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
