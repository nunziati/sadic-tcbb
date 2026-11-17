"""Loading and normalisation of biological-assembly structures (mmCIF via gemmi).

Normalisation rules (as described in the paper):
- first model only (assembly copies are separate chains inside each model of the
  RCSB assembly mmCIF; extra models are NMR conformers);
- first alternative conformation only (gemmi remove_alternative_conformations);
- ATOM records only: HETATM (ligands, waters, metals, modified residues such as MSE)
  are excluded from the solid model;
- hydrogens kept when present in the deposition.
"""

import os

import gemmi
import numpy as np


def load_assembly(cif_gz_path):
    """Load an RCSB assembly mmCIF (.cif.gz), normalise, return a gemmi Structure."""
    return normalize(gemmi.read_structure(cif_gz_path, format=gemmi.CoorFormat.Mmcif))


def normalize(st):
    """Apply the normalisation rules above to a gemmi Structure (in place) and return it."""
    st.setup_entities()
    while len(st) > 1:  # keep first model only
        del st[1]
    st.remove_alternative_conformations()
    st.remove_empty_chains()
    # single-character chain names, applied BEFORE extraction so that the npz arrays
    # and the normalised PDB (input of DEPTH/DPX) share the same chain naming
    st.shorten_chain_names()
    return st


def extract_atoms(st):
    """Extract ATOM-record atoms from a normalised Structure.

    Returns a dict of numpy arrays: coords (N,3), element, chain, resname, resseq,
    icode, atomname.
    """
    coords, element, chain, resname, resseq, icode, atomname = [], [], [], [], [], [], []
    model = st[0]
    for ch in model:
        for res in ch:
            if res.het_flag != "A":
                continue
            for atom in res:
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                element.append(atom.element.name.upper())
                chain.append(ch.name)
                resname.append(res.name)
                resseq.append(res.seqid.num)
                icode.append(res.seqid.icode.strip())
                atomname.append(atom.name)
    return {
        "coords": np.array(coords, dtype=np.float64),
        "element": np.array(element),
        "chain": np.array(chain),
        "resname": np.array(resname),
        "resseq": np.array(resseq, dtype=np.int32),
        "icode": np.array(icode),
        "atomname": np.array(atomname),
    }


def write_normalized_pdb(st, out_path):
    """Write the ATOM-only normalised structure as a legacy PDB (input of DEPTH and DPX)."""
    st2 = st.clone()
    model = st2[0]
    for ch in model:
        to_del = [i for i, res in enumerate(ch) if res.het_flag != "A"]
        for i in reversed(to_del):
            del ch[i]
    st2.remove_empty_chains()
    st2.write_pdb(out_path, gemmi.PdbWriteOptions(cryst1_record=False, end_record=True))


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_npz(atoms, npz_path):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **atoms)
