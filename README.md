# SADIC: protein atom depth with morphological transformations

Code and data for the paper

> G. Nunziati, S. Marziali, F. Romagnoli, S. Brunetti,
> "SADIC: Improving Protein Atom Depth Computation Using Morphological Transformations",
> *IEEE Transactions on Computational Biology and Bioinformatics*, 2026.

SADIC computes a volume-based depth index for every atom of a protein. The protein is
represented as a solid on a voxel grid, refined with a morphological closing (MorphPC);
the depth index of an atom measures how much of a reference sphere centred on the atom is
occupied by the protein (TranSph). The index ranges in [0, 2]: close to 0 for exposed
atoms, larger for buried ones.

This repository contains the implementation used for all the experiments of the paper
(`code/pipeline/sadicpipe`), the scripts that reproduce every number and figure of the
Results section and of the Supplementary Material, and the list of the 1000 PDB entries
of the dataset.

## Quick start

Requires Python 3.12.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./code/run                  # SADIC on the structures in data/examples/, results in code/results/
```

To process your own structures (PDB or mmCIF, optionally gzipped):

```bash
python code/pipeline/demo.py --input my_protein.cif another.pdb --output my_results
```

For each structure the demo writes `{name}_depth.csv` (one row per atom) and
`{name}_depth.pdb`, the normalised structure with the depth index in the B-factor column,
ready to be coloured in PyMOL or ChimeraX. Processing takes from a fraction of a second to a
few seconds per protein. The examples are three entries of the dataset (1G2B, 3LEE, 2A3W):
their depth indices coincide with those of the paper's runs.

Structures are normalised as in the paper: first model, first alternative location,
ATOM records only (ligands, waters, ions and modified residues are excluded), hydrogens
kept when deposited.

## Repository layout

```
code/
  pipeline/
    sadicpipe/               SADIC implementation
      vdw.py                 Van der Waals radii, probe radius
      structure.py           loading and normalisation of structures (gemmi)
      align.py               oriented-bounding-box alignment (genetic algorithm + Nelder-Mead)
      voxel.py               discretisation, closing, cavity filling, reference radius, depth indices
      runner.py              per-protein pipeline and batch runs
    demo.py                  depth indices for arbitrary structures
    dataset/                 download and normalisation of the dataset
    run_*.py                 experimental runs (SADIC, DEPTH, DPX, resolution sensitivity)
    analyze_*.py, make_*.py  numbers, figures and supplementary material
    paths.py                 directory layout
  run                        demo entry point (also used by Code Ocean)
  run_all.sh                 full reproduction, in order
data/
  sample_1000.csv          the 1000 PDB entries of the dataset, with size bin
  protein_metadata.csv     per-entry metadata (atoms, chains, method, resolution, ...)
  ablation_subset.csv      the 150 entries used for the resolution sensitivity analysis
  exclusions.csv           entries excluded during curation and their replacements
  examples/                three assemblies for the demo
```

The layout follows the [Code Ocean](https://codeocean.com) convention (`code/`, `data/`);
the scripts in `code/pipeline/` write their outputs to `runs/`, `analysis/` and `outputs/`
next to `code/` (override with `SADIC_WORKSPACE`).

## Reproducing the paper

`code/run_all.sh` runs every step in order; on a 16-core workstation the whole reproduction took
about 15 hours. The steps:

| Step | Scripts | Output | Notes |
|---|---|---|---|
| Dataset | `dataset/download_assemblies.py`, `dataset/normalize.py` | `data/` | ~600 MB download from RCSB, ~2.8 GB in total after normalisation |
| SADIC runs | `run_experiments.py <config>` | `runs/<config>/` | 5 configurations; ~2 h for MorphPC + TranSph, ~3.5 h for InflPC + ReconSph (1 worker) |
| DPX | `run_dpx.py` | `runs/dpx/` | FreeSASA, minutes |
| DEPTH | `run_depth.py` | `runs/depth/` | requires DEPTH (see below); ~47 h of single-process time, ~12 h with 4 workers |
| Resolution sensitivity | `run_ablation.py` | `runs/ablation/` | 150 proteins x 5 resolutions, ~2.5 h with 2 workers; 0.3 A grids need several GB of RAM per worker |
| Analyses | `analyze_main.py`, `analyze_tools.py`, `analyze_dpx_depth.py`, `analyze_ablation.py` | `analysis/`, `outputs/figures/` | |
| Supplementary | `make_supplementary.py`, `make_supplementary_tex.py` | `outputs/supplementary/` | |
| Protein renderings | `make_protein_figures.py` | `outputs/figures/notables/` | needs `requirements-figures.txt` (PyMOL) |

Every run script is resumable: proteins already processed are skipped. The alignment
heuristic is stochastic, with a seed derived from the PDB identifier, so the results are
deterministic.

The paper's sections map to the scripts as follows:

| Paper | Script |
|---|---|
| Dataset (Fig. 5, Supplementary Material) | `make_supplementary.py`, `make_supplementary_tex.py` |
| Protein rotation, TranSph vs ReconSph, MorphPC vs InflPC, execution time | `analyze_main.py` |
| Comparison with DEPTH, comparison with DPX | `analyze_tools.py`, `analyze_dpx_depth.py` |
| Sensitivity to resolution | `run_ablation.py`, `analyze_ablation.py` |
| Protein renderings (7RCO, 6K43, 2A3W/5T4O, 7LJS) | `make_protein_figures.py` |

### Dataset

The dataset is defined by `data/sample_1000.csv`: 1000 protein-only PDB entries, processed
as their first biological assembly, with 200 to 40,000 modelled atoms, drawn by stratified
sampling on the number of atoms (30 bins). The selection queries are documented in
`code/pipeline/dataset/rcsb_api.py`. The structures are downloaded from RCSB at run time; since
entries can be revised by the PDB, a fresh download may differ slightly from the files used
in the paper (downloaded in August–September 2026).

### Reference methods

- **DEPTH** (Tan et al.) is third-party software and is not distributed here. Obtain it from
  its authors and place the `DEPTH` executable and its `water_models/` directory in
  `external/depth/`, or set `SADIC_DEPTH_DIR`. It is run as
  `DEPTH -i X.pdb -o out -w water_models/spc216.pdb -survive 5 -n 200`.
- **DPX** (Pintar et al.), the depth index of PSAIA, is computed from its definition with
  FreeSASA (`run_dpx.py`), since PSAIA is no longer available in a runnable form.

## Citation

If you use this code, please cite the paper above. DOI: to be added on publication.

## License

MIT, see `LICENSE`. The example structures in `data/examples/` are from the
[RCSB Protein Data Bank](https://www.rcsb.org/) (CC0).
