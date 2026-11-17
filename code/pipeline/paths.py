"""Directory layout shared by all scripts.

Data and outputs live next to code/ (on Code Ocean: /data, /results); override with:
- SADIC_WORKSPACE  root for data/, runs/, analysis/ and outputs/ (default: repository root);
- SADIC_DEPTH_DIR  directory containing the DEPTH executable and water_models/
  (default: external/depth). DEPTH is third-party software and is not distributed here.
"""

import os

WORKSPACE = os.path.abspath(
    os.environ.get("SADIC_WORKSPACE", os.path.join(os.path.dirname(__file__), "..", "..")))

DATA = os.path.join(WORKSPACE, "data")
SAMPLE = os.path.join(DATA, "sample_1000.csv")
METADATA = os.path.join(DATA, "protein_metadata.csv")
ABLATION_SUBSET = os.path.join(DATA, "ablation_subset.csv")
ASSEMBLIES = os.path.join(DATA, "assemblies")      # downloaded mmCIF assemblies
NPZ_DIR = os.path.join(DATA, "normalized")         # atom arrays, input of SADIC
PDB_DIR = os.path.join(DATA, "normalized_pdb")     # ATOM-only PDB, input of DEPTH and DPX

RUNS = os.path.join(WORKSPACE, "runs")             # raw per-protein outputs
ANALYSIS = os.path.join(WORKSPACE, "analysis")     # tables and text reports
FIGURES = os.path.join(WORKSPACE, "outputs", "figures")
SUPPLEMENTARY = os.path.join(WORKSPACE, "outputs", "supplementary")

DEPTH_DIR = os.path.abspath(
    os.environ.get("SADIC_DEPTH_DIR", os.path.join(WORKSPACE, "external", "depth")))
