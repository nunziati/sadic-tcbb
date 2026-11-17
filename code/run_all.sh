#!/usr/bin/env bash
# Full reproduction of the experiments of the paper, in order. Long: see README for times.
# DEPTH must be installed separately (see README); set SADIC_DEPTH_DIR if not in external/depth.
set -euo pipefail
cd "$(dirname "$0")"   # code/
PY=${PYTHON:-python}

# 1. dataset: download the 1000 biological assemblies and normalise them
$PY pipeline/dataset/download_assemblies.py
$PY pipeline/dataset/normalize.py

# 2. SADIC runs (1 worker for the configurations whose timings are reported)
$PY pipeline/run_experiments.py morph_transph_aligned --workers 1
$PY pipeline/run_experiments.py infl_reconsph_aligned --workers 1
$PY pipeline/run_experiments.py morph_reconsph_aligned --workers 4
$PY pipeline/run_experiments.py infl_transph_aligned --workers 4
$PY pipeline/run_experiments.py morph_transph_unaligned --workers 4

# 3. reference methods
$PY pipeline/run_dpx.py --workers 4
$PY pipeline/run_depth.py --workers 4

# 4. sensitivity to the grid resolution (150-protein subset)
$PY pipeline/run_ablation.py --workers 2

# 5. analyses, figures and supplementary material
$PY pipeline/analyze_main.py
$PY pipeline/analyze_tools.py
$PY pipeline/analyze_dpx_depth.py
$PY pipeline/analyze_ablation.py
$PY pipeline/make_supplementary.py
$PY pipeline/make_supplementary_tex.py
# protein renderings (needs requirements-figures.txt)
$PY pipeline/make_protein_figures.py
