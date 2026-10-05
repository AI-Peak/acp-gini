#!/usr/bin/env bash
# Re-run every analysis on all 12 data sets after the four strongly collinear sets have been evaluated.
# Usage: bash run_all_extended.sh   (from the repository root)
set -euo pipefail
export PYTHONPATH=.
PY="python -W ignore"

for d in Parkinsons Musk1 Landsat Ozone; do
  until [ -f "results/_part_${d}.csv" ]; do sleep 20; done
done
$PY -m src.experiments.merge_parts
$PY -m src.experiments.exp7_extended
$PY -m src.experiments.analyze_stats
$PY -m src.experiments.exp8_depth_mechanism
$PY -m src.experiments.exp9_ablation_full
$PY -m src.experiments.exp10_review_checks
$PY -m src.experiments.exp11_path_pairs
$PY -m src.experiments.exp13_alpha_tradeoff
$PY -m src.experiments.exp6_real_sweep
$PY -m src.experiments.exp5_runtime
$PY make_tables.py
$PY make_tables_ext.py
$PY make_figures.py
$PY make_figures_ext.py
echo ALL_DONE
