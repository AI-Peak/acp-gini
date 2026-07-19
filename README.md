# ACP-Gini Mini-Research

Reproducible implementation of an ancestor-correlation penalty for single decision trees.

## Setup

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH='.'
```

Global seeds are 42, 43, and 44. Smoke runs use seed 42, two folds, and 10 bootstraps. Full runs use five folds, three seeds, and 50 bootstraps.

## Commands

```powershell
python -m src.experiments.exp1_sanity
$env:ACP_SMOKE='1'; python -m src.experiments.exp2_synthetic
$env:ACP_SMOKE='1'; python -m src.experiments.exp3_uci
$env:ACP_SMOKE='1'; python -m src.experiments.exp4_ablation
$env:ACP_SMOKE='1'; python -m src.experiments.exp5_runtime
Remove-Item Env:ACP_SMOKE
python -m src.experiments.exp2_synthetic
python -m src.experiments.exp3_uci
python -m src.experiments.exp4_ablation
python -m src.experiments.exp5_runtime
python make_tables.py
python make_figures.py
```

Expected full runtime is hardware-dependent and dominated by bootstrap refits; budget up to two CPU hours. All reported values originate in `results/*.csv`.
