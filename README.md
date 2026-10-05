# ACP-Gini Mini-Research

Reproducible implementation of an ancestor-correlation penalty for single decision trees.

## Setup

Reference environment: Python 3.10.11 on 64-bit Microsoft Windows 11 Pro,
AMD Ryzen 7 PRO 5850U CPU. The exact package versions are pinned in
`requirements.txt`.

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
$env:ACP_SMOKE='1'; python -m src.experiments.exp5_runtime
Remove-Item Env:ACP_SMOKE
python -m src.experiments.exp2_synthetic
python -m src.experiments.exp3_uci
python -m src.experiments.exp5_runtime
python -m src.experiments.derive_group_metrics
python -m src.experiments.exp6_real_sweep
# Extended evidence (needs results/uci_main.csv; joblib-parallel, roughly 20 min each on 22 cores)
python -m src.experiments.exp7_extended        # path/group metrics, nested alpha, Cluster+CART
python -m src.experiments.exp8_depth_mechanism # depth sensitivity, correlation-band mass, top path pairs
python -m src.experiments.exp9_ablation_full   # estimator/aggregation/scope ablation on all data sets
python -m src.experiments.exp10_review_checks  # tree-global scope, Spearman/NMI redundancy, filter threshold sweeps
python -m src.experiments.exp11_path_pairs     # pair-level provenance for the path correlation bands
python -m src.experiments.exp12_scenarios      # controlled scenarios: redundant proxies (rho sweep) and conditional structure
python -m src.experiments.exp13_alpha_tradeoff # accuracy, redundancy and stability as functions of alpha
python -m src.experiments.analyze_stats        # Nadeau-Bengio tests, Holm, sign tests, nested-vs-CART
python make_tables.py
python make_tables_ext.py
python make_figures.py
python make_figures_ext.py
```

Data sets: WDBC, Wine Quality, Ionosphere, Sonar and Pima, three scikit-learn bundled sets (Diabetes, Cultivar, Digits38) that need no download, and four strongly collinear UCI sets (Parkinsons, Musk v1, Statlog Landsat, Ozone eight-hour task) that `src/datasets.py::load_strong` downloads once from the UCI repository and caches as CSV in `Data/`. `exp3_uci` is slow on the larger sets; to run data sets in parallel use `ACP_DATASETS=<name> ACP_PART=<name> python -m src.experiments.exp3_uci` once per set, then `python -m src.experiments.merge_parts`. `bash run_all_extended.sh` re-runs every analysis for all 12 data sets after the parts exist. `exp3_uci` only re-runs the data sets named in `ACP_DATASETS` (for example `ACP_DATASETS=Diabetes,Cultivar,Digits38`) and merges them into `results/uci_main.csv`. `analyze_stats` also checks that `exp7_extended` reproduces `uci_main.csv` exactly for the four base methods.

The complete full experiment suite took approximately **35 wall-clock minutes** on the reference machine. Runtime is dominated by the inner-CV and bootstrap refits in `exp3_uci`; budget up to two hours on a slower CPU. All reported values originate in `results/*.csv`.

## Paper

`paper/ACP-Gini_Report_Springer.tex` is self-contained with `sn-jnl.cls`, `sn-basic.bst`, `references.bib`, `paper/generated/` tables and `paper/figures/`. Compile with `latexmk -pdf ACP-Gini_Report_Springer.tex` inside `paper/` (or upload the `paper/` folder to Overleaf with pdfLaTeX).
