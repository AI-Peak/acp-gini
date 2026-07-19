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
$env:ACP_SMOKE='1'; python -m src.experiments.exp4_ablation
$env:ACP_SMOKE='1'; python -m src.experiments.exp5_runtime
Remove-Item Env:ACP_SMOKE
python -m src.experiments.exp2_synthetic
python -m src.experiments.exp3_uci
python -m src.experiments.exp4_ablation
python -m src.experiments.exp5_runtime
python -m src.experiments.derive_group_metrics
python -m src.experiments.exp6_real_sweep
python make_tables.py
python make_figures.py
```

The complete full experiment suite took approximately **35 wall-clock minutes** on the reference machine. Runtime is dominated by the inner-CV and bootstrap refits in `exp3_uci`; budget up to two hours on a slower CPU. All reported values originate in `results/*.csv`.

## Paper and slides

- Upload `paper/ACP-Gini_Overleaf_Source.zip` as a new Overleaf project. The archive is self-contained with `sn-jnl.cls`, `sn-basic.bst`, references, generated tables, and figures; set `ACP-Gini_Report_Springer.tex` as the main file and compile with pdfLaTeX.
- Run `python paper/build_docx.py` to regenerate the Word report, then use LibreOffice for PDF preview/render QA.
- The PowerPoint deck is `slides/ACP-Gini_Presentation.pptx`; its source is `slides/build_deck.mjs`.
