from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path("paper/generated"); OUT.mkdir(parents=True, exist_ok=True)

HEADER_MAP = {"dataset": "Data set", "selected_alpha": "$\\alpha$", "selected_rrf_lambda": "$\\lambda$",
              "n": "Samples", "d": "Features", "positive_rate": "Positive rate",
              "pair_fraction_abs_r_gt_07": "Pairs $|r|>0.7$"}


def latex(frame, path, float_format=lambda x: f"{x:.3f}"):
    path.write_text(frame.rename(columns=HEADER_MAP).to_latex(index=False, float_format=float_format, escape=False),
                    encoding="utf-8")


main = pd.read_csv("results/uci_main.csv")
params = main.groupby("dataset")[["selected_alpha", "selected_rrf_lambda"]].first().reset_index()
stats = pd.read_csv("results/dataset_stats.csv").merge(params, on="dataset")
latex(stats, OUT / "table1_datasets.tex")

# Case study: the candidate that ACP-Gini selects (ACP rank 1) is set in bold.
trace = pd.read_csv("results/wdbc_split_trace.csv")
bold = trace["ACP Rank"] == 1
trace = trace.astype(object)
for col in trace.columns:
    trace[col] = [f"\\textbf{{{v:.4f}}}" if b and isinstance(v, float) else (f"\\textbf{{{v}}}" if b else (f"{v:.4f}" if isinstance(v, float) else v))
                  for v, b in zip(trace[col], bold)]
(OUT / "table8_case_study.tex").write_text(trace.to_latex(index=False, escape=False), encoding="utf-8")

# Secondary check: paired Wilcoxon tests (the primary tests are in src/experiments/analyze_stats.py).
METRICS = ["accuracy", "macro_f1", "auc", "feature_set_jaccard", "top5_jaccard", "importance_rank_corr", "structural_distance",
           "main_set_redundancy", "main_weighted_redundancy", "bootstrap_set_redundancy", "bootstrap_weighted_redundancy"]
tests = []
for dataset in main.dataset.unique():
    block = main[main.dataset == dataset]
    acp = block[block.method == "ACP-Gini"].sort_values(["seed", "fold"])
    for baseline in ["CART", "VIF+CART", "RRF-style"]:
        other = block[block.method == baseline].sort_values(["seed", "fold"])
        for metric in METRICS:
            diff = acp[metric].to_numpy() - other[metric].to_numpy()
            try:
                p = float(wilcoxon(diff).pvalue)
            except ValueError:
                p = 1.0
            tests.append({"dataset": dataset, "baseline": baseline, "metric": metric, "mean_difference": float(diff.mean()), "p_value": p})
pd.DataFrame(tests).to_csv("results/wilcoxon_tests.csv", index=False)
