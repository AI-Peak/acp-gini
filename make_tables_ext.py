"""Tables for the extended evidence. Run after analyze_stats.py."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("paper/generated"); OUT.mkdir(parents=True, exist_ok=True)
folds = pd.read_csv("results/merged_folds.csv")
st = pd.read_csv("results/stats_tests.csv")
order = list(pd.read_csv("results/dataset_stats.csv").sort_values("pair_fraction_abs_r_gt_07", ascending=False).dataset)
METHODS = ["CART", "VIF+CART", "Cluster+CART", "RRF-style", "ACP-Gini", "ACP-Gini (nested)", "RandomForest (reference)"]
SHORT = {"ACP-Gini (nested)": "ACP-Gini (nested)", "RandomForest (reference)": "Random forest (ref.)"}


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, header, rows, align):
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", header + " \\\\", "\\midrule"] + \
            [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ext_methods: mean over the eight data sets of the per-data-set fold means
cols = ["accuracy", "macro_f1", "auc", "bootstrap_weighted_redundancy", "path_corr_boot", "importance_rank_corr", "group_import_corr"]
avg = folds.groupby(["dataset", "method"])[cols].mean().groupby("method").mean().reindex(METHODS)
best = {c: (avg[c].idxmax() if c in ["accuracy", "macro_f1", "auc", "importance_rank_corr", "group_import_corr"] else avg[c].idxmin())
        for c in cols}
rows = []
for m in METHODS:
    cells = []
    for c in cols:
        t = fmt(avg.loc[m, c])
        cells.append(f"\\textbf{{{t}}}" if best[c] == m and not pd.isna(avg.loc[m, c]) and m != "RandomForest (reference)" else t)
    rows.append(f"{SHORT.get(m, m)} & " + " & ".join(cells))
write("ext_methods.tex", "Method & Acc. & F1 & AUC & $R_w$ & Path $|r|$ & Rank corr. & Group corr.", rows, "l" + "c" * 7)

# ext_inference: ACP-Gini minus CART per data set
rows = []
for d in order:
    def row(metric):
        return st[(st.baseline == "CART") & (st.metric == metric) & (st.dataset == d)].iloc[0]
    r, a, k = row("bootstrap_weighted_redundancy"), row("accuracy"), row("importance_rank_corr")
    rows.append(
        f"{d} & {100*r.rel_change:.1f} [{100*r.ci_lo/r.base:.1f}, {100*r.ci_hi/r.base:.1f}] & {pval(r.p_nb_holm)} & "
        f"{100*a['diff']:.2f} [{100*a.ci_lo:.2f}, {100*a.ci_hi:.2f}] & "
        f"{k['diff']:+.3f} [{k.ci_lo:+.3f}, {k.ci_hi:+.3f}] & {pval(k.p_nb_holm)}")
write("ext_inference.tex",
      "Data set & Redundancy change (\\%) [95\\% CI] & Holm $p$ & Accuracy change (pp) [95\\% CI] & Rank-corr. change [95\\% CI] & Holm $p$",
      rows, "lcccc c".replace(" ", ""))

# ext_accuracy: per-data-set accuracy of every method
acc = folds.groupby(["dataset", "method"]).accuracy.mean().unstack().reindex(order)[METHODS]
rows = [f"{d} & " + " & ".join(fmt(acc.loc[d, m]) for m in METHODS) for d in order]
write("ext_accuracy.tex", "Data set & " + " & ".join(SHORT.get(m, m).replace("(reference)", "(ref.)").replace(" (nested $\\alpha$)", " (nested)") for m in METHODS),
      rows, "l" + "c" * len(METHODS))

# ext_redundancy: weighted redundancy per method and data set (bootstrap), with the group-level companions
red = folds.groupby(["dataset", "method"]).bootstrap_weighted_redundancy.mean().unstack().reindex(order)
mrow = ["CART", "VIF+CART", "Cluster+CART", "RRF-style", "ACP-Gini"]
rows = []
for d in order:
    vals = red.loc[d, mrow]
    lo = vals.min()
    rows.append(f"{d} & " + " & ".join((f"\\textbf{{{fmt(v)}}}" if v == lo else fmt(v)) for v in vals))
write("ext_redundancy.tex", "Data set & " + " & ".join(mrow), rows, "l" + "c" * len(mrow))

# ext_path: sign tests and path-level metrics
sg = pd.read_csv("results/sign_tests.csv").query("baseline=='CART'").set_index("metric")
rows = []
for m, label in [("bootstrap_weighted_redundancy", "Weighted redundancy"), ("path_corr_boot", "Mean path $|r|$"),
                 ("path_flagged_boot", "Paths with a pair $|r|>0.7$"), ("importance_rank_corr", "Importance rank correlation"), ("feature_set_jaccard", "Used-feature Jaccard"),
                 ("top5_jaccard", "Top-five Jaccard"), ("group_import_corr", "Group-level importance correlation"),
                 ("group_jaccard", "Group-level Jaccard"), ("accuracy", "Accuracy")]:
    r = sg.loc[m]
    rows.append(f"{label} & {int(r.n_negative)}/{int(r.n_datasets)} & {r.sign_p_two_sided:.3f}")
write("ext_sign.tex", "Metric (ACP-Gini lower than CART) & Data sets & Sign-test $p$", rows, "lcc")
ab = pd.read_csv("results/ablation_full.csv").groupby(["dataset", "config"])[["accuracy", "weighted_redundancy", "path_corr"]].mean()     .groupby("config").mean()
cart = ab.loc["CART"]
LAB = {"CART": "CART (alpha = 0)", "pearson/product/global": "Pearson, product, global (default)", "spearman/product/global": "Spearman, product, global",
       "pearson/min/global": "Pearson, min, global", "pearson/mean/global": "Pearson, mean, global", "pearson/product/node": "Pearson, product, node-local"}
rows = [f"{LAB[c].replace('alpha = 0', chr(36)+chr(92)+'alpha=0'+chr(36))} & {ab.loc[c,'accuracy']:.4f} & {ab.loc[c,'weighted_redundancy']:.3f} & "
        f"{100*(ab.loc[c,'weighted_redundancy']/cart.weighted_redundancy-1):.1f} & {ab.loc[c,'path_corr']:.3f}" for c in LAB]
write("ext_ablation.tex", "Configuration & Accuracy & Weighted red. & Change (\\%) & Path $|r|$", rows, "lcccc")

# ext_noninf: accuracy change vs CART with consensus and fully nested alpha
nest = pd.read_csv("results/stats_nested_vs_cart.csv").query("metric=='accuracy'").set_index("dataset")
cons = st[(st.baseline == "CART") & (st.metric == "accuracy")].set_index("dataset")
rows = []
for d in order:
    c, n = cons.loc[d], nest.loc[d]
    rows.append(f"{d} & {100*c['diff']:+.2f} [{100*c.ci_lo:+.2f}, {100*c.ci_hi:+.2f}] & "
                f"{100*n['diff']:+.2f} [{100*n.ci_lo:+.2f}, {100*n.ci_hi:+.2f}]")
write("ext_noninf.tex", "Data set & Consensus $\\alpha$ & Nested $\\alpha$", rows, "lcc")

# ext_vs_filters: ACP-Gini minus each baseline in bootstrap weighted redundancy (Holm-adjusted corrected tests)
rows = []
for d in order:
    cells = []
    for b in ["VIF+CART", "Cluster+CART", "RRF-style"]:
        r = st[(st.baseline == b) & (st.metric == "bootstrap_weighted_redundancy") & (st.dataset == d)].iloc[0]
        cells.append(f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] ({pval(r.p_nb_holm)})")
    rows.append(f"{d} & " + " & ".join(cells))
write("ext_vs_filters.tex", "Data set & vs VIF+CART & vs Cluster+CART & vs RRF-style", rows, "llll")

# ext_scope: scope and filter-threshold sweeps on single trees (exp10)
rc = pd.read_csv("results/review_checks.csv")
avg = rc.groupby(["dataset", "method"])[["accuracy", "wred_pearson", "wred_spearman", "wred_nmi", "n_features_kept"]].mean() \
        .groupby("method").mean()
LAB = {"CART": "CART", "ACP-Gini": "ACP-Gini (path-local)", "Tree-global corr.": "Correlation penalty, tree-global",
       "VIF@5": "VIF+CART, threshold 5", "VIF@10": "VIF+CART, threshold 10", "VIF@20": "VIF+CART, threshold 20",
       "Cluster@0.5": "Cluster+CART, $|r|\\ge0.5$", "Cluster@0.6": "Cluster+CART, $|r|\\ge0.6$",
       "Cluster@0.7": "Cluster+CART, $|r|\\ge0.7$", "Cluster@0.8": "Cluster+CART, $|r|\\ge0.8$",
       "Cluster@0.9": "Cluster+CART, $|r|\\ge0.9$"}
rows = [f"{LAB[m]} & {avg.loc[m,'accuracy']:.3f} & {avg.loc[m,'wred_pearson']:.3f} & {avg.loc[m,'wred_spearman']:.3f} & "
        f"{avg.loc[m,'wred_nmi']:.3f} & {avg.loc[m,'n_features_kept']:.1f}" for m in LAB]
write("ext_scope.tex", "Configuration & Acc. & $R_w$ Pearson & $R_w$ Spearman & $R_w$ NMI & Features kept", rows, "lccccc")

print("tables written")
