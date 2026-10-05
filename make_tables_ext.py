"""Tables for the extended evidence. Run after analyze_stats.py.
In every comparison table, results where the proposed method is better are set in bold."""
from pathlib import Path
import pandas as pd

OUT = Path("paper/generated"); OUT.mkdir(parents=True, exist_ok=True)
folds = pd.read_csv("results/merged_folds.csv")
st = pd.read_csv("results/stats_tests.csv")
order = list(pd.read_csv("results/dataset_stats.csv").sort_values("pair_fraction_abs_r_gt_07", ascending=False).dataset)
BASELINES = ["CART", "VIF+CART", "Cluster+CART", "RRF-style"]
PROPOSED = ["ACP-Gini", "ACP-Gini (nested)"]


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def bf(text, on):
    return f"\\textbf{{{text}}}" if on else text


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, header, rows, align):
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", header + " \\\\", "\\midrule"] + \
            [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ext_methods: mean over the eight data sets of the per-data-set fold means.
# A proposed-method value is bold when it is better than every baseline.
cols = ["accuracy", "macro_f1", "auc", "bootstrap_weighted_redundancy", "path_corr_boot", "importance_rank_corr", "group_import_corr"]
higher = {"accuracy", "macro_f1", "auc", "importance_rank_corr", "group_import_corr"}
avg = folds.groupby(["dataset", "method"])[cols].mean().groupby("method").mean()
rows = []
for m in BASELINES + PROPOSED:
    cells = []
    for c in cols:
        v = avg.loc[m, c]
        best_base = avg.loc[BASELINES, c].max() if c in higher else avg.loc[BASELINES, c].min()
        better = (round(v, 3) > round(best_base, 3)) if c in higher else (round(v, 3) < round(best_base, 3))
        cells.append(bf(fmt(v), m in PROPOSED and bool(better)))
    rows.append(f"{m} & " + " & ".join(cells))
write("ext_methods.tex", "Method & Acc. & F1 & AUC & $R_w$ & Path $|r|$ & Rank corr. & Group corr.", rows, "l" + "c" * 7)

# ext_inference: ACP-Gini minus CART per data set; significant favourable results in bold
rows = []
for d in order:
    def row(metric):
        return st[(st.baseline == "CART") & (st.metric == metric) & (st.dataset == d)].iloc[0]
    r, a, k = row("bootstrap_weighted_redundancy"), row("accuracy"), row("importance_rank_corr")
    r_good = r.p_nb_holm < 0.05 and r["diff"] < 0
    k_good = k.p_nb_holm < 0.05 and k["diff"] > 0
    r_txt = "%.1f [%.1f, %.1f]" % (100 * r.rel_change, 100 * r.ci_lo / r.base, 100 * r.ci_hi / r.base)
    a_txt = "%.2f [%.2f, %.2f]" % (100 * a["diff"], 100 * a.ci_lo, 100 * a.ci_hi)
    k_txt = "%+.3f [%+.3f, %+.3f]" % (k["diff"], k.ci_lo, k.ci_hi)
    rows.append(f"{d} & {bf(r_txt, r_good)} & {bf(pval(r.p_nb_holm), r_good)} & {a_txt} & "
                f"{bf(k_txt, k_good)} & {bf(pval(k.p_nb_holm), k_good)}")
write("ext_inference.tex",
      "Data set & Redundancy change (\\%) [95\\% CI] & Holm $p$ & Accuracy change (pp) [95\\% CI] & Rank-corr. change [95\\% CI] & Holm $p$",
      rows, "lccccc")

# ext_redundancy: bootstrap importance-weighted redundancy per method; bold when ACP-Gini is the lowest
red = folds.groupby(["dataset", "method"]).bootstrap_weighted_redundancy.mean().unstack().reindex(order)
mrow = BASELINES + ["ACP-Gini"]
rows = []
for d in order:
    vals = red.loc[d, mrow]
    acp_best = round(vals["ACP-Gini"], 3) <= round(vals.min(), 3)
    rows.append(f"{d} & " + " & ".join(bf(fmt(vals[m]), m == "ACP-Gini" and acp_best) for m in mrow))
write("ext_redundancy.tex", "Data set & " + " & ".join(mrow), rows, "l" + "c" * len(mrow))

# ext_noninf: accuracy change vs CART with consensus and fully nested alpha (no bold: no significant difference)
nest = pd.read_csv("results/stats_nested_vs_cart.csv").query("metric=='accuracy'").set_index("dataset")
cons = st[(st.baseline == "CART") & (st.metric == "accuracy")].set_index("dataset")
rows = []
for d in order:
    c, n = cons.loc[d], nest.loc[d]
    rows.append(f"{d} & {100*c['diff']:+.2f} [{100*c.ci_lo:+.2f}, {100*c.ci_hi:+.2f}] & "
                f"{100*n['diff']:+.2f} [{100*n.ci_lo:+.2f}, {100*n.ci_hi:+.2f}]")
write("ext_noninf.tex", "Data set & Consensus $\\alpha$ & Nested $\\alpha$", rows, "lcc")

# ext_vs_filters: ACP-Gini minus each baseline in bootstrap R_w; bold when ACP-Gini is significantly lower
rows = []
for d in order:
    cells = []
    for b in ["VIF+CART", "Cluster+CART", "RRF-style"]:
        r = st[(st.baseline == b) & (st.metric == "bootstrap_weighted_redundancy") & (st.dataset == d)].iloc[0]
        good = r.p_nb_holm < 0.05 and r["diff"] < 0
        txt = "%+.3f [%+.3f, %+.3f] (%s)" % (r["diff"], r.ci_lo, r.ci_hi, pval(r.p_nb_holm))
        cells.append(bf(txt, good))
    rows.append(f"{d} & " + " & ".join(cells))
write("ext_vs_filters.tex", "Data set & vs VIF+CART & vs Cluster+CART & vs RRF-style", rows, "llll")

print("tables written")
