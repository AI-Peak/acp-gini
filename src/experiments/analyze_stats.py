"""Paired inference for every headline claim.

Three repeats of five-fold CV give 15 overlapping resamples, so the plain paired t-test and
Wilcoxon test are anti-conservative. The primary p-values and CIs use the Nadeau-Bengio (2003)
corrected resampled t-test (variance factor 1/J + n_test/n_train). Wilcoxon p-values are kept as
a secondary check. Holm adjustment is applied across datasets within each comparison and metric.
"""
import numpy as np
import pandas as pd
from scipy import stats

J, RATIO = 15, 0.25  # 15 resamples; 5-fold CV has n_test/n_train = 1/4
UCI = ["accuracy", "macro_f1", "auc", "bootstrap_weighted_redundancy", "bootstrap_set_redundancy",
       "top5_jaccard", "importance_rank_corr", "feature_set_jaccard", "structural_distance"]
EXT = ["path_flagged_boot", "path_corr_boot", "group_jaccard", "group_import_corr"]


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def nb_test(d):
    d = np.asarray(d, float); mean = d.mean(); var = d.var(ddof=1)
    if var == 0:
        return mean, 0.0, 1.0, (mean, mean), 0.0
    se = np.sqrt((1 / J + RATIO) * var); t = mean / se
    p = 2 * stats.t.sf(abs(t), J - 1); q = stats.t.ppf(0.975, J - 1)
    return mean, se, p, (mean - q * se, mean + q * se), mean / np.sqrt(var)


def main():
    main_df = pd.read_csv("results/uci_main.csv"); ext = pd.read_csv("results/extended.csv")
    # Base methods come from uci_main.csv; the extended run supplies path-level and group-level
    # metrics for them, plus every metric for the three methods that exp3 did not evaluate.
    keys = ["dataset", "method", "seed", "fold"]
    base = main_df[keys + UCI]
    on_base = ext[ext.method.isin(["CART", "VIF+CART", "RRF-style", "ACP-Gini"])][keys + EXT]
    both = base.merge(on_base, on=keys, how="left")
    extra = ext[ext.method.isin(["ACP-Gini (nested)", "Cluster+CART"])]
    both = pd.concat([both, extra[keys + [c for c in UCI + EXT if c in extra.columns]]], ignore_index=True)
    # Reproducibility check: exp7 refits the four base methods and must match exp3 exactly.
    chk = ext[ext.method.isin(["CART", "VIF+CART", "RRF-style", "ACP-Gini"])].merge(base, on=keys, suffixes=("_x", ""))
    for c in ["accuracy", "bootstrap_weighted_redundancy", "importance_rank_corr", "top5_jaccard"]:
        if c + "_x" in chk:
            print("repro max |diff|", c, float((chk[c + "_x"] - chk[c]).abs().max()))
    both.to_csv("results/merged_folds.csv", index=False)

    rows = []
    for other in ["CART", "VIF+CART", "RRF-style", "Cluster+CART"]:
        for metric in UCI + EXT:
            for ds in both.dataset.unique():
                a = both[(both.dataset == ds) & (both.method == "ACP-Gini")].sort_values(["seed", "fold"])
                b = both[(both.dataset == ds) & (both.method == other)].sort_values(["seed", "fold"])
                if a[metric].isna().any() or b[metric].isna().any() or len(a) != 15 or len(b) != 15:
                    continue
                d = a[metric].to_numpy() - b[metric].to_numpy()
                mean, se, p, ci, dz = nb_test(d)
                try:
                    pw = float(stats.wilcoxon(d).pvalue)
                except ValueError:
                    pw = 1.0
                rows.append({"baseline": other, "metric": metric, "dataset": ds, "acp": a[metric].mean(),
                             "base": b[metric].mean(), "diff": mean, "ci_lo": ci[0], "ci_hi": ci[1],
                             "rel_change": mean / b[metric].mean() if b[metric].mean() else np.nan,
                             "p_nb": p, "p_wilcoxon": pw, "dz": dz})
    out = pd.DataFrame(rows)
    out["p_nb_holm"] = out.groupby(["baseline", "metric"]).p_nb.transform(lambda s: holm(s.values))
    out["p_wilcoxon_holm"] = out.groupby(["baseline", "metric"]).p_wilcoxon.transform(lambda s: holm(s.values))
    out.to_csv("results/stats_tests.csv", index=False)

    # Dataset-level sign test across datasets (unit = dataset mean difference).
    sign = []
    for (b, m), g in out.groupby(["baseline", "metric"]):
        k, n = int((g["diff"] < 0).sum()), len(g)
        sign.append({"baseline": b, "metric": m, "n_datasets": n, "n_negative": k,
                     "sign_p_two_sided": float(stats.binomtest(k, n, 0.5).pvalue)})
    pd.DataFrame(sign).to_csv("results/sign_tests.csv", index=False)

    # Fully nested ACP-Gini (alpha tuned per fold on training data only) against CART.
    nest = []
    for metric in ["accuracy", "macro_f1", "auc", "bootstrap_weighted_redundancy", "path_corr_boot", "importance_rank_corr"]:
        for ds in both.dataset.unique():
            a = both[(both.dataset == ds) & (both.method == "ACP-Gini (nested)")].sort_values(["seed", "fold"])
            b = both[(both.dataset == ds) & (both.method == "CART")].sort_values(["seed", "fold"])
            if a[metric].isna().any() or b[metric].isna().any():
                continue
            mean, se, p, ci, dz = nb_test(a[metric].to_numpy() - b[metric].to_numpy())
            nest.append({"metric": metric, "dataset": ds, "acp_nested": a[metric].mean(), "cart": b[metric].mean(),
                         "diff": mean, "ci_lo": ci[0], "ci_hi": ci[1], "p_nb": p, "dz": dz})
    nest = pd.DataFrame(nest); nest["p_nb_holm"] = nest.groupby("metric").p_nb.transform(lambda v: holm(v.values))
    nest.to_csv("results/stats_nested_vs_cart.csv", index=False)

    # Non-inferiority: lower end of the 95% NB CI for the accuracy difference, ACP minus CART.
    ni = out[(out.baseline == "CART") & (out.metric == "accuracy")][["dataset", "diff", "ci_lo", "ci_hi", "p_nb"]]
    print(ni.round(4).to_string())
    print(pd.DataFrame(sign).query("baseline=='CART'").round(4).to_string())


if __name__ == "__main__":
    main()
