"""Ablation of correlation type, aggregation and scope on all data sets, scored on
redundancy as well as accuracy (15 held-out folds, selected alpha per data set)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from src.datasets import load_all_real
from src.experiments.exp7_extended import abs_corr_matrix, path_metrics
from src.metrics import predictive_metrics, redundancy_metrics
from src.tree import ACPGiniTreeClassifier

CONFIGS = [("pearson", "product", "global"), ("spearman", "product", "global"), ("pearson", "min", "global"),
           ("pearson", "mean", "global"), ("pearson", "product", "node")]


def job(name, X, y, alpha, seed, fold, tr, te, corr):
    rows = []
    for cm, agg, scope in [("pearson", "product", "global")] + CONFIGS:
        a = 0.0 if len(rows) == 0 else alpha  # first row is the CART reference
        m = ACPGiniTreeClassifier(alpha=a, max_depth=6, min_samples_leaf=5, corr_method=cm, penalty_agg=agg,
                                  corr_scope=scope, random_state=seed).fit(X[tr], y[tr])
        pm = predictive_metrics(y[te], m.predict(X[te]), m.predict_proba(X[te]))
        rows.append({"dataset": name, "config": "CART" if a == 0 else f"{cm}/{agg}/{scope}", "seed": seed, "fold": fold,
                     **pm, "weighted_redundancy": redundancy_metrics(m, corr)["weighted_redundancy"],
                     "path_corr": path_metrics(m, corr)[1]})
    return rows


def main():
    main_df = pd.read_csv("results/uci_main.csv"); jobs = []
    for name, (X, y, _) in load_all_real("Data").items():
        alpha = float(main_df[(main_df.dataset == name) & (main_df.method == "ACP-Gini")].selected_alpha.iloc[0])
        corr = abs_corr_matrix(X)
        for seed in [42, 43, 44]:
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                jobs.append(delayed(job)(name, X, y, alpha, seed, fold, tr, te, corr))
    rows = [r for out in Parallel(n_jobs=-1)(jobs) for r in out]
    pd.DataFrame(rows).to_csv("results/ablation_full.csv", index=False)

if __name__ == "__main__": main()
