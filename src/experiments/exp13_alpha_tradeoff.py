"""Accuracy, redundancy and stability as functions of alpha, on every data set (15 held-out folds, 20 bootstrap
refits per fold and alpha). Rows are comparable with the other experiments: same folds, same bootstrap seeding."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from src.baselines import make_method
from src.datasets import load_all_real
from src.experiments.exp7_extended import abs_corr_matrix
from src.metrics import predictive_metrics, redundancy_metrics, stability_metrics

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
B = 20


def job(name, X, y, seed, fold, tr, te, corr):
    rows = []
    for alpha in ALPHAS:
        kw = dict(max_depth=6, min_samples_leaf=5, random_state=seed)
        model = make_method("ACP-Gini", alpha=alpha, **kw).fit(X[tr], y[tr])
        pm = predictive_metrics(y[te], model.predict(X[te]), model.predict_proba(X[te]))
        rng = np.random.default_rng(seed * 100 + fold)
        trees = [make_method("ACP-Gini", alpha=alpha, **kw).fit(X[b], y[b])
                 for b in (rng.choice(tr, len(tr), replace=True) for _ in range(B))]
        st = stability_metrics(trees)
        red = float(np.mean([redundancy_metrics(t, corr)["weighted_redundancy"] for t in trees]))
        rows.append({"dataset": name, "alpha": alpha, "seed": seed, "fold": fold, "accuracy": pm["accuracy"],
                     "bootstrap_weighted_redundancy": red, "importance_rank_corr": st["importance_rank_corr"],
                     "feature_set_jaccard": st["feature_set_jaccard"], "top5_jaccard": st["top5_jaccard"]})
    return rows


def main():
    jobs = []
    for name, (X, y, _) in load_all_real("Data").items():
        corr = abs_corr_matrix(X)
        for seed in [42, 43, 44]:
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                jobs.append(delayed(job)(name, X, y, seed, fold, tr, te, corr))
    rows = [r for out in Parallel(n_jobs=-1)(jobs) for r in out]
    pd.DataFrame(rows).to_csv("results/alpha_tradeoff.csv", index=False)


if __name__ == "__main__":
    main()
