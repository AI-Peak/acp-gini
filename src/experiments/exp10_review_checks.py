"""Checks prompted by the mock review.

(a) Scope ablation: a correlation-aware but tree-global penalty (candidate penalized against every
    feature used anywhere so far in the tree) against the path-local ACP-Gini penalty.
(b) Metrics not derived from the Pearson matrix used by the penalty: weighted redundancy recomputed
    with Spearman |rho| and with normalized mutual information on quantile bins.
(c) Threshold sweeps for the two filter baselines so that they are not judged at one fixed setting.
All rows use the same folds, alphas and trees as exp3_uci.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold

from src.baselines import ClusterCARTClassifier, VIFCARTClassifier
from src.datasets import load_all_real
from src.experiments.exp7_extended import abs_corr_matrix
from src.metrics import predictive_metrics, redundancy_metrics
from src.tree import ACPGiniTreeClassifier


class TreeGlobalCorr(ACPGiniTreeClassifier):
    """Correlation-aware penalty over all features used so far in the tree, not only the path."""

    def _penalty(self, feature, ancestors, corr):
        others = [p for p in self._used_global if p != feature]
        if not others or self.alpha == 0:
            return 1.0
        return float(np.prod(np.clip(1.0 - self.alpha * corr[feature, others], 0.0, 1.0)))


def spearman_matrix(X):
    return abs_corr_matrix(np.apply_along_axis(rankdata, 0, X))


def nmi_matrix(X, bins=10):
    codes = np.column_stack([np.digitize(X[:, j], np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1])))
                             for j in range(X.shape[1])])
    d = X.shape[1]; m = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            m[i, j] = m[j, i] = normalized_mutual_info_score(codes[:, i], codes[:, j])
    return m


def job(name, X, y, alpha, seed, fold, tr, te, mats):
    kw = dict(max_depth=6, min_samples_leaf=5, random_state=seed)
    models = {
        "CART": ACPGiniTreeClassifier(alpha=0, **kw), "ACP-Gini": ACPGiniTreeClassifier(alpha=alpha, **kw),
        "Tree-global corr.": TreeGlobalCorr(alpha=alpha, **kw),
    }
    for t in [5, 10, 20]:
        models[f"VIF@{t}"] = VIFCARTClassifier(threshold=float(t), **kw)
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        models[f"Cluster@{t}"] = ClusterCARTClassifier(threshold=t, **kw)
    rows = []
    for label, model in models.items():
        model.fit(X[tr], y[tr])
        pm = predictive_metrics(y[te], model.predict(X[te]), model.predict_proba(X[te]))
        row = {"dataset": name, "method": label, "seed": seed, "fold": fold, **pm}
        for k, m in mats.items():
            row[f"wred_{k}"] = redundancy_metrics(model, m)["weighted_redundancy"]
        row["n_features_kept"] = len(getattr(model, "kept_features_", range(X.shape[1])))
        rows.append(row)
    return rows


def main():
    main_df = pd.read_csv("results/uci_main.csv"); jobs = []
    for name, (X, y, _) in load_all_real("Data").items():
        alpha = float(main_df[(main_df.dataset == name) & (main_df.method == "ACP-Gini")].selected_alpha.iloc[0])
        mats = {"pearson": abs_corr_matrix(X), "spearman": spearman_matrix(X), "nmi": nmi_matrix(X)}
        for seed in [42, 43, 44]:
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                jobs.append(delayed(job)(name, X, y, alpha, seed, fold, tr, te, mats))
    rows = [r for out in Parallel(n_jobs=-1)(jobs) for r in out]
    pd.DataFrame(rows).to_csv("results/review_checks.csv", index=False)


if __name__ == "__main__":
    main()
