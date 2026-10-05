"""Extended evidence: path-level redundancy, group-level stability, a truly nested
ACP-Gini (alpha chosen per fold on training data only) and a Cluster+CART baseline. Splits, bootstrap seeds and selected alphas replicate
exp3_uci exactly so every row is paired with results/uci_main.csv.
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from src.baselines import make_method
from src.datasets import load_all_real
from src.metrics import predictive_metrics, redundancy_metrics, stability_metrics

THRESH = 0.7


def abs_corr_matrix(X):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)), nan=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def feature_groups(corr):
    """Complete-linkage groups: every within-group pair has |r| >= THRESH."""
    if corr.shape[0] < 2:
        return np.zeros(corr.shape[0], int)
    return fcluster(linkage(squareform(1 - corr, checks=False), "complete"), t=1 - THRESH, criterion="distance") - 1


def path_metrics(model, corr):
    """Sample-weighted share of leaves whose path holds a pair with |r|>THRESH, and the
    sample-weighted mean pairwise |r| among distinct features on a path."""
    flagged = mean_corr = total = 0.0
    for path, n in model.leaf_paths():
        feats = sorted(set(path)); total += n
        if len(feats) < 2:
            continue
        vals = [corr[i, j] for a, i in enumerate(feats) for j in feats[a + 1:]]
        flagged += n * (max(vals) > THRESH); mean_corr += n * float(np.mean(vals))
    return (flagged / total, mean_corr / total) if total else (0.0, 0.0)


def group_stability(trees, groups):
    """Jaccard of used groups and Spearman correlation of group-aggregated importance."""
    k = int(groups.max()) + 1
    sets, vecs = [], []
    for t in trees:
        used = {int(groups[f]) for f in t.used_features}; sets.append(used)
        v = np.zeros(k); np.add.at(v, groups, t.feature_importances_); vecs.append(v)
    jac, rho = [], []
    for a in range(len(trees)):
        for b in range(a + 1, len(trees)):
            jac.append(len(sets[a] & sets[b]) / len(sets[a] | sets[b]) if sets[a] | sets[b] else 1.0)
            r = spearmanr(vecs[a], vecs[b]).statistic if k > 2 else np.nan
            rho.append(0.0 if np.isnan(r) else r)
    return float(np.mean(jac)), float(np.mean(rho))


def run_fold(name, X, y, main, seed, fold, train, test, B, corr, groups):
    sel = main[(main.dataset == name) & (main.seed == seed) & (main.fold == fold) & (main.method == "ACP-Gini")].iloc[0]
    alpha, fold_alpha = float(sel.selected_alpha), float(sel.fold_tuned_alpha)
    kwargs = dict(max_depth=6, min_samples_leaf=5, random_state=seed)
    specs = [("CART", {}), ("VIF+CART", {}), ("RRF-style", {"rrf_lambda": float(sel.selected_rrf_lambda)}),
             ("ACP-Gini", {"alpha": alpha}), ("ACP-Gini (nested)", {"alpha": fold_alpha}), ("Cluster+CART", {})]
    rows = []
    for label, extra in specs:
        method = "ACP-Gini" if label.startswith("ACP-Gini") else label
        model = make_method(method, **extra, **kwargs).fit(X[train], y[train])
        pm = predictive_metrics(y[test], model.predict(X[test]), model.predict_proba(X[test]))
        row = {"dataset": name, "method": label, "seed": seed, "fold": fold, "alpha_used": extra.get("alpha", np.nan),
               **pm, "n_nodes": model.n_nodes, "depth": model.depth}
        row["path_flagged_main"], row["path_corr_main"] = path_metrics(model, corr)
        rng = np.random.default_rng(seed * 100 + fold); trees = []
        for _ in range(B):
            boot = rng.choice(train, len(train), replace=True)
            trees.append(make_method(method, **extra, **kwargs).fit(X[boot], y[boot]))
        pf, pc = zip(*[path_metrics(t, corr) for t in trees])
        row["path_flagged_boot"], row["path_corr_boot"] = float(np.mean(pf)), float(np.mean(pc))
        row["group_jaccard"], row["group_import_corr"] = group_stability(trees, groups)
        row.update({f"main_{k}": v for k, v in redundancy_metrics(model, corr).items()})
        boot_red = [redundancy_metrics(t, corr) for t in trees]
        row["bootstrap_set_redundancy"] = float(np.mean([r["set_redundancy"] for r in boot_red]))
        row["bootstrap_weighted_redundancy"] = float(np.mean([r["weighted_redundancy"] for r in boot_red]))
        row.update(stability_metrics(trees))
        rows.append(row)
    return rows


def main():
    main_df = pd.read_csv("results/uci_main.csv"); frames = []
    B = int(os.getenv("ACP_B", "50"))
    for name, (X, y, _) in load_all_real("Data").items():
        corr = abs_corr_matrix(X); groups = feature_groups(corr)
        jobs = []
        for seed in [42, 43, 44]:
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                jobs.append(delayed(run_fold)(name, X, y, main_df, seed, fold, tr, te, B, corr, groups))
        out = Parallel(n_jobs=-1)(jobs); frames.append(pd.DataFrame([r for rows in out for r in rows]))
        print(name, "done; groups:", int(groups.max()) + 1, "of", X.shape[1], flush=True)
    pd.concat(frames, ignore_index=True).to_csv("results/extended.csv", index=False)


if __name__ == "__main__":
    main()
