"""Controlled scenarios (exp12).

S1  Redundant proxies: the group generator over rho in {0, 0.3, ..., 0.99, 1.0} (rho = 1 gives exact duplicates).
S2  Conditional structure: a gate chooses between two regimes. In the left regime the label depends on a latent z that
    feature a reproduces exactly (b is a noisy copy). In the right regime a is noise and b carries the label together
    with a weaker feature d. Globally a and b are strongly correlated, but on the right path a is never used, so only a
    tree-global penalty (which remembers a from the left subtree) penalizes b there. Filters that keep one of a and b
    lose the right regime.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from src.baselines import ClusterCARTClassifier, VIFCARTClassifier
from src.datasets import make_correlated
from src.experiments.exp10_review_checks import TreeGlobalCorr
from src.metrics import predictive_metrics
from src.tree import ACPGiniTreeClassifier

KW = dict(max_depth=6, min_samples_leaf=5)


def make_conditional(n=3000, p_left=0.7, seed=0):
    """Gate g selects the regime. Left: y = [z + noise > -1]; right: y = [-(s + d/2) + noise > 1], so each regime has a
    strongly skewed class balance (the gate has gain and is chosen at the root) and b predicts with opposite signs in
    the two regimes (it is useless before the gate). a copies z on the left and is noise on the right; b is a noisy
    copy of z on the left and tracks s on the right, so corr(a, b) is high globally and low inside the right regime."""
    rng = np.random.default_rng(seed)
    right = rng.random(n) > p_left
    g = np.where(right, 1.0, -1.0) + 0.3 * rng.normal(size=n)
    z, s, d = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    a = np.where(right, rng.normal(size=n), z)
    b = np.where(right, s + 0.1 * rng.normal(size=n), z + 0.3 * rng.normal(size=n))
    y_left = z + 0.3 * rng.normal(size=n) > -1.0
    y_right = -(s + 0.5 * d) + 0.3 * rng.normal(size=n) > 1.0
    y = np.where(right, y_right, y_left).astype(int)
    noise = rng.normal(size=(n, 5))
    return np.column_stack([g, a, b, d, noise]), y, right


def group_metrics(importance, n_signal=5, n_copies=3):
    group = np.r_[np.arange(n_signal), np.repeat(np.arange(n_signal), n_copies)]
    imp = importance[: len(group)]
    top = np.argsort(importance)[::-1][:5]
    covered = len({int(group[j]) for j in top if j < len(group)})
    conc = np.mean([imp[group == g].max() / imp[group == g].sum() if imp[group == g].sum() > 0 else 0 for g in range(n_signal)])
    return covered / n_signal, float(conc), float(importance[len(group):].sum())


def methods(alpha_list=(0.5, 1.0)):
    m = {"CART": lambda: ACPGiniTreeClassifier(alpha=0, **KW)}
    for a in alpha_list:
        m[f"ACP-Gini a={a}"] = (lambda a=a: ACPGiniTreeClassifier(alpha=a, **KW))
        m[f"Tree-global a={a}"] = (lambda a=a: TreeGlobalCorr(alpha=a, **KW))
    m["ACP-Gini node a=1.0"] = lambda: ACPGiniTreeClassifier(alpha=1.0, corr_scope="node", **KW)
    m["VIF+CART"] = lambda: VIFCARTClassifier(**KW)
    m["Cluster+CART"] = lambda: ClusterCARTClassifier(**KW)
    return m


def run_s1(rho, rep):
    X, y, _, _ = make_correlated(rho=rho, task_seed=100 + rep)
    tr, te = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=100 + rep)
    rows = []
    for name, ctor in methods().items():
        model = ctor().fit(X[tr], y[tr])
        pm = predictive_metrics(y[te], model.predict(X[te]), model.predict_proba(X[te]))
        imp = getattr(model, "feature_importances_", None)
        cov, conc, noise = group_metrics(np.asarray(imp))
        rows.append({"scenario": "redundant", "rho": rho, "rep": rep, "method": name, **pm,
                     "group_coverage": cov, "group_concentration": conc, "noise_importance": noise})
    return rows


def run_s2(p_left, rep):
    X, y, right = make_conditional(p_left=p_left, seed=200 + rep)
    tr, te = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=200 + rep)
    corr_ab = float(np.corrcoef(X[tr, 1], X[tr, 2])[0, 1])
    rows = []
    for name, ctor in methods().items():
        model = ctor().fit(X[tr], y[tr])
        pred = model.predict(X[te]); pm = predictive_metrics(y[te], pred, model.predict_proba(X[te]))
        r = right[te]
        rows.append({"scenario": "conditional", "p_left": p_left, "rep": rep, "method": name, **pm,
                     "acc_left": float((pred == y[te])[~r].mean()), "acc_right": float((pred == y[te])[r].mean()),
                     "root_is_gate": int(model.root_feature == 0), "corr_ab": corr_ab})
    return rows


def main():
    reps = 30
    jobs = [delayed(run_s1)(rho, r) for rho in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0] for r in range(reps)]
    jobs += [delayed(run_s2)(pl, r) for pl in [0.5, 0.6, 0.7, 0.8] for r in range(50)]
    out = Parallel(n_jobs=-1)(jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df[df.scenario == "redundant"].dropna(axis=1, how="all").to_csv("results/scenarios_redundant.csv", index=False)
    df[df.scenario == "conditional"].dropna(axis=1, how="all").to_csv("results/scenarios_conditional.csv", index=False)


if __name__ == "__main__":
    main()
