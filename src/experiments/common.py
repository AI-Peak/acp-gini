import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.baselines import make_method
from src.metrics import predictive_metrics, stability_metrics

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def config():
    smoke = os.getenv("ACP_SMOKE", "0") == "1"
    return {"smoke": smoke, "seeds": [42] if smoke else [42, 43, 44],
            "folds": 2 if smoke else 5, "B": 10 if smoke else 50}


def _select_strength(X, y, train, mode, seed, smoke):
    grid = [0.2, 0.6, 1.0] if smoke and mode == "acp" else ([0.5, 0.7, 0.9] if mode == "rrf" else [0.2, 0.4, 0.6, 0.8, 1.0])
    inner = StratifiedKFold(2 if smoke else 3, shuffle=True, random_state=seed + 1000)
    base_f1 = []
    candidates = {value: {"f1": [], "stability": []} for value in grid}
    tune_bootstraps = 4 if smoke else 10
    for inner_fold, (sub_train, validation) in enumerate(inner.split(X[train], y[train])):
        fit_idx, val_idx = train[sub_train], train[validation]
        kwargs = dict(max_depth=6, min_samples_leaf=5, random_state=seed)
        base = make_method("CART", **kwargs).fit(X[fit_idx], y[fit_idx])
        base_f1.append(f1_score(y[val_idx], base.predict(X[val_idx]), average="macro"))
        for value in grid:
            method = "ACP-Gini" if mode == "acp" else "RRF-style"
            model = make_method(method, alpha=value, rrf_lambda=value, **kwargs).fit(X[fit_idx], y[fit_idx])
            candidates[value]["f1"].append(f1_score(y[val_idx], model.predict(X[val_idx]), average="macro"))
            rng = np.random.default_rng(seed * 10000 + inner_fold * 100 + int(value * 10))
            trees = []
            for _ in range(tune_bootstraps):
                boot = rng.choice(fit_idx, len(fit_idx), replace=True)
                trees.append(make_method(method, alpha=value, rrf_lambda=value, **kwargs).fit(X[boot], y[boot]))
            candidates[value]["stability"].append(stability_metrics(trees)["importance_rank_corr"])
    tolerance = float(np.std(base_f1, ddof=1)) if len(base_f1) > 1 else 0.0
    eligible = [value for value in grid if np.mean(candidates[value]["f1"]) >= np.mean(base_f1) - tolerance]
    if not eligible:
        eligible = [grid[0]]
    return max(eligible, key=lambda value: (np.mean(candidates[value]["stability"]), -value))


def evaluate_dataset(name, X, y):
    cfg = config(); rows = []
    for seed in cfg["seeds"]:
        cv = StratifiedKFold(cfg["folds"], shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(X, y)):
            alpha = _select_strength(X, y, train, "acp", seed + fold, cfg["smoke"])
            rrf_lambda = _select_strength(X, y, train, "rrf", seed + fold, cfg["smoke"])
            for method in ["CART", "VIF+CART", "RRF-style", "ACP-Gini"]:
                kwargs = dict(max_depth=6, min_samples_leaf=5, random_state=seed)
                model = make_method(method, alpha=alpha, rrf_lambda=rrf_lambda, **kwargs).fit(X[train], y[train])
                pred = model.predict(X[test]); proba = model.predict_proba(X[test])
                row = {"dataset": name, "method": method, "seed": seed, "fold": fold,
                       "selected_alpha": alpha, "selected_rrf_lambda": rrf_lambda,
                       **predictive_metrics(y[test], pred, proba),
                       "n_nodes": model.n_nodes, "depth": model.depth,
                       "vif_dropped": len(getattr(model, "dropped_features_", []))}
                rng = np.random.default_rng(seed * 100 + fold)
                trees = []
                for _ in range(cfg["B"]):
                    boot = rng.choice(train, len(train), replace=True)
                    trees.append(make_method(method, alpha=alpha, rrf_lambda=rrf_lambda, **kwargs).fit(X[boot], y[boot]))
                row.update(stability_metrics(trees)); rows.append(row)
    return pd.DataFrame(rows)
