import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.datasets import load_all_real
from src.metrics import predictive_metrics, redundancy_metrics
from src.tree import ACPGiniTreeClassifier


def main():
    smoke = os.getenv("ACP_SMOKE", "0") == "1"
    seeds = [42] if smoke else [42, 43, 44]
    folds = 2 if smoke else 5
    alphas = [0, 0.6, 1.0] if smoke else [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for dataset, (X, y, _) in load_all_real("Data").items():
        with np.errstate(invalid="ignore", divide="ignore"):
            abs_corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)), nan=0.0)
        for seed in seeds:
            cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(cv.split(X, y)):
                for alpha in alphas:
                    model = ACPGiniTreeClassifier(alpha=alpha, max_depth=6,
                                                   min_samples_leaf=5,
                                                   random_state=seed).fit(X[train], y[train])
                    row = {"dataset": dataset, "alpha": alpha, "seed": seed, "fold": fold,
                           **predictive_metrics(y[test], model.predict(X[test]), model.predict_proba(X[test])),
                           **redundancy_metrics(model, abs_corr)}
                    rows.append(row)
    pd.DataFrame(rows).to_csv("results/real_alpha_sweep.csv", index=False)


if __name__ == "__main__":
    main()
