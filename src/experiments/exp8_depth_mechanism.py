"""(a) Sensitivity to the depth cap. (b) How much of the path-level correlation mass sits in each
|r| band for CART versus ACP-Gini (pair-level detail is in exp11_path_pairs)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from src.baselines import make_method
from src.datasets import load_all_real
from src.experiments.exp7_extended import abs_corr_matrix, path_metrics
from src.metrics import predictive_metrics, redundancy_metrics

DEPTHS = [3, 4, 6, 8, 10]
BANDS = [(0, .3), (.3, .5), (.5, .7), (.7, 1.0001)]


def band_mass(model, corr):
    mass = np.zeros(len(BANDS))
    for path, n in model.leaf_paths():
        feats = sorted(set(path))
        for a, i in enumerate(feats):
            for j in feats[a + 1:]:
                r = corr[i, j]
                for k, (lo, hi) in enumerate(BANDS):
                    if lo <= r < hi:
                        mass[k] += n
    return mass


def fold_job(name, X, y, alpha, seed, fold, tr, te, corr):
    rows, band_rows = [], []
    for depth in DEPTHS:
        for label, a in [("CART", 0.0), ("ACP-Gini", alpha)]:
            m = make_method("ACP-Gini", alpha=a, max_depth=depth, min_samples_leaf=5, random_state=seed).fit(X[tr], y[tr])
            pm = predictive_metrics(y[te], m.predict(X[te]), m.predict_proba(X[te]))
            rows.append({"dataset": name, "depth_cap": depth, "method": label, "seed": seed, "fold": fold,
                         **pm, "n_nodes": m.n_nodes, **redundancy_metrics(m, corr),
                         "path_flagged": path_metrics(m, corr)[0]})
            if depth == 6:
                mass = band_mass(m, corr)
                band_rows.append({"dataset": name, "method": label, "seed": seed, "fold": fold,
                                  **{f"band_{lo}_{min(hi,1)}": v / mass.sum() if mass.sum() else 0.0
                                     for (lo, hi), v in zip(BANDS, mass)}})
    return rows, band_rows


def main():
    main_df = pd.read_csv("results/uci_main.csv"); R, Bd = [], []
    for name, (X, y, _) in load_all_real("Data").items():
        alpha = float(main_df[(main_df.dataset == name) & (main_df.method == "ACP-Gini")].selected_alpha.iloc[0])
        corr = abs_corr_matrix(X); jobs = []
        for seed in [42, 43, 44]:
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
                jobs.append(delayed(fold_job)(name, X, y, alpha, seed, fold, tr, te, corr))
        for rows, bands in Parallel(n_jobs=-1)(jobs):
            R += rows; Bd += bands
        print(name, "done", flush=True)
    pd.DataFrame(R).to_csv("results/depth_sensitivity.csv", index=False)
    pd.DataFrame(Bd).to_csv("results/path_band_mass.csv", index=False)


if __name__ == "__main__":
    main()
