"""Full pair-level provenance for the path correlation bands: for every data set and method (CART,
ACP-Gini at the selected alpha), the share of sample-weighted path pair mass carried by each feature
pair with |r| >= 0.5, pooled over the 15 held-out-fold training trees (main fits, depth 6)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.baselines import make_method
from src.datasets import load_all_real
from src.experiments.exp7_extended import abs_corr_matrix


def main():
    main_df = pd.read_csv("results/uci_main.csv"); rows = []
    for name, (X, y, feats) in load_all_real("Data").items():
        alpha = float(main_df[(main_df.dataset == name) & (main_df.method == "ACP-Gini")].selected_alpha.iloc[0])
        corr = abs_corr_matrix(X)
        for label, a in [("CART", 0.0), ("ACP-Gini", alpha)]:
            mass, total = {}, 0.0
            for seed in [42, 43, 44]:
                for tr, _ in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                    m = make_method("ACP-Gini", alpha=a, max_depth=6, min_samples_leaf=5, random_state=seed).fit(X[tr], y[tr])
                    for path, n in m.leaf_paths():
                        f = sorted(set(path))
                        for p, i in enumerate(f):
                            for j in f[p + 1:]:
                                total += n
                                if corr[i, j] >= 0.5:
                                    mass[(i, j)] = mass.get((i, j), 0.0) + n
            high = sum(mass.values())
            for (i, j), v in sorted(mass.items(), key=lambda kv: -kv[1]):
                rows.append({"dataset": name, "method": label, "feature_i": feats[i], "feature_j": feats[j],
                             "abs_r": corr[i, j], "share_of_pairs_ge_0.5": v / high,
                             "share_of_all_pair_mass": v / total})
            if not mass:
                rows.append({"dataset": name, "method": label, "feature_i": "", "feature_j": "", "abs_r": np.nan,
                             "share_of_pairs_ge_0.5": 0.0, "share_of_all_pair_mass": 0.0})
    pd.DataFrame(rows).to_csv("results/path_pairs_high_corr.csv", index=False)


if __name__ == "__main__":
    main()
