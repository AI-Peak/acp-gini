import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.tree import ACPGiniTreeClassifier


def main():
    lines = []
    for seed in range(5):
        datasets = [load_breast_cancer(return_X_y=True), load_iris(return_X_y=True),
                    make_classification(n_samples=500, n_features=12, n_informative=6, random_state=seed)]
        for di, (X, y) in enumerate(datasets):
            # A 40% holdout keeps the requested 0.01 accuracy tolerance meaningful
            # on the smallest dataset (one prediction is otherwise > 0.01).
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, stratify=y, random_state=seed)
            kwargs = dict(alpha=0, max_depth=6, min_samples_leaf=5, random_state=seed)
            pearson = ACPGiniTreeClassifier(corr_method="pearson", **kwargs).fit(Xtr, ytr)
            spearman = ACPGiniTreeClassifier(corr_method="spearman", **kwargs).fit(Xtr, ytr)
            assert json.dumps(pearson.tree_structure(), sort_keys=True) == json.dumps(spearman.tree_structure(), sort_keys=True)
            assert np.array_equal(pearson.predict(Xte), spearman.predict(Xte))
            sk = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=seed).fit(Xtr, ytr)
            delta = abs(np.mean(pearson.predict(Xte) == yte) - sk.score(Xte, yte))
            assert delta <= .01 + 1e-12, (seed, di, delta)
            lines.append(f"PASS seed={seed} dataset={di} sklearn_accuracy_delta={delta:.6f}")
    Path("results").mkdir(exist_ok=True); Path("results/sanity.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Sanity checks passed:", len(lines))

if __name__ == "__main__": main()
