import numpy as np

from .tree import ACPGiniTreeClassifier


def variance_inflation_factors(X):
    X = np.asarray(X, float)
    result = []
    for j in range(X.shape[1]):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        design = np.column_stack([np.ones(len(X)), others])
        prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        ss_res = np.sum((y - prediction) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        result.append(np.inf if r2 >= 1 else 1 / max(1 - r2, 1e-12))
    return np.asarray(result)


class VIFCARTClassifier:
    def __init__(self, threshold=10.0, **tree_kwargs):
        self.threshold = threshold; self.tree_kwargs = tree_kwargs

    def fit(self, X, y):
        X = np.asarray(X, float); kept = list(range(X.shape[1])); self.dropped_features_ = []
        while len(kept) > 1:
            vifs = variance_inflation_factors(X[:, kept])
            worst = int(np.argmax(vifs))
            if vifs[worst] < self.threshold:
                break
            self.dropped_features_.append(kept.pop(worst))
        self.kept_features_ = kept
        self.tree_ = ACPGiniTreeClassifier(alpha=0, **self.tree_kwargs).fit(X[:, kept], y)
        self.classes_ = self.tree_.classes_
        self.feature_importances_ = np.zeros(X.shape[1])
        self.feature_importances_[kept] = self.tree_.feature_importances_
        return self

    def predict(self, X): return self.tree_.predict(np.asarray(X)[:, self.kept_features_])
    def predict_proba(self, X): return self.tree_.predict_proba(np.asarray(X)[:, self.kept_features_])
    @property
    def root_feature(self):
        f = self.tree_.root_feature
        return None if f is None else self.kept_features_[f]
    @property
    def used_features(self): return {self.kept_features_[f] for f in self.tree_.used_features}
    @property
    def n_nodes(self): return self.tree_.n_nodes
    @property
    def depth(self): return self.tree_.depth
    def split_composition(self):
        return type(self.tree_.split_composition())({(d, self.kept_features_[f]): n for (d, f), n in self.tree_.split_composition().items()})


def make_method(name, alpha=0.6, rrf_lambda=0.7, **kwargs):
    if name == "CART": return ACPGiniTreeClassifier(alpha=0, **kwargs)
    if name == "ACP-Gini": return ACPGiniTreeClassifier(alpha=alpha, **kwargs)
    if name == "RRF-style": return ACPGiniTreeClassifier(alpha=0, penalty_mode="rrf", rrf_lambda=rrf_lambda, **kwargs)
    if name == "VIF+CART": return VIFCARTClassifier(**kwargs)
    raise ValueError(name)

