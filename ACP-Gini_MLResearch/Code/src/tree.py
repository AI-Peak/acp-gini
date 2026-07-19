from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import rankdata


RANDOM_STATE = 42


@dataclass
class _Node:
    prediction: int
    probabilities: np.ndarray
    n_samples: int
    impurity: float
    depth: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    raw_gain: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class ACPGiniTreeClassifier:
    def __init__(self, alpha=0.0, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, corr_method="pearson", corr_scope="global",
                 penalty_agg="product", random_state=RANDOM_STATE,
                 penalty_mode="acp", rrf_lambda=0.7):
        self.alpha = float(alpha)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.corr_method = corr_method
        self.corr_scope = corr_scope
        self.penalty_agg = penalty_agg
        self.random_state = random_state
        self.penalty_mode = penalty_mode
        self.rrf_lambda = float(rrf_lambda)

    @staticmethod
    def _gini(counts):
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return float(1.0 - np.dot(p, p))

    def _correlation(self, X):
        values = X
        if self.corr_method == "spearman":
            values = np.apply_along_axis(rankdata, 0, X)
        elif self.corr_method != "pearson":
            raise ValueError("corr_method must be 'pearson' or 'spearman'")
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(values, rowvar=False)
        if np.ndim(corr) == 0:
            corr = np.ones((1, 1))
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and aligned with y")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self._X, self._y = X, encoded
        self._global_corr = self._correlation(X)
        self._used_global = set()
        self._importance_raw = np.zeros(self.n_features_in_)
        self._root = self._grow(np.arange(len(y)), 0, set())
        total = self._importance_raw.sum()
        self.feature_importances_ = self._importance_raw / total if total else self._importance_raw.copy()
        del self._X, self._y
        return self

    def _penalty(self, feature, ancestors, corr):
        if self.penalty_mode == "rrf":
            return 1.0 if feature in self._used_global else self.rrf_lambda
        # A feature is never penalized against itself: continuous features
        # legitimately split multiple times along a path.
        others = [p for p in ancestors if p != feature]
        if not others or self.alpha == 0:
            return 1.0
        terms = np.clip(1.0 - self.alpha * corr[feature, others], 0.0, 1.0)
        if self.penalty_agg == "product":
            return float(np.prod(terms))
        if self.penalty_agg == "min":
            return float(np.min(terms))
        if self.penalty_agg == "mean":
            return float(np.mean(terms))
        raise ValueError("penalty_agg must be product, min, or mean")

    def _best_for_feature(self, idx, feature, parent_impurity):
        values = self._X[idx, feature]
        order = np.argsort(values, kind="mergesort")
        sv, sy = values[order], self._y[idx][order]
        valid = sv[:-1] < sv[1:]
        positions = np.flatnonzero(valid) + 1
        positions = positions[(positions >= self.min_samples_leaf) &
                              ((len(idx) - positions) >= self.min_samples_leaf)]
        if positions.size == 0:
            return None
        onehot = np.eye(self.n_classes_, dtype=float)[sy]
        cumulative = np.cumsum(onehot, axis=0)
        left_counts = cumulative[positions - 1]
        total = cumulative[-1]
        right_counts = total - left_counts
        left_n = positions.astype(float)
        right_n = len(idx) - left_n
        left_p = left_counts / left_n[:, None]
        right_p = right_counts / right_n[:, None]
        left_g = 1.0 - np.sum(left_p * left_p, axis=1)
        right_g = 1.0 - np.sum(right_p * right_p, axis=1)
        gains = parent_impurity - (left_n * left_g + right_n * right_g) / len(idx)
        best = int(np.argmax(gains))
        threshold = float((sv[positions[best] - 1] + sv[positions[best]]) / 2.0)
        return float(gains[best]), threshold

    def _grow(self, idx, depth, ancestors):
        counts = np.bincount(self._y[idx], minlength=self.n_classes_)
        impurity = self._gini(counts)
        node = _Node(int(np.argmax(counts)), counts / counts.sum(), len(idx), impurity, depth)
        stopped = (impurity <= 0 or len(idx) < self.min_samples_split or
                   (self.max_depth is not None and depth >= self.max_depth))
        if stopped:
            return node
        corr = self._global_corr
        if self.corr_scope == "node" and len(idx) >= 30:
            corr = self._correlation(self._X[idx])
        best = None
        for feature in range(self.n_features_in_):
            candidate = self._best_for_feature(idx, feature, impurity)
            if candidate is None:
                continue
            raw_gain, threshold = candidate
            score = raw_gain * self._penalty(feature, ancestors, corr)
            key = (score, raw_gain, -feature)
            if best is None or key > best[0]:
                best = (key, feature, threshold, raw_gain)
        # Stop only when no split yields real impurity reduction; the penalty
        # reorders candidates but never turns an informative node into a leaf.
        if best is None or best[3] <= 1e-15:
            return node
        _, feature, threshold, raw_gain = best
        mask = self._X[idx, feature] <= threshold
        left_idx, right_idx = idx[mask], idx[~mask]
        if min(len(left_idx), len(right_idx)) < self.min_samples_leaf:
            return node
        node.feature, node.threshold, node.raw_gain = feature, threshold, raw_gain
        self._importance_raw[feature] += len(idx) / len(self._y) * raw_gain
        self._used_global.add(feature)
        next_ancestors = set(ancestors)
        next_ancestors.add(feature)
        node.left = self._grow(left_idx, depth + 1, next_ancestors)
        node.right = self._grow(right_idx, depth + 1, next_ancestors)
        return node

    def _leaf(self, row):
        node = self._root
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.vstack([self._leaf(row).probabilities for row in X])

    def predict(self, X):
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]

    def tree_structure(self):
        def convert(node):
            if node.feature is None:
                return {"prediction": self.classes_[node.prediction].item(), "n": node.n_samples}
            return {"feature": node.feature, "threshold": node.threshold,
                    "left": convert(node.left), "right": convert(node.right)}
        return convert(self._root)

    @property
    def n_nodes(self):
        def count(node):
            return 1 + (count(node.left) + count(node.right) if node.feature is not None else 0)
        return count(self._root)

    @property
    def depth(self):
        def walk(node):
            return node.depth if node.feature is None else max(walk(node.left), walk(node.right))
        return walk(self._root)

    @property
    def root_feature(self):
        return self._root.feature

    @property
    def used_features(self):
        found = set()
        def walk(node):
            if node.feature is not None:
                found.add(node.feature); walk(node.left); walk(node.right)
        walk(self._root)
        return found

    def split_composition(self):
        result = Counter()
        def walk(node):
            if node.feature is not None:
                result[(node.depth, node.feature)] += 1; walk(node.left); walk(node.right)
        walk(self._root)
        return result

