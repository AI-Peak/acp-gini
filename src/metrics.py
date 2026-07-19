from collections import Counter
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def predictive_metrics(y, pred, proba):
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "auc": roc_auc_score(y, proba[:, 1])}


def _mean_pairs(values, fn):
    pairs = list(combinations(values, 2))
    return float(np.mean([fn(a, b) for a, b in pairs])) if pairs else 1.0


def stability_metrics(trees):
    roots = [t.root_feature for t in trees]
    root_agreement = Counter(roots).most_common(1)[0][1] / len(roots)
    jaccard = _mean_pairs([t.used_features for t in trees], lambda a, b: len(a & b) / len(a | b) if a | b else 1)
    def rank_corr(a, b):
        value = spearmanr(a, b).statistic
        return 0.0 if np.isnan(value) else value
    rank = _mean_pairs([t.feature_importances_ for t in trees], rank_corr)
    def multi_jaccard(a, b):
        keys = set(a) | set(b); inter = sum(min(a[k], b[k]) for k in keys); union = sum(max(a[k], b[k]) for k in keys)
        return inter / union if union else 1.0
    distance = _mean_pairs([t.split_composition() for t in trees], lambda a, b: 1 - multi_jaccard(a, b))
    return {"root_agreement": root_agreement, "feature_set_jaccard": jaccard,
            "importance_rank_corr": rank, "structural_distance": distance}


def synthetic_metrics(tree, signal_idx, n_signal=5, n_copies=15):
    top = np.argsort(tree.feature_importances_)[::-1][:5]
    return {"signal_recovery_rate": len(set(top) & set(signal_idx)) / 5,
            "copy_dilution": float(tree.feature_importances_[n_signal:n_signal+n_copies].sum())}

