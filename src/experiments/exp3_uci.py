import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.baselines import make_method
from src.datasets import collinearity_fraction, load_all_real
from src.experiments.common import evaluate_dataset


def main():
    datasets = load_all_real("Data"); frames=[]; stats=[]
    requested = os.getenv("ACP_DATASETS")
    selected_names = set(requested.split(",")) if requested else set(datasets)
    for name,(X,y,features) in datasets.items():
        stats.append({"dataset":name,"n":len(X),"d":X.shape[1],"positive_rate":float(y.mean()),
                      "pair_fraction_abs_r_gt_07":collinearity_fraction(X)})
        if name in selected_names:
            frames.append(evaluate_dataset(name,X,y))
    fresh = pd.concat(frames,ignore_index=True)
    main_path = "results/uci_main.csv"
    if requested and os.path.exists(main_path):
        previous = pd.read_csv(main_path)
        main = pd.concat([previous[~previous.dataset.isin(selected_names)], fresh], ignore_index=True)
    else:
        main = fresh
    pd.DataFrame(stats).to_csv("results/dataset_stats.csv",index=False)
    main.to_csv("results/uci_main.csv",index=False)

    if requested and "WDBC" not in selected_names and os.path.exists("results/wdbc_error_analysis.csv"):
        return
    X, y, _ = datasets["WDBC"]
    errors = []
    seeds = sorted(main.seed.unique())
    n_folds = int(main.fold.max()) + 1
    for seed in seeds:
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=int(seed))
        for fold, (train, test) in enumerate(cv.split(X, y)):
            alpha = float(main[(main.dataset == "WDBC") & (main.seed == seed) &
                               (main.fold == fold)].selected_alpha.iloc[0])
            kwargs = dict(max_depth=6, min_samples_leaf=5, random_state=seed)
            cart = make_method("CART", **kwargs).fit(X[train], y[train])
            acp = make_method("ACP-Gini", alpha=alpha, **kwargs).fit(X[train], y[train])
            cp, ap = cart.predict(X[test]), acp.predict(X[test])
            disagree = cp != ap
            boundary = ((cart.predict_proba(X[test])[:, 1] >= .4) & (cart.predict_proba(X[test])[:, 1] <= .6)) | ((acp.predict_proba(X[test])[:, 1] >= .4) & (acp.predict_proba(X[test])[:, 1] <= .6))
            errors.append({"seed":seed,"fold":fold,"selected_alpha":alpha,"n_test":len(test),
                           "disagreements":int(disagree.sum()),
                           "disagreements_near_boundary":int((disagree & boundary).sum()),
                           "cart_correct_acp_wrong":int(((cp==y[test]) & (ap!=y[test])).sum()),
                           "acp_correct_cart_wrong":int(((ap==y[test]) & (cp!=y[test])).sum())})
    pd.DataFrame(errors).to_csv("results/wdbc_error_analysis.csv", index=False)

if __name__ == "__main__": main()
