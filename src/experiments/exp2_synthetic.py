import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.baselines import make_method
from src.datasets import make_correlated
from src.metrics import predictive_metrics, synthetic_metrics


def main():
    smoke = os.getenv("ACP_SMOKE", "0") == "1"
    rhos = [0.9] if smoke else [0.7, 0.8, 0.9, 0.95, 0.99]
    alphas = [0, 0.6, 1.0] if smoke else [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    reps = range(2 if smoke else 10); rows = []
    for rho in rhos:
        for rep in reps:
            X, y, signals, _ = make_correlated(rho=rho, task_seed=42 + rep)
            tr, te = train_test_split(range(len(y)), test_size=.3, stratify=y, random_state=42+rep)
            for method in ["CART", "RRF-style"]:
                model = make_method(method, max_depth=6, min_samples_leaf=5).fit(X[tr], y[tr])
                rows.append({"rho":rho,"alpha":0,"rep":rep,"method":method,
                             **predictive_metrics(y[te], model.predict(X[te]), model.predict_proba(X[te])),
                             **synthetic_metrics(model, signals), "root_feature":model.root_feature,
                             "n_nodes":model.n_nodes,"depth":model.depth})
            for alpha in alphas:
                model = make_method("ACP-Gini", alpha=alpha, max_depth=6, min_samples_leaf=5).fit(X[tr], y[tr])
                rows.append({"rho":rho,"alpha":alpha,"rep":rep,"method":"ACP-Gini",
                             **predictive_metrics(y[te], model.predict(X[te]), model.predict_proba(X[te])),
                             **synthetic_metrics(model, signals), "root_feature":model.root_feature,
                             "n_nodes":model.n_nodes,"depth":model.depth})
    pd.DataFrame(rows).to_csv("results/synthetic_sweep.csv", index=False)

if __name__ == "__main__": main()
