from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42


def make_correlated(n_samples=1000, n_signal=5, n_copies_per_signal=3,
                    rho=0.9, n_noise=5, task_seed=RANDOM_STATE):
    rng = np.random.default_rng(task_seed)
    z = rng.normal(size=(n_samples, n_signal))
    weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])[:n_signal]
    y = (z @ weights + rng.normal(0, 0.5, n_samples) > 0).astype(int)
    copies = [rho * z[:, [j]] + np.sqrt(1 - rho ** 2) * rng.normal(size=(n_samples, 1))
              for j in range(n_signal) for _ in range(n_copies_per_signal)]
    noise = rng.normal(size=(n_samples, n_noise))
    X = np.hstack([z] + copies + [noise])
    names = ([f"signal_{i+1}" for i in range(n_signal)] +
             [f"copy_{j+1}_{k+1}" for j in range(n_signal) for k in range(n_copies_per_signal)] +
             [f"noise_{i+1}" for i in range(n_noise)])
    return X, y, set(range(n_signal)), names


def _cache(name, X, y, feature_names, data_dir):
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(X, columns=feature_names); frame["target"] = y
    frame.to_csv(data_dir / f"{name}.csv", index=False)
    return X.astype(float), np.asarray(y), list(feature_names)


def load_wdbc(data_dir="Data"):
    ds = load_breast_cancer()
    return _cache("wdbc", ds.data, ds.target, ds.feature_names, data_dir)


def load_uci(dataset_id, name, data_dir="Data", target_transform=None):
    path = Path(data_dir) / f"{name}.csv"
    if path.exists():
        frame = pd.read_csv(path)
        return frame.drop(columns="target").to_numpy(float), frame.target.to_numpy(), list(frame.columns[:-1])
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=dataset_id)
    X = ds.data.features.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = ds.data.targets.iloc[:, 0]
    if target_transform:
        y = target_transform(y)
    else:
        y = pd.factorize(y)[0]
    return _cache(name, X.to_numpy(), np.asarray(y), X.columns, data_dir)


def load_all_real(data_dir="Data"):
    datasets = {"WDBC": load_wdbc(data_dir)}
    specs = [
        (186, "Wine", lambda y: (pd.to_numeric(y) >= 6).astype(int)),
        (52, "Ionosphere", None), (151, "Sonar", None),
    ]
    for dataset_id, name, transform in specs:
        datasets[name] = load_uci(dataset_id, name.lower(), data_dir, transform)
    pima_path = Path(data_dir) / "pima.csv"
    if pima_path.exists():
        frame = pd.read_csv(pima_path)
        datasets["Pima"] = (frame.drop(columns="target").to_numpy(float), frame.target.to_numpy(), list(frame.columns[:-1]))
    else:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=37, as_frame=True, parser="auto")
        X = ds.data.apply(pd.to_numeric, errors="coerce").fillna(0)
        y = (ds.target.astype(str) == "tested_positive").astype(int)
        datasets["Pima"] = _cache("pima", X.to_numpy(), y, X.columns, data_dir)
    return datasets


def collinearity_fraction(X, threshold=0.7):
    corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)), nan=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(upper > threshold)) if len(upper) else 0.0
