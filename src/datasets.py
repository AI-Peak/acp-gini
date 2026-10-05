from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine

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


def load_extra(data_dir="Data"):
    """Three scikit-learn bundled datasets, binarized (no download needed)."""
    out = {}
    ds = load_diabetes()  # classic multicollinear design (s1/s2 r=0.90); target above median
    out["Diabetes"] = _cache("diabetes", ds.data, (ds.target > np.median(ds.target)).astype(int), ds.feature_names, data_dir)
    ds = load_wine()  # cultivar 0 versus the other two
    out["Cultivar"] = _cache("cultivar", ds.data, (ds.target == 0).astype(int), ds.feature_names, data_dir)
    ds = load_digits()  # confusable pair 3 versus 8; constant pixels dropped
    keep = np.isin(ds.target, [3, 8]); Xd = ds.data[keep]; Xd = Xd[:, Xd.std(axis=0) > 0]
    names = [f"pixel_{i}" for i in np.flatnonzero(ds.data[keep].std(axis=0) > 0)]
    out["Digits38"] = _cache("digits38", Xd, (ds.target[keep] == 8).astype(int), names, data_dir)
    return out


def _read_cached(name, data_dir):
    frame = pd.read_csv(Path(data_dir) / f"{name}.csv")
    return frame.drop(columns="target").to_numpy(float), frame.target.to_numpy(), list(frame.columns[:-1])


def load_strong(data_dir="Data"):
    """Four UCI data sets with strongly correlated predictors (share of pairs with |r|>0.7 between 8% and 44%).
    Downloaded once from the UCI repository and cached as CSV in Data/."""
    out = {}
    cache = {"Parkinsons": "parkinsons", "Musk1": "musk1", "Landsat": "landsat", "Ozone": "ozone8"}
    if all((Path(data_dir) / f"{f}.csv").exists() for f in cache.values()):
        return {name: _read_cached(f, data_dir) for name, f in cache.items()}
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=174)  # Parkinsons: voice recordings, status = Parkinson's disease
    X = ds.data.features.apply(pd.to_numeric)
    _cache("parkinsons", X.to_numpy(), ds.data.targets.iloc[:, 0].to_numpy(), X.columns, data_dir)
    ds = fetch_ucirepo(id=74)  # Musk version 1: 166 conformation descriptors, class = musk
    X = ds.data.features.drop(columns=["molecule_name", "conformation_name"]).apply(pd.to_numeric)
    _cache("musk1", X.to_numpy(), ds.data.targets.iloc[:, 0].astype(int).to_numpy(), X.columns, data_dir)
    ds = fetch_ucirepo(id=146)  # Statlog Landsat Satellite: class 1 (red soil) against the other five classes
    X = ds.data.features.apply(pd.to_numeric)
    _cache("landsat", X.to_numpy(), (ds.data.targets.iloc[:, 0].to_numpy() == 1).astype(int), X.columns, data_dir)
    # Ozone level detection: the eight-hour task only (the packaged data set mixes the one-hour and eight-hour files)
    import io
    import urllib.request
    import zipfile
    url = "https://archive.ics.uci.edu/static/public/172/ozone+level+detection.zip"
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())) as z:
        raw = pd.read_csv(z.open("eighthr.data"), header=None, na_values="?")
    X = raw.iloc[:, 1:-1].astype(float)
    X = X.fillna(X.median())
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    _cache("ozone8", X.to_numpy(), raw.iloc[:, -1].astype(int).to_numpy(), X.columns, data_dir)
    return {name: _read_cached(f, data_dir) for name, f in cache.items()}


def load_all_real(data_dir="Data", extra=True, strong=True):
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
    if extra:
        datasets.update(load_extra(data_dir))
    if strong:
        datasets.update(load_strong(data_dir))
    return datasets


def collinearity_fraction(X, threshold=0.7):
    corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)), nan=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(upper > threshold)) if len(upper) else 0.0
