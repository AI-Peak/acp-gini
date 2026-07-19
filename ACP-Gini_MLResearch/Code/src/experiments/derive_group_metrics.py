from pathlib import Path

import numpy as np
import pandas as pd


def derive(frame):
    importance_cols = [f"importance_{i}" for i in range(25)]
    missing = [column for column in importance_cols if column not in frame]
    if missing:
        raise ValueError(f"Missing importance columns: {missing}")
    values = frame[importance_cols].to_numpy(float)
    groups = [[j, 5 + 3 * j, 6 + 3 * j, 7 + 3 * j] for j in range(5)]
    coverage, concentration = [], []
    for row in values:
        top5 = set(np.argsort(row)[::-1][:5])
        coverage.append(sum(bool(top5 & set(group)) for group in groups) / 5)
        ratios = []
        for group in groups:
            mass = float(row[group].sum())
            ratios.append(float(row[group].max() / mass) if mass > 0 else 0.0)
        concentration.append(float(np.mean(ratios)))
    frame = frame.copy()
    frame["group_coverage"] = coverage
    frame["group_concentration"] = concentration
    frame["noise_importance"] = values[:, 20:25].sum(axis=1)
    return frame


def main():
    path = Path("results/synthetic_sweep.csv")
    derive(pd.read_csv(path)).to_csv(path, index=False)
    print(f"Derived group metrics for {path}")


if __name__ == "__main__":
    main()
