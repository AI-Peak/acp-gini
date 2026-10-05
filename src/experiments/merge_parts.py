"""Merge the per-data-set parts written by exp3_uci (ACP_PART mode) into results/uci_main.csv and refresh
results/dataset_stats.csv for every data set."""
import glob
import os

import pandas as pd

from src.datasets import collinearity_fraction, load_all_real


def main():
    main_df = pd.read_csv("results/uci_main.csv")
    for path in sorted(glob.glob("results/_part_*.csv")):
        part = pd.read_csv(path)
        main_df = pd.concat([main_df[~main_df.dataset.isin(part.dataset.unique())], part], ignore_index=True)
        os.remove(path)
    main_df.to_csv("results/uci_main.csv", index=False)
    stats = [{"dataset": name, "n": len(X), "d": X.shape[1], "positive_rate": float(y.mean()),
              "pair_fraction_abs_r_gt_07": collinearity_fraction(X)} for name, (X, y, _) in load_all_real("Data").items()]
    pd.DataFrame(stats).to_csv("results/dataset_stats.csv", index=False)


if __name__ == "__main__":
    main()
