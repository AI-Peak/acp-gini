"""Figures 7-10: inference forest plot, path-level correlation mass, depth sensitivity and
the accuracy-redundancy operating curves for all eight datasets. Figure style follows the
publication rcParams used across the project (7 pt sans-serif, editable PDF text)."""
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False,
})
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
GREY, BLUE, RED, TEAL = "#8a8f98", "#2b6f8a", "#c44e52", "#4c9a8f"
MM = 1 / 25.4

stats = pd.read_csv("results/stats_tests.csv")
dstats = pd.read_csv("results/dataset_stats.csv").sort_values("pair_fraction_abs_r_gt_07", ascending=False)
ORDER = list(dstats.dataset)
LABEL = {d: f"{d} ({100*f:.1f}%)" for d, f in zip(dstats.dataset, dstats.pair_fraction_abs_r_gt_07)}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, text):
    ax.text(-0.02, 1.08, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def forest(ax, metric, scale, xlabel, relative=False):
    g = stats[(stats.baseline == "CART") & (stats.metric == metric)].set_index("dataset").loc[ORDER].copy()
    if relative:  # express the effect and its interval as a share of the CART value
        for c in ["diff", "ci_lo", "ci_hi"]:
            g[c] = g[c] / g["base"]
    y = np.arange(len(g))[::-1]
    sig = g.p_nb_holm < 0.05
    ax.axvline(0, color="#444", lw=0.8)
    for yi, (_, r), s in zip(y, g.iterrows(), sig):
        ax.plot([r.ci_lo * scale, r.ci_hi * scale], [yi, yi], color=BLUE if s else GREY, lw=1.4)
        ax.plot(r["diff"] * scale, yi, "o", ms=4, color=BLUE if s else GREY, mfc=BLUE if s else "white", mew=1)
    ax.set_yticks(y); ax.set_xlabel(xlabel); ax.set_ylim(-0.7, len(g) - 0.3)
    return [LABEL[d] for d in g.index]


# Figure 7: forest plot of ACP-Gini minus CART with Nadeau-Bengio 95% intervals
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 70 * MM), sharey=True, gridspec_kw={"wspace": 0.08})
labels = forest(axes[0], "bootstrap_weighted_redundancy", 100, "Weighted redundancy change (%)", relative=True)
axes[0].set_yticklabels(labels)
forest(axes[1], "accuracy", 100, "Accuracy difference (percentage points)")
forest(axes[2], "importance_rank_corr", 1, "Importance rank correlation difference")
for ax, t in zip(axes, "abc"):
    panel_label(ax, t)
axes[0].set_ylabel("Data set (share of pairs with |r| > 0.7)")
for ax in axes[1:]:
    ax.tick_params(labelleft=False)
axes[0].text(0.02, -0.28, "Filled markers, Holm-adjusted p < 0.05 (corrected resampled t-test).\nBars, 95% intervals over 15 held-out folds.",
             transform=axes[0].transAxes, fontsize=6, color="#444", va="top")
save(fig, "fig7_forest")

# Figure 8: where the path-level correlation mass sits
bands = pd.read_csv("results/path_band_mass.csv").groupby(["dataset", "method"]).mean(numeric_only=True)
cols = ["band_0_0.3", "band_0.3_0.5", "band_0.5_0.7", "band_0.7_1"]
names = ["|r| < 0.3", "0.3 to 0.5", "0.5 to 0.7", "|r| ≥ 0.7"]
colors = ["#d9e4ea", "#9dbccb", "#e0a458", RED]
fig, ax = plt.subplots(figsize=(183 * MM, 62 * MM))
width = 0.36
for k, d in enumerate(ORDER):
    for off, m, hatch in [(-width / 2 - 0.01, "CART", ""), (width / 2 + 0.01, "ACP-Gini", "")]:
        bottom = 0.0
        for c, col, lab in zip(cols, colors, names):
            v = bands.loc[(d, m), c]
            ax.bar(k + off, v, width, bottom=bottom, color=col, edgecolor="white", lw=0.4,
                   label=lab if (k == 0 and m == "CART") else None)
            bottom += v
        ax.text(k + off, 1.015, "C" if m == "CART" else "A", ha="center", va="bottom", fontsize=6, color="#333")
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABEL[d].replace(" (", "\n(") for d in ORDER])
ax.set_ylabel("Share of path feature-pair mass"); ax.set_ylim(0, 1.08)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.07), title=None, columnspacing=1.2, handlelength=1.2)
ax.text(1.0, -0.30, "C, CART; A, ACP-Gini. Pair mass is weighted by training samples at each leaf; means over 15 folds.",
        transform=ax.transAxes, ha="right", fontsize=6, color="#444")
save(fig, "fig8_path_mass")

# Figure 9: depth-cap sensitivity
d = pd.read_csv("results/depth_sensitivity.csv")
g = d.groupby(["dataset", "depth_cap", "method"])[["accuracy", "weighted_redundancy"]].mean().unstack("method")
dacc = (g["accuracy"]["ACP-Gini"] - g["accuracy"]["CART"]) * 100
rred = (g["weighted_redundancy"]["ACP-Gini"] / g["weighted_redundancy"]["CART"] - 1) * 100
palette = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), gridspec_kw={"wspace": 0.28})
for i, ds in enumerate(ORDER):
    for ax, series in zip(axes, [rred, dacc]):
        s = series.xs(ds, level="dataset")
        ax.plot(s.index, s.values, marker="o", ms=3, lw=1, color=palette(i), label=ds)
for ax, yl, t in zip(axes, ["Weighted redundancy change (%)", "Accuracy change (percentage points)"], "ab"):
    ax.axhline(0, color="#444", lw=0.8); ax.set_xlabel("Maximum tree depth"); ax.set_ylabel(yl)
    ax.set_xticks([3, 4, 6, 8, 10]); panel_label(ax, t)
axes[1].legend(ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.02), handlelength=1.2)
save(fig, "fig9_depth")

# Figure 10: operating curves for all eight data sets (mean over 15 folds)
sw = pd.read_csv("results/real_alpha_sweep.csv")
q = sw.groupby(["dataset", "alpha"])[["accuracy", "weighted_redundancy"]].mean().reset_index()
sel = pd.read_csv("results/uci_main.csv").query("method=='ACP-Gini'").groupby("dataset").selected_alpha.first()
fig, axes = plt.subplots(2, 4, figsize=(183 * MM, 88 * MM), sharey=True, gridspec_kw={"hspace": 0.62, "wspace": 0.12})
for ax, ds in zip(axes.ravel(), ORDER):
    s = q[q.dataset == ds]
    base_r = float(s[s.alpha == 0].weighted_redundancy.iloc[0]); base_a = float(s[s.alpha == 0].accuracy.iloc[0])
    ax.plot((s.weighted_redundancy / base_r - 1) * 100, (s.accuracy - base_a) * 100, "-", color=GREY, lw=0.9, zorder=1)
    ax.scatter((s.weighted_redundancy / base_r - 1) * 100, (s.accuracy - base_a) * 100, s=14, color=BLUE, zorder=2)
    a = float(sel[ds]); r = s[np.isclose(s.alpha, a)].iloc[0]
    ax.scatter([(r.weighted_redundancy / base_r - 1) * 100], [(r.accuracy - base_a) * 100], s=34, facecolor="none",
               edgecolor=RED, lw=1.1, zorder=3)
    ax.axhline(0, color="#444", lw=0.6); ax.axvline(0, color="#444", lw=0.6)
    ax.set_title(ds, fontsize=7, fontweight="bold", pad=3)
for ax in axes[-1]:
    ax.set_xlabel("Redundancy change (%)")
for ax in axes[:, 0]:
    ax.set_ylabel("Accuracy change (pp)")
axes[0, 0].set_ylim(-4.6, 1.3)
fig.text(0.5, -0.04, "Each point is one alpha in {0.2,...,1.0} relative to alpha = 0 (CART); red ring marks the selected alpha.",
         ha="center", fontsize=6, color="#444")
save(fig, "fig10_operating_curves")

paper_dir = Path("paper/figures"); paper_dir.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("fig*.png"):
    shutil.copy(f, paper_dir / f.name)
