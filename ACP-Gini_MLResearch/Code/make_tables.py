from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

OUT=Path("paper/generated"); OUT.mkdir(parents=True,exist_ok=True)
for stale in OUT.glob("table*.tex"):
    stale.unlink()

HEADER_MAP={
    "dataset":"Data set","method":"Method","accuracy":"Accuracy","macro_f1":"Macro-F1","auc":"AUC",
    "main_set_redundancy":"Set red. (main)","main_weighted_redundancy":"Weighted red. (main)",
    "bootstrap_set_redundancy":"Set red. (boot.)","bootstrap_weighted_redundancy":"Weighted red. (boot.)",
    "feature_set_jaccard":"Feature-set Jaccard","top5_jaccard":"Top-5 Jaccard",
    "importance_rank_corr":"Importance rank corr.","structural_distance":"Split-comp. distance",
    "n_nodes":"Nodes","depth":"Depth","vif_dropped":"VIF dropped","selected_alpha":"$\\alpha$",
    "selected_rrf_lambda":"$\\lambda$","corr_method":"Correlation","penalty_agg":"Aggregation",
    "corr_scope":"Scope","fit_seconds":"Fit time (s)","alpha":"$\\alpha$",
    "set_redundancy":"Set redundancy","weighted_redundancy":"Weighted redundancy",
    "n":"Samples","d":"Features","positive_rate":"Positive rate",
    "pair_fraction_abs_r_gt_07":"Pairs with $|r|>0.7$",
}

def latex(frame,path,index=False):
    if isinstance(frame.columns,pd.MultiIndex):
        frame=frame.copy()
        frame.columns=[HEADER_MAP.get(a,a) if not b else f"{HEADER_MAP.get(a,a)} ({'mean' if b=='mean' else 'std.' if b=='std' else b})" for a,b in frame.columns]
    else:
        frame=frame.rename(columns=HEADER_MAP)
    path.write_text(frame.to_latex(index=index,float_format=lambda x:f"{x:.3f}",escape=False),encoding="utf-8")

stats=pd.read_csv("results/dataset_stats.csv")
latex(stats,OUT/"table1_datasets.tex")
main=pd.read_csv("results/uci_main.csv")
pred=main.groupby(["dataset","method"])[["accuracy","macro_f1","auc"]].agg(["mean","std"]).reset_index()
stab=main.groupby(["dataset","method"])[["feature_set_jaccard","top5_jaccard","importance_rank_corr","structural_distance"]].mean().reset_index()
redundancy=main.groupby(["dataset","method"])[["main_set_redundancy","main_weighted_redundancy","bootstrap_set_redundancy","bootstrap_weighted_redundancy"]].mean().reset_index()
complexity=main.groupby(["dataset","method"])[["n_nodes","depth","vif_dropped"]].mean().reset_index()
latex(pred,OUT/"table2_predictive.tex"); latex(redundancy,OUT/"table3_redundancy.tex"); latex(stab,OUT/"table4_stability.tex"); latex(complexity,OUT/"table5_complexity.tex")
selected=main.groupby("dataset")[["selected_alpha","selected_rrf_lambda"]].first().reset_index()
latex(selected,OUT/"table_selected_parameters.tex")
if Path("results/real_alpha_sweep.csv").exists():
    sweep=pd.read_csv("results/real_alpha_sweep.csv")
    sweep_summary=sweep.groupby(["dataset","alpha"])[["accuracy","macro_f1","set_redundancy","weighted_redundancy"]].mean().reset_index()
    latex(sweep_summary,OUT/"table_real_alpha_sweep.tex")
latex(pd.read_csv("results/ablation.csv").groupby(["dataset","corr_method","penalty_agg","corr_scope"])[["accuracy","macro_f1","fit_seconds"]].mean().reset_index(),OUT/"table6_ablation.tex")
latex(pd.read_csv("results/runtime.csv").groupby(["dataset","method"]).fit_seconds.agg(["mean","std"]).reset_index(),OUT/"table7_runtime.tex")

tests=[]
for dataset in main.dataset.unique():
    block=main[main.dataset==dataset]
    acp=block[block.method=="ACP-Gini"].sort_values(["seed","fold"])
    for baseline in ["CART","VIF+CART","RRF-style"]:
        other=block[block.method==baseline].sort_values(["seed","fold"])
        for metric in ["accuracy","macro_f1","auc","feature_set_jaccard","top5_jaccard","importance_rank_corr","structural_distance","main_set_redundancy","main_weighted_redundancy","bootstrap_set_redundancy","bootstrap_weighted_redundancy"]:
            diff=acp[metric].to_numpy()-other[metric].to_numpy()
            try: p=float(wilcoxon(diff).pvalue)
            except ValueError: p=1.0
            tests.append({"dataset":dataset,"baseline":baseline,"metric":metric,"mean_difference":float(diff.mean()),"p_value":p})
pd.DataFrame(tests).to_csv("results/wilcoxon_tests.csv",index=False)
