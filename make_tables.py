from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

OUT=Path("paper/generated"); OUT.mkdir(parents=True,exist_ok=True)

def latex(frame,path,index=False):
    path.write_text(frame.to_latex(index=index,float_format=lambda x:f"{x:.3f}",escape=True),encoding="utf-8")

stats=pd.read_csv("results/dataset_stats.csv")
latex(stats,OUT/"table1_datasets.tex")
main=pd.read_csv("results/uci_main.csv")
pred=main.groupby(["dataset","method"])[["accuracy","macro_f1","auc"]].agg(["mean","std"]).reset_index()
stab=main.groupby(["dataset","method"])[["root_agreement","feature_set_jaccard","importance_rank_corr","structural_distance"]].mean().reset_index()
complexity=main.groupby(["dataset","method"])[["n_nodes","depth","vif_dropped"]].mean().reset_index()
latex(pred,OUT/"table2_predictive.tex"); latex(stab,OUT/"table3_stability.tex"); latex(complexity,OUT/"table4_complexity.tex")
latex(pd.read_csv("results/ablation.csv").groupby(["dataset","corr_method","penalty_agg","corr_scope"])[["accuracy","macro_f1","fit_seconds"]].mean().reset_index(),OUT/"table5_ablation.tex")
latex(pd.read_csv("results/runtime.csv").groupby(["dataset","method"]).fit_seconds.agg(["mean","std"]).reset_index(),OUT/"table6_runtime.tex")

tests=[]
for dataset in main.dataset.unique():
    block=main[main.dataset==dataset]
    acp=block[block.method=="ACP-Gini"].sort_values(["seed","fold"])
    for baseline in ["CART","VIF+CART","RRF-style"]:
        other=block[block.method==baseline].sort_values(["seed","fold"])
        for metric in ["accuracy","macro_f1","auc","root_agreement","feature_set_jaccard","importance_rank_corr","structural_distance"]:
            diff=acp[metric].to_numpy()-other[metric].to_numpy()
            try: p=float(wilcoxon(diff).pvalue)
            except ValueError: p=1.0
            tests.append({"dataset":dataset,"baseline":baseline,"metric":metric,"mean_difference":float(diff.mean()),"p_value":p})
pd.DataFrame(tests).to_csv("results/wilcoxon_tests.csv",index=False)
