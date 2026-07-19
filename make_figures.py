from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.baselines import make_method
from src.datasets import load_wdbc

OUT=Path("figures"); OUT.mkdir(exist_ok=True)
syn=pd.read_csv("results/synthetic_sweep.csv")
fig,axes=plt.subplots(1,2,figsize=(9,3.6))
best=syn.copy(); best["label"]=best.apply(lambda r: r["method"] if r["method"]!="ACP-Gini" else f"ACP a={r['alpha']:.1f}",axis=1)
for label,g in best.groupby("label"):
    if label not in ["CART","RRF-style","ACP a=0.6"]: continue
    q=g.groupby("rho").agg(signal=("signal_recovery_rate","mean"),root=("root_feature",lambda x:x.value_counts(normalize=True).iloc[0])).reset_index()
    axes[0].plot(q.rho,q.signal,marker="o",label=label); axes[1].plot(q.rho,q.root,marker="o",label=label)
axes[0].set(xlabel="Correlation rho",ylabel="Signal recovery"); axes[1].set(xlabel="Correlation rho",ylabel="Root agreement"); axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/"fig2_synthetic.png",dpi=300); fig.savefig(OUT/"fig2_synthetic.pdf")
sens=syn[(syn.method=="ACP-Gini") & (syn.rho==.9)].groupby("alpha")[["accuracy","signal_recovery_rate"]].mean()
fig,ax=plt.subplots(figsize=(5.5,3.6)); sens.plot(marker="o",ax=ax); ax.set(xlabel="Alpha",ylabel="Mean metric"); fig.tight_layout(); fig.savefig(OUT/"fig3_sensitivity.png",dpi=300); fig.savefig(OUT/"fig3_sensitivity.pdf")
main=pd.read_csv("results/uci_main.csv"); pivot=main.groupby(["dataset","method"]).importance_rank_corr.mean().unstack()
ax=pivot.plot.bar(figsize=(8,4)); ax.set(ylabel="Importance rank correlation",xlabel=""); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/"fig4_importance_stability.png",dpi=300); plt.savefig(OUT/"fig4_importance_stability.pdf")

X,y,names=load_wdbc("Data")
alpha=float(main[(main.dataset=="WDBC") & (main.method=="ACP-Gini")].selected_alpha.mode().iloc[0])
cart=make_method("CART",max_depth=6,min_samples_leaf=5).fit(X,y)
acp=make_method("ACP-Gini",alpha=alpha,max_depth=6,min_samples_leaf=5).fit(X,y)
def draw_top(ax,tree,title):
    ax.axis("off"); ax.set_title(title,fontsize=11,fontweight="bold")
    def walk(node,x,y0,spread,depth):
        if depth>2 or "feature" not in node: return
        label=f"{names[node['feature']]}\n<= {node['threshold']:.3f}"
        ax.text(x,y0,label,ha="center",va="center",fontsize=7,bbox=dict(boxstyle="round,pad=.25",fc="#eef3f5",ec="#335c67"))
        for child,dx in [(node["left"],-spread),(node["right"],spread)]:
            if "feature" in child and depth<2:
                ax.plot([x,x+dx],[y0-.04,y0-.27],color="#777",lw=.8)
            walk(child,x+dx,y0-.31,spread*.52,depth+1)
    walk(tree.tree_structure(),.5,.88,.24,0); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig,axes=plt.subplots(1,2,figsize=(10,4)); draw_top(axes[0],cart,"CART: top three levels"); draw_top(axes[1],acp,f"ACP-Gini: top three levels (alpha={alpha:.1f})"); fig.tight_layout(); fig.savefig(OUT/"fig5_tree_comparison.png",dpi=300); fig.savefig(OUT/"fig5_tree_comparison.pdf")
