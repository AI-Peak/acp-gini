from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.baselines import make_method
from src.datasets import load_wdbc

OUT=Path("figures"); OUT.mkdir(exist_ok=True)
syn=pd.read_csv("results/synthetic_sweep.csv")
fig,axes=plt.subplots(1,2,figsize=(9,3.6))
cart=syn[syn.method=="CART"]
rrf=syn[syn.method=="RRF-style"]
selected=[]
for rho,g in syn[syn.method=="ACP-Gini"].groupby("rho"):
    cart_acc=cart[cart.rho==rho].accuracy
    floor=cart_acc.mean()-cart_acc.std(ddof=1)
    eligible=g.groupby("alpha").filter(lambda x:x.accuracy.mean()>=floor)
    means=eligible.groupby("alpha").group_coverage.mean()
    chosen=float(means.idxmax())
    selected.append(g[g.alpha==chosen])
best=pd.concat(selected)
for label,g in [("CART",cart),("RRF-style",rrf),("ACP-Gini (selected)",best)]:
    q=g.groupby("rho").agg(coverage=("group_coverage","mean"),concentration=("group_concentration","mean")).reset_index()
    axes[0].plot(q.rho,q.coverage,marker="o",label=label); axes[1].plot(q.rho,q.concentration,marker="o",label=label)
axes[0].set(xlabel="Correlation rho",ylabel="Group coverage"); axes[1].set(xlabel="Correlation rho",ylabel="Group concentration"); axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/"fig2_synthetic.png",dpi=300); fig.savefig(OUT/"fig2_synthetic.pdf")
block=syn[(syn.method=="ACP-Gini") & (syn.rho==.9)]
sens=block.groupby("alpha")[["accuracy","group_coverage","group_concentration","noise_importance"]].mean()
fig,axes=plt.subplots(1,2,figsize=(9,3.6)); sens[["accuracy","group_coverage","group_concentration"]].plot(marker="o",ax=axes[0]); sens[["noise_importance"]].plot(marker="o",ax=axes[1],color="#c44e52"); axes[0].set(xlabel="Alpha",ylabel="Mean metric"); axes[1].set(xlabel="Alpha",ylabel="Noise importance"); fig.tight_layout(); fig.savefig(OUT/"fig3_sensitivity.png",dpi=300); fig.savefig(OUT/"fig3_sensitivity.pdf")
main=pd.read_csv("results/uci_main.csv")

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

# Copy all figures to the paper figures directory to keep them in sync
paper_figures_dir = Path("paper/figures")
paper_figures_dir.mkdir(exist_ok=True, parents=True)
import shutil
for f in OUT.glob("fig*.png"):
    shutil.copy(f, paper_figures_dir / f.name)

