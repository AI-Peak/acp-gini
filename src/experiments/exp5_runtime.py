import os
import time
import pandas as pd

from src.datasets import load_all_real
from src.tree import ACPGiniTreeClassifier


def main():
    smoke=os.getenv("ACP_SMOKE","0")=="1"; rows=[]
    for name,(X,y,_) in load_all_real("Data").items():
        for method,alpha,scope in [("CART",0,"global"),("ACP-global",.6,"global"),("ACP-node",.6,"node")]:
            for rep in range(2 if smoke else 10):
                start=time.perf_counter(); ACPGiniTreeClassifier(alpha=alpha,corr_scope=scope,max_depth=6,min_samples_leaf=5).fit(X,y)
                rows.append({"dataset":name,"method":method,"rep":rep,"fit_seconds":time.perf_counter()-start})
    pd.DataFrame(rows).to_csv("results/runtime.csv",index=False)

if __name__ == "__main__": main()

