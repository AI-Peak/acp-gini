import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets import load_wdbc, make_correlated
from src.metrics import predictive_metrics
from src.tree import ACPGiniTreeClassifier


def main():
    smoke=os.getenv("ACP_SMOKE","0")=="1"; rows=[]
    datasets={"WDBC":load_wdbc("Data")[:2],"Synthetic":make_correlated(rho=.9)[:2]}
    configs=[("pearson","product","global"),("spearman","product","global"),
             ("pearson","min","global"),("pearson","mean","global"),("pearson","product","node")]
    for name,(X,y) in datasets.items():
        repetitions=range(1 if smoke else 10)
        for rep in repetitions:
            tr,te=train_test_split(range(len(y)),test_size=.3,stratify=y,random_state=42+rep)
            for corr,agg,scope in configs:
                start=time.perf_counter(); m=ACPGiniTreeClassifier(alpha=.6,max_depth=6,min_samples_leaf=5,corr_method=corr,penalty_agg=agg,corr_scope=scope).fit(X[tr],y[tr])
                rows.append({"dataset":name,"rep":rep,"corr_method":corr,"penalty_agg":agg,"corr_scope":scope,"fit_seconds":time.perf_counter()-start,**predictive_metrics(y[te],m.predict(X[te]),m.predict_proba(X[te]))})
    pd.DataFrame(rows).to_csv("results/ablation.csv",index=False)

if __name__ == "__main__": main()

