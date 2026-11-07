from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from mterd.io import dump_json


def analyze(h, seed, max_components=10):
    x=np.asarray(h,float).reshape(-1,1)
    if len(x)<3 or not np.isfinite(x).all() or (x<=0).any():
        raise ValueError('Need at least three finite positive neuron horizons')
    candidates=[];models=[]
    for k in range(1,min(max_components,len(x)-1)+1):
        g=GaussianMixture(k,n_init=10,random_state=seed,reg_covar=1e-6).fit(x)
        models.append(g);candidates.append(dict(k=k,bic=float(g.bic(x)),converged=bool(g.converged_)))
    selected=int(np.argmin([c['bic'] for c in candidates]));best=models[selected]
    labels=best.predict(x);n_labels=len(set(labels.tolist()))
    order=np.argsort(best.means_.ravel());components=[]
    for rank,j in enumerate(order):
        components.append(dict(component_rank=rank+1,center_seconds=float(best.means_[j,0]),
                    sd_seconds=float(np.sqrt(best.covariances_[j,0,0])),weight=float(best.weights_[j])))
    validity=dict(best_k=best.n_components,n_neurons=len(x),
         silhouette=float(silhouette_score(x,labels)) if 1<n_labels<len(x) else None,
         calinski_harabasz=float(calinski_harabasz_score(x,labels)) if 1<n_labels<len(x) else None)
    return candidates,components,validity

def main():
    p=argparse.ArgumentParser(description='Unconstrained K=1..10 BIC selection per fold and seed')
    p.add_argument('--runs',required=True);p.add_argument('--out',default='results/metrics/gmm')
    p.add_argument('--column',choices=['horizon_seconds','retention_horizon_seconds'],default='horizon_seconds')
    a=p.parse_args();paths=sorted(Path(a.runs).glob('*/memory_horizons.csv'))
    if not paths:raise ValueError('No measured memory horizons found')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    bics=[];centers=[];validity=[];neurons=[]
    for path in paths:
        raw=pd.read_csv(path)
        for (fold,seed),frame in raw.groupby(['fold','seed']):
            # Equal session weights; then one observation per neuron for GMM.
            frame=frame.groupby(['layer','neuron'])[a.column].mean().reset_index()
            frame['fold']=int(fold);frame['seed']=int(seed);neurons.append(frame)
            b,c,v=analyze(frame[a.column],int(seed))
            bics.extend([dict(fold=int(fold),seed=int(seed),**r) for r in b])
            centers.extend([dict(fold=int(fold),seed=int(seed),best_k=v['best_k'],**r) for r in c])
            validity.append(dict(fold=int(fold),seed=int(seed),**v))
    pd.concat(neurons).to_csv(out/'neuron_horizons.csv',index=False)
    pd.DataFrame(bics).to_csv(out/'bic.csv',index=False)
    pd.DataFrame(centers).to_csv(out/'components.csv',index=False)
    pd.DataFrame(validity).to_csv(out/'validity.csv',index=False)
    dump_json(out/'definition.json',dict(column=a.column,gmm_space='raw_seconds',
             aggregation='mean_over_windows_then_equal_session_mean',forced_component_count=False))
    print(pd.DataFrame(validity).to_string(index=False))

if __name__=='__main__':main()
