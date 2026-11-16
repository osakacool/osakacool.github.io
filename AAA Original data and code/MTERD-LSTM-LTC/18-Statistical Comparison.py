"""Paired fold-level inference and Holm-adjusted p-values."""
import numpy as np
from scipy.stats import ttest_rel,t
from statsmodels.stats.multitest import multipletests

def paired_comparisons(reference,baselines):
    a=np.asarray(reference,dtype=float);results={};names=[];pvals=[]
    for name,values in baselines.items():
        b=np.asarray(values,dtype=float)
        if a.shape!=b.shape or len(a)<2:raise ValueError('Paired vectors must match and contain at least 2 folds')
        delta=a-b;sd=np.std(delta,ddof=1)
        p=float(ttest_rel(a,b).pvalue) if sd>0 else (0.0 if np.mean(delta)!=0 else 1.0)
        results[name]={'mean_difference':float(np.mean(delta)),'cohens_dz':float(np.mean(delta)/sd) if sd>0 else None,'p_raw':p}
        names.append(name);pvals.append(p)
    adjusted=multipletests(pvals,method='holm')[1]
    for name,p in zip(names,adjusted):results[name]['p_holm']=float(p)
    return results

def mean_ci(values,confidence=0.95):
    x=np.asarray(values,dtype=float)
    if len(x)<2:raise ValueError('At least two fold values required')
    mean=float(np.mean(x));margin=float(t.ppf((1+confidence)/2,len(x)-1)*np.std(x,ddof=1)/np.sqrt(len(x)))
    return mean-margin,mean+margin
