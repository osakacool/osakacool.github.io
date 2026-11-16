"""Evaluation metrics for continuous emotional-response predictions."""
import numpy as np
from scipy.stats import pearsonr, kendalltau

def _arrays(y_true, y_pred):
    a, b = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape or a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Equal-length finite arrays with at least two observations are required")
    return a, b

def rmse(y_true, y_pred):
    a,b=_arrays(y_true,y_pred)
    return float(np.sqrt(np.mean((a-b)**2)))

def pearson_r(y_true,y_pred):
    a,b=_arrays(y_true,y_pred)
    return float(pearsonr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else float('nan')

def ccc(y_true,y_pred):
    a,b=_arrays(y_true,y_pred)
    va,vb=np.var(a),np.var(b)
    denom=va+vb+(np.mean(a)-np.mean(b))**2
    return float(2*np.mean((a-np.mean(a))*(b-np.mean(b)))/denom) if denom>0 else float('nan')

def temporal_concordance(y_true,y_pred):
    """Operational TCC: Lin's CCC of aligned temporal observations."""
    return ccc(y_true,y_pred)

def dtw_distance(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    if not len(a) or not len(b): raise ValueError('Nonempty series required')
    prev=np.full(len(b)+1,np.inf);prev[0]=0
    for x in a:
        cur=np.full(len(b)+1,np.inf)
        for j,y in enumerate(b,1):cur[j]=abs(x-y)+min(prev[j],cur[j-1],prev[j-1])
        prev=cur
    return float(prev[-1]/max(len(a),len(b)))

def multiscale_dtw(y_true,y_pred,scales=(1,5,10)):
    """Mean normalized DTW over nonoverlapping pooled scales (configurable)."""
    a,b=_arrays(y_true,y_pred)
    def pool(x,s):
        n=len(x)//s
        if n<1:raise ValueError('Series shorter than pooling scale')
        return x[:n*s].reshape(n,s).mean(axis=1)
    return float(np.mean([dtw_distance(pool(a,s),pool(b,s)) for s in scales]))

def therapeutic_effect_consistency(y_true_sessions,y_pred_sessions):
    """Fraction of sessions with matching first-to-last trend signs; ties match ties."""
    if len(y_true_sessions)!=len(y_pred_sessions) or not y_true_sessions:raise ValueError('Matching nonempty session lists required')
    correct=[]
    for a,b in zip(y_true_sessions,y_pred_sessions):
        a,b=_arrays(a,b)
        correct.append(np.sign(a[-1]-a[0])==np.sign(b[-1]-b[0]))
    return float(np.mean(correct))

def session_kendall_tau(y_true,y_pred):
    a,b=_arrays(y_true,y_pred)
    return float(kendalltau(a,b,variant='b').statistic)

def summary(y_true,y_pred):
    return dict(rmse=rmse(y_true,y_pred),pearson_r=pearson_r(y_true,y_pred),ccc=ccc(y_true,y_pred),tcc=temporal_concordance(y_true,y_pred),ms_dtw=multiscale_dtw(y_true,y_pred))
