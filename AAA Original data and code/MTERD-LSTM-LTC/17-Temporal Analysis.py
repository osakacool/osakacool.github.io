"""Analyze learned time constants and select Gaussian mixture components."""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score,calinski_harabasz_score

def analyze_horizons(horizons,components=range(1,7),random_state=42):
    x=np.asarray(horizons,dtype=float).reshape(-1,1)
    if len(x)<max(components) or not np.isfinite(x).all() or np.any(x<=0):raise ValueError('Positive finite horizons and sufficient samples required')
    models={k:GaussianMixture(n_components=k,random_state=random_state).fit(x) for k in components}
    bic={k:float(m.bic(x)) for k,m in models.items()};k=min(bic,key=bic.get)
    labels=models[k].predict(x)
    return {'selected_components':k,'bic':bic,'centers_seconds':sorted(models[k].means_.ravel().tolist()),
            'silhouette':float(silhouette_score(x,labels)) if 1<k<len(x) else None,
            'calinski_harabasz':float(calinski_harabasz_score(x,labels)) if 1<k<len(x) else None}
