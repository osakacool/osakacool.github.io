"""Metrics with explicit conventions; see docs/metric_definitions.md."""
import numpy as np
from scipy.stats import pearsonr

def ccc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denominator = np.var(y) + np.var(p) + (y.mean()-p.mean())**2
    if denominator == 0:
        return float('nan')  # Undefined agreement for two identical constant series.
    return float(2*np.mean((y-y.mean())*(p-p.mean()))/denominator)

def block_means(x, scale):
    n = len(x)//scale
    return np.asarray(x[:n*scale]).reshape(n, scale).mean(axis=1)

def dtw_distance(y, p, band_fraction=0.1, warp_penalty=0.01):
    """Absolute local cost, additive off-diagonal penalty, path-length normalization.

    The manuscript leaves these choices unspecified; this is a named operational
    definition, not a claim to reproduce its original MS-DTW implementation.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    n, m = len(y), len(p)
    if not n or not m:
        return float('nan')
    width = max(abs(n-m), int(np.ceil(max(n,m)*band_fraction)), 1)
    previous = np.full(m+1, np.inf); previous[0] = 0
    lengths = np.zeros(m+1, dtype=int)
    for i in range(1, n+1):
        current = np.full(m+1, np.inf); clen = np.zeros(m+1, dtype=int)
        for j in range(max(1,i-width), min(m,i+width)+1):
            choices = (previous[j-1], previous[j]+warp_penalty, current[j-1]+warp_penalty)
            best = int(np.argmin(choices))
            current[j] = abs(y[i-1]-p[j-1]) + choices[best]
            clen[j] = (lengths[j-1] if best==0 else lengths[j] if best==1 else clen[j-1])+1
        previous, lengths = current, clen
    return float(previous[m]/lengths[m])

def ms_dtw(y, p, scales=(1,5,10), band_fraction=0.1, warp_penalty=0.01):
    scores = [dtw_distance(block_means(y,s), block_means(p,s),
                          band_fraction, warp_penalty) for s in scales if len(y)//s>=2]
    return float(np.mean(scores)) if scores else float('nan')

def tec(y, p, reference='centered', start_points=10):
    y, p = np.asarray(y), np.asarray(p)
    if reference == 'centered':
        return float(np.mean(np.sign(y-y.mean()) == np.sign(p-p.mean())))
    if reference == 'initial':
        return float(np.mean(np.sign(y-y[:start_points].mean()) ==
                             np.sign(p-p[:start_points].mean())))
    raise ValueError('TEC reference must be centered or initial')

def regression(y, p):
    y, p = np.asarray(y,float), np.asarray(p,float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError('Targets and predictions must be matching nonempty 1D arrays')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metrics reject nonfinite values')
    r = float(pearsonr(y,p).statistic) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else float('nan')
    return dict(RMSE=float(np.sqrt(np.mean((y-p)**2))), Pearson_r=r, CCC=ccc(y,p))
