"""Leakage-safe feature scaling and PCA for pre-extracted acoustic feature matrices."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AcousticPreprocessor:
    def __init__(self,n_components=128):
        self.scaler=StandardScaler();self.pca=PCA(n_components=n_components,svd_solver='randomized',random_state=42)
    def fit(self,training_frames):
        x=np.asarray(training_frames,dtype=float)
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Expected finite frame-by-feature matrix')
        self.pca.fit(self.scaler.fit_transform(x));return self
    def transform(self,frames):
        x=np.asarray(frames,dtype=float)
        if not np.isfinite(x).all():raise ValueError('Missing/nonfinite feature values')
        return self.pca.transform(self.scaler.transform(x))
    def fit_transform(self,training_frames):return self.fit(training_frames).transform(training_frames)

def pool_100ms_to_1hz(frame_predictions):
    """Aggregate each complete block of 10 consecutive 100-ms predictions."""
    x=np.asarray(frame_predictions,dtype=float).ravel();n=len(x)//10
    if not n:raise ValueError('At least ten frames required')
    return x[:n*10].reshape(n,10).mean(axis=1)
