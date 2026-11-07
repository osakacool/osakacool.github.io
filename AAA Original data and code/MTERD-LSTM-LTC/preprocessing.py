import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureTransform:
    """Training-participant-only scaling and PCA, with deterministic frame sampling."""
    def __init__(self, components=128, max_frames=100000, seed=2026):
        self.components, self.max_frames, self.seed = components, max_frames, seed
        self.scaler, self.pca = StandardScaler(), None

    def fit(self, arrays):
        factory = arrays if callable(arrays) else lambda: iter(arrays)
        # Reopen one training session at a time; do not hold the dataset in RAM.
        total = 0
        for x in factory():
            self.scaler.partial_fit(x)
            total += len(x)
        if self.max_frames is None or total <= self.max_frames:
            selected = np.arange(total)
        else:
            selected = np.sort(np.random.default_rng(self.seed).choice(total, self.max_frames, replace=False))
        chunks, offset = [], 0
        for x in factory():
            idx = selected[(selected>=offset) & (selected<offset+len(x))] - offset
            if len(idx):
                chunks.append(self.scaler.transform(x[idx]))
            offset += len(x)
        fit_x = np.concatenate(chunks)
        if min(fit_x.shape) < self.components:
            raise ValueError(f'PCA {self.components} exceeds training matrix rank bound {fit_x.shape}')
        self.pca = PCA(n_components=self.components, svd_solver='randomized', random_state=self.seed)
        self.pca.fit(fit_x)
        self.explained_variance = float(self.pca.explained_variance_ratio_.sum())
        self.fit_frames = len(fit_x)
        return self

    def transform(self, x):
        return self.pca.transform(self.scaler.transform(x)).astype(np.float32)
