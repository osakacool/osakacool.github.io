import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class TimeConstantAnalyzer:
    """
    对应论文 Eq. 11, Figure 3 & Supplementary Figure S1：
    从模型权重中提取 tau，计算有效记忆范围 H_i，并进行 GMM 聚类和 BIC 验证。
    """
    def __init__(self, model):
        self.model = model

    def extract_effective_memory_horizons(self):
        """
        提取所有 256 个神经元的 Effective Memory Horizon (Eq. 11)
        """
        horizons = []
        # 遍历 2 层 LSTM-LTC Cell (每层 128 神经元，共 256)
        for layer in self.model.cells:
            b_tau = layer.b_tau.detach().cpu().numpy()
            
            # 提取 forget gate 的 bias (对应 Eq. 8 中的 tau_f)
            tau_f_logits = b_tau[:layer.hidden_size] 
            # 还原 tau 值 (范围 0.1 ~ 10)
            tau_f = 0.1 + 9.9 * (1.0 / (1.0 + np.exp(-tau_f_logits)))
            
            # Eq. 11: 假设帧间隔 dt = 0.1s，H_i = tau * dt
            H_i = tau_f * 0.1 
            horizons.extend(H_i)
            
        return np.array(horizons)

    def fit_gmm_and_validate(self, horizons, max_k=10):
        """
        拟合 GMM 并计算 BIC, Silhouette Score, Calinski-Harabasz Index
        用于生成 Supplementary Figure S1 和验证 Figure 3 的聚类质量。
        """
        horizons = horizons.reshape(-1, 1)
        results = {}
        
        for k in range(1, max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
            gmm.fit(horizons)
            labels = gmm.predict(horizons)
            
            bic = gmm.bic(horizons)
            sil = silhouette_score(horizons, labels) if k > 1 else -1
            ch = calinski_harabasz_score(horizons, labels) if k > 1 else 0
            
            results[k] = {
                'bic': bic,
                'silhouette': sil,
                'calinski_harabasz': ch,
                'means': gmm.means_.flatten(),
                'model': gmm
            }
            
        # 找到 BIC 最小的 K (论文中证明 K=3 是全局最优)
        best_k = min(results, key=lambda x: results[x]['bic'])
        return best_k, results