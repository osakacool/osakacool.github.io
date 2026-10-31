import numpy as np
from scipy import stats

class StatisticalValidator:
    """
    对应论文 5.3 节 & Supplementary Table S1/S2 Note：
    提供严谨的统计检验代码，证明 LSTM-LTC 的提升具有统计学和实际临床意义。
    """
    
    @staticmethod
    def cohens_d(y1, y2):
        """
        计算 Cohen's d 效应量 (对应 Supp Table S2)。
        审稿人看重：不仅要看 p-value，还要看 effect size（实际提升幅度）。
        """
        n1, n2 = len(y1), len(y2)
        var1, var2 = np.var(y1, ddof=1), np.var(y2, ddof=1)
        # 计算合并标准差 (Pooled Standard Deviation)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(y1) - np.mean(y2)) / (pooled_std + 1e-8)

    @staticmethod
    def bootstrap_confidence_interval(data, n_bootstraps=10000, ci=0.95):
        """
        计算 Bootstrap 95% CI (对应 Supp Table S1 Note)。
        不依赖正态分布假设，通过 10,000 次重采样评估指标精度。
        """
        bootstrapped_means = []
        n = len(data)
        for _ in range(n_bootstraps):
            sample = np.random.choice(data, n, replace=True)
            bootstrapped_means.append(np.mean(sample))
        lower = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
        return lower, upper

    @staticmethod
    def holm_bonferroni_correction(p_values, alpha=0.05):
        """
        Holm-Bonferroni 多重比较校正 (对应 5.3 节 & Table 2/3 Note)。
        相比传统 Bonferroni，它在控制 Family-wise error rate 的同时保留了更高的统计效力。
        """
        p_values = np.array(p_values)
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        adjusted_p = np.zeros(n)
        for i, p in enumerate(sorted_p):
            adjusted_p[i] = p * (n - i)
            
        # 确保单调递增 (Cumulative maximum)
        for i in range(1, n):
            adjusted_p[i] = max(adjusted_p[i], adjusted_p[i-1])
            
        adjusted_p = np.minimum(adjusted_p, 1.0)
        
        # 恢复原始顺序
        final_adjusted_p = np.zeros(n)
        final_adjusted_p[sorted_indices] = adjusted_p
        
        return final_adjusted_p < alpha, final_adjusted_p

    @staticmethod
    def kendalls_tau_with_ci(pred_trends, gt_labels, n_bootstraps=10000):
        """计算 Kendall's tau 及其 Bootstrap 95% CI (对应 Supp Table S1)"""
        tau, p_value = stats.kendalltau(pred_trends, gt_labels)
        taus = []
        n = len(pred_trends)
        for _ in range(n_bootstraps):
            idx = np.random.randint(0, n, n)
            t, _ = stats.kendalltau(pred_trends[idx], gt_labels[idx])
            taus.append(t)
        ci_lower, ci_upper = np.percentile(taus, 2.5), np.percentile(taus, 97.5)
        return tau, p_value, (ci_lower, ci_upper)