import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import linregress

class AdvancedMetricsCalculator:
    """
    对应论文 Eq. 16, 17, 18 & Table 3：
    实现专为本文设计的多尺度时间评估指标。
    """
    
    @staticmethod
    def temporal_concordance_correlation(y_true, y_pred):
        """TCC (Eq. 16): 衡量连续预测的精度和时间对齐 (Lin's CCC 变体)"""
        mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
        var_true, var_pred = np.var(y_true), np.var(y_pred)
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        
        ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)
        return ccc

    @staticmethod
    def multi_scale_dtw(y_true, y_pred, scales=[1, 2, 4]):
        """
        MS-DTW (Eq. 17): 多尺度 DTW 距离。
        在不同下采样率下计算 DTW 并求均值，评估模型对多尺度模式的对齐能力。
        """
        distances = []
        for scale in scales:
            y_t = y_true[::scale]
            y_p = y_pred[::scale]
            distance, _ = fastdtw(y_t.reshape(-1, 1), y_p.reshape(-1, 1), dist=euclidean)
            distances.append(distance)
        return np.mean(distances)

    @staticmethod
    def therapeutic_effect_consistency(y_true, y_pred):
        """
        TEC (Eq. 18): 治疗趋势一致性。
        评估模型是否正确识别了 session 内的改善/恶化趋势 (二值准确率)。
        """
        # 将 session 分为前后两半，计算均值差以判断宏观趋势
        mid = len(y_true) // 2
        trend_true = np.mean(y_true[mid:]) - np.mean(y_true[:mid])
        trend_pred = np.mean(y_pred[mid:]) - np.mean(y_pred[:mid])
        
        # 趋势方向一致得 1，不一致得 0
        return float(np.sign(trend_true) == np.sign(trend_pred))