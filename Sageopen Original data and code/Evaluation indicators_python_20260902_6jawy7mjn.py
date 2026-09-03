from scipy.stats import pearsonr, kendalltau
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def calc_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient (Eq. 16 基础)"""
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return 2 * covariance / (var_true + var_pred + (mean_true - mean_pred)**2)

def calc_tec(y_true, y_pred):
    """Therapeutic Effect Consistency (Eq. 18): 趋势识别二值准确率"""
    # 计算 session 内的改善/恶化趋势
    trend_true = np.sign(y_true[-1] - y_true[0])
    trend_pred = np.sign(y_pred[-1] - y_pred[0])
    return float(trend_true == trend_pred)

def calc_ms_dtw(y_true, y_pred):
    """Multi-scale DTW (Eq. 17): 多尺度动态时间规整距离"""
    # 简化版多尺度DTW：在不同下采样率下计算DTW并求均值
    distances = []
    for scale in [1, 2, 4]:
        y_t = y_true[::scale]
        y_p = y_pred[::scale]
        distance, _ = fastdtw(y_t.reshape(-1, 1), y_p.reshape(-1, 1), dist=euclidean)
        distances.append(distance)
    return np.mean(distances)

def calc_kendall_tau(tec_preds, ground_truth_labels):
    """Supplementary Table S1: Kendall's rank correlation"""
    tau, p_value = kendalltau(tec_preds, ground_truth_labels)
    return tau, p_value