"""
metrics_utils.py
期刊投稿专用指标工具
实现论文全部评价指标：RMSE, RRMSE, R², PC‑Index(Eq.39‑40), Moran's I残差空间自相关，95%置信区间
"""
import numpy as np
import torch
from scipy import stats
import pysal.lib as ps
from pysal.explore import esda


def rmse(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))


def rrmse(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """Relative‑RMSE 论文公式(37)"""
    mu = np.mean(y_true)
    return rmse(y_true,y_pred)/mu *100.0


def r2_score_custom(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot


def pc_index(physics_loss:float, beta:float=2.2, v_max:float=1.0) -> float:
    """
    Physical Consistency Index PC‑Index，论文公式(39‑40)
    :param physics_loss: PDE残差损失
    :param beta: scale coefficient
    :return: pc‑index ∈ [0, 1] higher=better physical plausibility
    """
    vi = min(physics_loss, v_max)
    pc = np.exp(-beta * vi)
    return float(np.clip(pc, 0.0,1.0))


def moran_i_residual(y_true:np.ndarray, y_pred:np.ndarray, w:ps.weights.W):
    """
    Moran’s I for prediction residuals 论文6.2空间残差自相关
    :param y_true: ground truth
    :param y_pred: prediction
    :param w: pysal spatial weight matrix
    :return: I statistic, p‑value
    """
    res = y_true - y_pred
    mi = esda.Moran(res, w)
    return mi.I, mi.p_norm


def compute_95ci(arr:np.ndarray):
    """
    计算均值 ±95%置信区间，多seed重复实验
    return mean, ci_lower, ci_upper
    """
    arr = np.asarray(arr)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    ci = stats.t.interval(confidence=0.95, df=len(arr)-1, loc=mean, scale=sem)
    return float(mean), float(ci[0]), float(ci[1])


def wilcoxon_test(arr_base:np.ndarray, arr_compare:np.ndarray):
    """Wilcoxon signed‑rank test 论文统计检验"""
    stat, p = stats.wilcoxon(arr_base, arr_compare)
    return {"statistic":stat, "p_value":p}


def torch_to_np(tensor:torch.Tensor):
    """安全转换tensor到numpy，detach cpu"""
    return tensor.detach().cpu().numpy().reshape(-1)
#（注：内容由AI生成）
