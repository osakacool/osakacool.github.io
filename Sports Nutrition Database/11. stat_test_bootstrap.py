# -*- coding:utf-8 -*-
"""
论文统计分析：Paired Bootstrap重采样 + 配对t‑test
用于对比 tri‑model ensemble vs Transformer‑only baseline
指标：NA, CSR, PD, GF
"""
import numpy as np
from scipy import stats


def paired_bootstrap_diff(arr_a, arr_b, n_resample=20000, alpha=0.05):
    """
    arr_a:主模型每个测试样本指标数组
    arr_b:对比基线每个测试样本指标数组
    return: mean_diff, ci_low, ci_high
    """
    assert len(arr_a) == len(arr_b)
    diff_origin = arr_a - arr_b
    n = len(diff_origin)
    resample_diffs = []
    np.random.seed(42)
    for _ in range(n_resample):
        idx = np.random.randint(0, n, size=n)
        sample_diff = diff_origin[idx]
        resample_diffs.append(np.mean(sample_diff))
    resample_diffs = np.array(resample_diffs)
    mean_diff = np.mean(diff_origin)
    ci_low = np.percentile(resample_diffs, 100*alpha/2)
    ci_high = np.percentile(resample_diffs, 100*(1-alpha/2))
    # bootstrap p‑value: 看0是否在置信区间内
    p_val = np.sum(resample_diffs <= 0)/n_resample if mean_diff>0 else np.sum(resample_diffs >=0)/n_resample
    return mean_diff, ci_low, ci_high, p_val


if __name__ == "__main__":
    # 仿真测试样本，替换成真实测试集每个样本指标
    np.random.seed(42)
    n_sample = 1860  #论文测试集样本数量
    # 主模型
    na_ensemble = np.random.normal(loc=85.2, scale=3.0, size=n_sample)
    csr_ensemble = np.random.normal(loc=97.9, scale=1.5, size=n_sample)
    pd_ensemble = np.random.normal(loc=0.63, scale=0.08, size=n_sample)
    gf_ensemble = np.random.normal(loc=0.66, scale=0.07, size=n_sample)

    # Transformer‑only基线
    na_trans = np.random.normal(loc=80.6, scale=3.2, size=n_sample)
    csr_trans = np.random.normal(loc=93.4, scale=1.8, size=n_sample)
    pd_trans = np.random.normal(loc=0.51, scale=0.09, size=n_sample)
    gf_trans = np.random.normal(loc=0.58, scale=0.08, size=n_sample)

    metrics = [
        ("NA(%)", na_ensemble, na_trans),
        ("CSR(%)", csr_ensemble, csr_trans),
        ("PD", pd_ensemble, pd_trans),
        ("GF", gf_ensemble, gf_trans)
    ]

    print(f"{'指标':<8}{'均值差值':<12}{'95%CI_low':<12}{'95%CI_high':<12}{'Bootstrap p':<12}{'paired‑t p':<12}")
    for name, arrA, arrB in metrics:
        md, cil, cih, p_boot = paired_bootstrap_diff(arrA, arrB, n_resample=10000)
        t_stat, p_t = stats.ttest_rel(arrA, arrB)
        print(f"{name:<8}{md:<12.3f}{cil:<12.3f}{cih:<12.3f}{p_boot:<12.4f}{p_t:<12.4f}")
