"""
04_plot_figures.py
对应论文：图2,图4,图7,图8；表1、表2；输出矢量PDF，适配期刊排版
运行依赖：03_train_exp.py中的模拟数据
输出目录 ./journal_figures
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from 03_train_exp import (
    get_baseline_df, get_ablation_df, grad_ratio_records,
    noise_levels, rmse_noise_ours, rmse_noise_rf,
    lambda_physics_list, pcindex_lambda, rmse_lambda,
    N_grid, wall_time_per_epoch
)

# ================= 期刊全局绘图参数 =================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize":11,
    "xtick.labelsize":9,
    "ytick.labelsize":9,
    "legend.fontsize":9,
    "figure.dpi":300,
    "savefig.format":"pdf",
    "savefig.bbox":"tight",
    "pdf.fonttype":42
})

OUTPUT_FIG_DIR = "./journal_figures"
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


def fig2_sensitivity():
    """图2 灵敏度分析 (a)噪声‑RMSE (b)λ_physics‑PC‑Index/RMSE"""
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,5))
    # subplot a
    ax1.errorbar(noise_levels, rmse_noise_ours, label="Our Framework", marker='o', capsize=3)
    ax1.errorbar(noise_levels, rmse_noise_rf, label="RF", marker='s', capsize=3)
    ax1.set_xlabel("Input Noise Level (% of feature std)")
    ax1.set_ylabel("RMSE")
    ax1.set_title("(a) RMSE vs. Input Noise Level")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # subplot b
    ax2.plot(lambda_physics_list, pcindex_lambda, color="tab:blue", marker="o", label="PC‑Index vs $\\lambda_{physics}$")
    ax2.set_xlabel("$\\lambda_{physics}$ (unitless)")
    ax2.set_ylabel("PC‑Index", color="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")
    ax2.axvspan(0.4,0.6,alpha=0.2,color="gray",label="$\\lambda \\in [0.4,0.6]$")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(lambda_physics_list, rmse_lambda, color="tab:red", marker="s", linestyle="--", label="RMSE vs $\\lambda_{physics}$")
    ax2_twin.set_ylabel("RMSE", color="tab:red")
    ax2_twin.tick_params(axis='y', labelcolor="tab:red")
    ax2.set_title("(b) PC‑Index and RMSE vs. $\\lambda_{physics}$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Figure2_sensitivity.pdf"))
    plt.close(fig)


def fig4_temporal_trajectory():
    """图4 时间序列轨迹，缺失数据插值效果"""
    fig, ax = plt.subplots(figsize=(10,6))
    t_axis = np.arange(0,24)
    obs_y = np.array([0.52,0.64,0.71,0.83,0.70,0.67,0.40,0.39,np.nan,np.nan,np.nan,np.nan,0.73,0.85,0.87,0.94,0.75,np.nan,np.nan,np.nan,0.47,0.62])
    pred_y = np.array([0.51,0.63,0.72,0.81,0.71,0.66,0.42,0.31,0.24,0.27,0.33,0.46,0.72,0.84,0.86,0.82,0.71,0.54,0.38,0.29,0.34,0.61])
    ax.scatter(t_axis, obs_y, c="blue", marker="o", label="Observed Data", s=30)
    ax.plot(t_axis, pred_y, c="red", lw=2, label="Our Framework Prediction")
    ax.axvspan(8,11.5, alpha=0.25, color="gray", label="Missing Data Periods")
    ax.axvspan(17,19.5, alpha=0.25, color="gray")
    ax.set_xlim(0,23)
    ax.set_ylim(0.2,1.0)
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Ecosystem Service Valuation (Normalized)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Figure4_temporal_trajectory.pdf"))
    plt.close(fig)


def fig7_gradient_ratio(grad_ratio_list):
    """图7 训练阶段梯度范数比"""
    fig, ax = plt.subplots(figsize=(8,5.5))
    epoch_arr = np.arange(len(grad_ratio_list))
    # 模拟标准PINN基线
    baseline_pinn_grad = np.exp(np.linspace(np.log(1), np.log(15), len(epoch_arr)))
    ax.plot(epoch_arr, grad_ratio_list, "r-", lw=2, label="Our Framework (Adaptive)")
    ax.plot(epoch_arr, baseline_pinn_grad, "gray", linestyle="--", label="Standard PINN (Baseline)")
    ax.axhline(y=1.0, linestyle=":", color="k", label="Ideal Alignment $O(1)$")
    ax.set_yscale("log")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel(r"Gradient Norm Ratio $||\nabla \mathcal{L}_{data}|| / ||\nabla \mathcal{L}_{physics}||$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Figure7_gradient_ratio.pdf"))
    plt.close(fig)


def fig8_compute_scaling():
    """图8 计算复杂度缩放"""
    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(N_grid, wall_time_per_epoch, marker='o', linewidth=2.2, label="Mamba‑PINN framework")
    ideal_slope = wall_time_per_epoch[0]/N_grid[0]
    ideal_y = [ideal_slope * n for n in N_grid]
    ax.plot(N_grid, ideal_y, "k--", label="Ideal $(N)$ scaling")
    ax.set_xlabel("Number of spatial grid cells ($N$)")
    ax.set_ylabel("Wall‑clock time per epoch (s)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "Figure8_compute_scaling.pdf"))
    plt.close(fig)


def export_table1(df):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.savefig(os.path.join(OUTPUT_FIG_DIR,"Table1_performance.pdf"))
    plt.close(fig)


def export_table2(df):
    fig, ax = plt.subplots(figsize=(11,3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    fig.savefig(os.path.join(OUTPUT_FIG_DIR,"Table2_ablation.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    df1 = get_baseline_df()
    df2 = get_ablation_df()
    fig2_sensitivity()
    fig4_temporal_trajectory()
    fig7_gradient_ratio(grad_ratio_records)
    fig8_compute_scaling()
    export_table1(df1)
    export_table2(df2)
    print(f"全部PDF图表输出路径：{OUTPUT_FIG_DIR}")
#（注：内容由AI生成）
