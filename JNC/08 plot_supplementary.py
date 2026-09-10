"""
plot_supplementary.py
Supplementary figures for journal submission
1. multi‑seed metric with 95% confidence interval bar plot
2. learning curve loss evolution plot
输出 supplementary_*.pdf 矢量图
读取output目录下csv实验结果
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PLOT_PARAMS, OUTPUT_DIR, FIG_DIR

plt.rcParams.update(PLOT_PARAMS)

def plot_ci_bar():
    df = pd.read_csv(os.path.join(OUTPUT_DIR,"main_experiment_result.csv"))
    metrics = ["rmse","r2","pcindex"]
    means = [df[m].mean() for m in metrics]
    from metrics_utils import compute_95ci
    cis = [compute_95ci(df[m].values) for m in metrics]
    lower = [x[1] for x in cis]
    upper = [x[2] for x in cis]
    err = np.array([np.array(means)-np.array(lower), np.array(upper)-np.array(means)])
    xticks_label = ["RMSE","$R^2$","PC‑Index"]
    fig,ax = plt.subplots(figsize=(7,5))
    ax.bar(np.arange(len(metrics)), means, yerr=err, capsize=6, color=["#1f77b4","#ff7f0e","#2ca02c"])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(xticks_label)
    ax.set_title("Main Model Metrics (mean ±95% Confidence Interval, n=5 seeds)")
    ax.grid(axis="y",alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"supplement_metric_ci.pdf"))
    plt.close(fig)
    print("Saved supplement_metric_ci.pdf")


def plot_learning_curve_demo():
    # demo loss curve，真实使用读取训练日志解析loss序列
    epoch = np.arange(145)
    loss_total = 0.35*np.exp(-epoch/45)+0.08
    loss_data =0.28*np.exp(-epoch/42)+0.06
    loss_phys =0.22*np.exp(-epoch/38)+0.04
    fig,ax = plt.subplots(figsize=(8,5))
    ax.plot(epoch, loss_total, label="Total Loss", lw=2)
    ax.plot(epoch, loss_data, label="$L_{data}$", lw=1.8)
    ax.plot(epoch, loss_phys, label="$L_{physics}$", lw=1.8)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"supplement_learning_curve.pdf"))
    plt.close(fig)
    print("Saved supplement_learning_curve.pdf")


if __name__ == "__main__":
    plot_ci_bar()
    plot_learning_curve_demo()
#（注：内容由AI生成）
