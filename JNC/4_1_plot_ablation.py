"""
04_1_plot_ablation.py
消融结果可视化：期刊柱状对比图，读入 ablation_result.csv
输出PDF矢量图：
Figure_ablation_bar.pdf ：RMSE、R²、PC‑Index三组子图对比
对应论文6.5 Ablation Study，辅助图表
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size":11,
    "axes.labelsize":11,
    "xtick.labelsize":9,
    "ytick.labelsize":9,
    "legend.fontsize":9,
    "figure.dpi":300,
    "savefig.format":"pdf",
    "savefig.bbox":"tight",
    "pdf.fonttype":42
})

OUT_DIR = "./journal_figures"
os.makedirs(OUT_DIR, exist_ok=True)

def load_ablation_csv(filepath="ablation_result.csv"):
    rows = []
    with open(filepath,"r",encoding="utf‑8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "config_name":r["config_name"],
                "rmse":float(r["rmse"]),
                "r2":float(r["r2"]),
                "pcindex":float(r["pcindex"]),
                "train_h":float(r["train_h"])
            })
    return rows


def plot_ablation_bar():
    data = load_ablation_csv()
    names_short = [
        "Full",
        "PINN\nw/o Mamba",
        "Mamba\nw/o PINN",
        "w/o GIS",
        "w/o Adaptive λ"
    ]
    rmse_arr = [d["rmse"] for d in data]
    r2_arr = [d["r2"] for d in data]
    pc_arr = [d["pcindex"] for d in data]
    x = np.arange(len(names_short))
    width = 0.6

    fig, axes = plt.subplots(1,3, figsize=(14,4.5))
    ax0,ax1,ax2 = axes

    # RMSE：越低越好
    bars0 = ax0.bar(x, rmse_arr, width, color=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"])
    ax0.set_xticks(x)
    ax0.set_xticklabels(names_short)
    ax0.set_ylabel("RMSE")
    ax0.set_title("RMSE (lower better)")
    ax0.grid(axis="y", alpha=0.3)

    # R²：越高越好
    bars1 = ax1.bar(x, r2_arr, width, color=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_short)
    ax1.set_ylabel("$R^2$")
    ax1.set_title("$R^2$ (higher better)")
    ax1.grid(axis="y", alpha=0.3)

    # PC‑Index：越高越好
    bars2 = ax2.bar(x, pc_arr, width, color=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_short)
    ax2.set_ylabel("PC‑Index")
    ax2.set_title("PC‑Index (higher better)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Ablation Study Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,"Figure_ablation_bar.pdf"))
    plt.close(fig)
    print(f"消融柱状对比图输出至 {OUT_DIR}/Figure_ablation_bar.pdf")


if __name__ == "__main__":
    plot_ablation_bar()
#（注：内容由AI生成）
