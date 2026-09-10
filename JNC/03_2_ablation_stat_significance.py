"""
03_2_ablation_stat_significance.py
论文补充材料：消融实验统计显著性检验
执行Wilcoxon符号秩检验，对比Full Framework与各个消融变体
输出：ablation_stat_test.txt
"""
import numpy as np
from scipy.stats import wilcoxon

# 模拟多次重复运行（5次重复，模拟5个随机种子）
# 真实使用替换为多次运行得到的RMSE/R²/PC‑Index数组
np.random.seed(42)

if __name__ == "__main__":
    # 模拟5次重复实验结果
    full_rmse = np.array([0.109,0.111,0.113,0.110,0.114])
    abl1_rmse = np.array([0.134,0.139,0.136,0.141,0.135]) # pure pinn w/o mamba
    abl2_rmse = np.array([0.122,0.127,0.124,0.126,0.129]) # pure mamba w/o pinn
    abl3_rmse = np.array([0.118,0.123,0.120,0.122,0.124]) # w/o gis
    abl4_rmse = np.array([0.114,0.119,0.116,0.117,0.121]) # w/o adaptive

    out_txt = "ablation_stat_test.txt"
    with open(out_txt,"w",encoding="utf‑8") as f:
        f.write("Wilcoxon signed‑rank test (Full Framework vs Ablation variants)\n")
        f.write("H0: No performance difference\n")
        for name, arr_abl in zip([
            "Pure PINN w/o Mamba",
            "Pure Mamba w/o PINN",
            "w/o GIS integration",
            "w/o adaptive weighting"],
            [abl1_rmse, abl2_rmse, abl3_rmse, abl4_rmse]):
            stat, p = wilcoxon(full_rmse, arr_abl)
            f.write(f"\n{name:25s} | statistic={stat:.3f} | p‑value={p:.4f}")
            if p <0.05:
                f.write(" → significant difference p<0.05")
            else:
                f.write(" → not significant")
    print(f"统计检验结果保存到 {out_txt}")
    with open(out_txt,"r",encoding="utf‑8") as f:
        print(f.read())
#（注：内容由AI生成）
