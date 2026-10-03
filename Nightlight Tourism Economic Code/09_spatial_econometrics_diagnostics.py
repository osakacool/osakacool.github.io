import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
from esda.moran import Moran, Moran_Local
from spreg import ML_Lag
import matplotlib.pyplot as plt

class SpatialEconometricsPipeline:
    """
    Full pipeline for Spatial Autoregressive (SAR) modeling and diagnostics.
    Corresponds to Section 5.4: Economic Radiation & Spillover Effects.
    """
    def __init__(self, gdf, y_col='merchant_revenue', x_cols=['activity_index', 'poi_density']):
        self.gdf = gdf
        self.y = gdf[y_col].values
        self.X = gdf[x_cols].values
        self.y_col = y_col
        self.x_cols = x_cols

    def build_spatial_weights(self):
        """构建 Queen 邻接空间权重矩阵并进行行标准化"""
        # Queen Contiguity (共享边界或顶点即相邻)
        w = libpysal.weights.Queen.from_dataframe(self.gdf, use_index=False)
        w.transform = 'r' # 行标准化 (Row-standardization)，确保权重和为1
        self.w = w
        print(f"Spatial Weights Matrix built. Islands (no neighbors): {w.islands}")
        return w

    def run_morans_i_diagnostics(self):
        """
        1. 全局 Moran's I (检验因变量是否存在空间自相关)
        2. 局部 LISA (识别冷热点)
        """
        # 全局 Moran's I
        mi = Moran(self.y, self.w, permutations=999)
        print(f"Global Moran's I: {mi.I:.4f}, p-value: {mi.p_sim:.4f}")
        if mi.p_sim < 0.01:
            print("-> Significant spatial autocorrelation detected (p < 0.01). SAR model is justified.")
        
        # 局部 LISA (Getis-Ord Gi* 的替代方案，用于聚类分析)
        lisa = Moran_Local(self.y, self.w, permutations=999)
        self.gdf['lisa_cluster'] = lisa.q # 1=HH, 2=LH, 3=LL, 4=HL
        self.gdf['lisa_pval'] = lisa.p_sim
        
        return mi, lisa

    def fit_sar_model_and_decompose(self):
        """
        拟合 SAR (Spatial Lag) 模型，并计算直接效应与溢出效应
        """
        # 拟合 SAR 模型 (Maximum Likelihood)
        model = ML_Lag(
            y=self.y, x=self.X, w=self.w,
            name_y=self.y_col, name_x=self.x_cols,
            name_w='Spatial_Weights', name_ds='Study_Area'
        )
        print(model.summary)
        
        # 提取系数
        rho = model.betas[0][0]  # 空间滞后系数 (Spatial Lag Coefficient)
        beta_x = model.betas[1:-1, 0] # 自变量系数
        
        print(f"\n--- Spatial Spillover Decomposition ---")
        print(f"Spatial Lag Coefficient (ρ): {rho:.4f}")
        print(f"Interpretation: A 1% increase in activity in Zone A increases revenue in adjacent zones by {rho*100:.2f}%")
        
        # 计算直接效应 (Direct Effect) 和 间接/溢出效应 (Indirect/Spillover Effect)
        # 公式: 总效应 = (I - ρW)^-1 * β
        # 这里使用 LeSage & Elhorst (2011) 的迹方法近似计算
        n = len(self.y)
        I_mat = np.eye(n)
        S = np.linalg.inv(I_mat - rho * self.w.full()[0])
        
        direct_effect = np.trace(S) / n * beta_x
        indirect_effect = (np.sum(S) / n - np.trace(S) / n) * beta_x
        total_effect = (np.sum(S) / n) * beta_x
        
        effects_df = pd.DataFrame({
            'Variable': self.x_cols,
            'Direct_Effect': direct_effect,
            'Indirect_Effect_(Spillover)': indirect_effect,
            'Total_Effect': total_effect
        })
        print(effects_df)
        return model, effects_df

    def plot_lisa_clusters(self):
        """绘制 LISA 聚类图 (High-High, Low-Low, etc.)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        # 仅绘制显著的聚类 (p < 0.05)
        sig_mask = self.gdf['lisa_pval'] < 0.05
        self.gdf[~sig_mask].plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)
        
        # 颜色映射: 1=HH(红), 2=LH(粉), 3=LL(蓝), 4=HL(浅蓝)
        colors = {1: 'red', 2: 'lightsalmon', 3: 'blue', 4: 'lightblue'}
        for q_val, color in colors.items():
            subset = self.gdf[(self.gdf['lisa_cluster'] == q_val) & sig_mask]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5, label=f'Cluster {q_val}')
                
        plt.title('LISA Cluster Map of Merchant Revenue (Significant at p < 0.05)')
        plt.legend()
        plt.axis('off')
        plt.savefig('FigS2_LISA_Cluster_Map.png', dpi=600, bbox_inches='tight')
        plt.show()

# 使用示例
# pipeline = SpatialEconometricsPipeline(gdf_study_area)
# pipeline.build_spatial_weights()
# pipeline.run_morans_i_diagnostics()
# sar_model, effects = pipeline.fit_sar_model_and_decompose()
# pipeline.plot_lisa_clusters()