import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 全局学术绘图设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# Table 4 & 5: 消融实验与模型性能对比 (分组柱状图)
# ==========================================
def plot_ablation_and_performance():
    # Table 4 数据 (Ablation Study)
    data_table4 = {
        'Input Features': ['VIIRS only', 'VIIRS+POI', 'VIIRS+Mobility', 'VIIRS+Social', 'Full Model'],
        'Overall Accuracy': [85.2, 89.4, 88.3, 86.7, 92.3],
        'Kappa': [0.79, 0.84, 0.82, 0.80, 0.89],
        'mIoU': [0.73, 0.78, 0.76, 0.74, 0.85]
    }
    df4 = pd.DataFrame(data_table4)
    
    # Table 5 数据 (Model Comparison)
    data_table5 = {
        'Model': ['DeepLabV3+ (Ours)', 'FCN', 'Random Forest'],
        'Overall Accuracy': [92.3, 86.5, 79.2],
        'Kappa': [0.89, 0.81, 0.73],
        'mIoU': [0.85, 0.76, 0.68]
    }
    df5 = pd.DataFrame(data_table5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制 Table 4 (Ablation)
    df4.plot(x='Input Features', y=['Overall Accuracy', 'Kappa', 'mIoU'], kind='bar', ax=axes[0],
             colormap='viridis', edgecolor='black', linewidth=1.2)
    axes[0].set_title('Table 4: Ablation Study (Data Sources)', fontweight='bold')
    axes[0].set_ylabel('Score / Percentage')
    axes[0].set_ylim(0, 100)
    axes[0].legend(title='Metrics')
    axes[0].tick_params(axis='x', rotation=15)
    
    # 绘制 Table 5 (Model Comparison)
    df5.plot(x='Model', y=['Overall Accuracy', 'Kappa', 'mIoU'], kind='bar', ax=axes[1],
             colormap='plasma', edgecolor='black', linewidth=1.2)
    axes[1].set_title('Table 5: Model Performance Comparison', fontweight='bold')
    axes[1].set_ylabel('Score / Percentage')
    axes[1].set_ylim(0, 100)
    axes[1].legend(title='Metrics')
    
    plt.tight_layout()
    plt.savefig('Fig_Table4_Table5_Performance_Comparison.png', dpi=600, bbox_inches='tight')
    plt.show()

# ==========================================
# Table 6: 混淆矩阵热力图 (Confusion Matrix)
# ==========================================
def plot_confusion_matrix():
    # 论文 Table 6 真实数据 (百分比)
    cm_data = np.array([
        [94, 3, 1, 1, 1],
        [4, 83, 8, 3, 2],
        [2, 10, 79, 5, 4],
        [1, 4, 3, 86, 6],
        [0, 1, 2, 3, 94]
    ])
    labels = ['Core Consumption', 'Dining Cluster', 'Cultural Performance', 'Transport Hub', 'Non-Tourism']
    
    plt.figure(figsize=(8, 6))
    # 使用 seaborn 绘制热力图
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray', cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('Table 6: Confusion Matrix of Semantic Segmentation (%)', fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Actual Label', fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Fig_Table6_Confusion_Matrix.png', dpi=600, bbox_inches='tight')
    plt.show()

# ==========================================
# Table 8: 经济量化模型对比 (带误差棒的柱状图)
# ==========================================
def plot_economic_models():
    models = ['Linear (OLS)', 'Gradient Boosting', 'SAR (Spatial)']
    r2_scores = [0.76, 0.79, 0.87]
    rmse = [12.4, 11.8, 9.8]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35
    
    # R2 柱状图
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R²', color='#4C72B0', edgecolor='black')
    ax1.set_ylabel('R² Score', color='#4C72B0', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    
    # RMSE 折线图 (双Y轴)
    ax2 = ax1.twinx()
    ax2.plot(x, rmse, 'o-', color='#DD8452', linewidth=2, markersize=8, label='RMSE')
    ax2.set_ylabel('RMSE', color='#DD8452', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#DD8452')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title('Table 8: Economic Output Estimation Models Comparison', fontweight='bold')
    
    # 添加图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Fig_Table8_Economic_Models.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_ablation_and_performance()
    plot_confusion_matrix()
    plot_economic_models()