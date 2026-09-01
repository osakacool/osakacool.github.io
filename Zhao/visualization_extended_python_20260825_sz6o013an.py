import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

# 期刊排版风格设置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (10, 6)
})

output_dir = 'figures'
data_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plot_spatiotemporal_occupancy():
    """
    复现 Figure 2: 优化调度下的时空占用分布 (热力图)
    """
    df = pd.read_csv(os.path.join(data_dir, 'spatiotemporal_occupancy.csv'))
    df.set_index('time_step', inplace=True)
    
    # 将 time_step 转换为小时 (0-24)
    df.index = df.index * 15 / 60 
    
    plt.figure(figsize=(12, 5))
    # 使用 seaborn 绘制热力图
    sns.heatmap(df.T, cmap='YlGnBu', cbar_kws={'label': 'Occupancy Rate'}, 
                xticklabels=4, yticklabels=1)
    
    plt.title('Figure 2. Spatial-temporal Occupancy Distribution under Optimized Scheduling')
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Facility Type')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_spatiotemporal_occupancy.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_facility_improvements():
    """
    复现 Table 4: 异构设施类型的性能改进 (分组柱状图)
    """
    df = pd.read_csv(os.path.join(data_dir, 'table4_facility_improvements.csv'))
    
    # 转换为长格式以便 seaborn 绘制
    df_melted = df.melt(id_vars='Facility Type', 
                        value_vars=['Energy Reduction vs IP (%)', 
                                    'Utilization Gain vs Greedy (%)', 
                                    'Conflict Rate Reduction (%)'],
                        var_name='Metric', value_name='Improvement (%)')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Facility Type', y='Improvement (%)', hue='Metric', data=df_melted, palette='Set2')
    
    plt.title('Table 4. Performance Improvements across Heterogeneous Facility Types')
    plt.xlabel('Facility Type')
    plt.ylabel('Improvement (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'table4_facility_improvements.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_scalability_analysis():
    """
    复现 Table 6: 可扩展性分析 (双Y轴折线图)
    """
    df = pd.read_csv(os.path.join(data_dir, 'table6_scalability.csv'))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Number of Facilities')
    ax1.set_ylabel('Decision Latency (ms)', color=color1)
    ax1.plot(df['Node Count'], df['Latency_Proposed (ms)'], 'o-', color=color1, label='Proposed MARL-GNN')
    ax1.plot(df['Node Count'], df['Latency_MAPPO (ms)'], 's--', color=color1, alpha=0.6, label='MAPPO')
    ax1.plot(df['Node Count'], df['Latency_SingleAgent (ms)'], '^:', color=color1, alpha=0.4, label='Single-Agent PPO')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(df['Node Count'])
    
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Occupancy Utilization', color=color2)  
    ax2.plot(df['Node Count'], df['Utilization_Proposed'], 'o-', color=color2, label='Utilization (Proposed)')
    ax2.plot(df['Node Count'], df['Utilization_MAPPO'], 's--', color=color2, alpha=0.6, label='Utilization (MAPPO)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Table 6. Scalability Analysis across Varying Facility Configurations')
    fig.tight_layout()
    
    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.savefig(os.path.join(output_dir, 'table6_scalability.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_privacy_utility_tradeoff():
    """
    复现 Table 7: 隐私-效用权衡曲线
    """
    df = pd.read_csv(os.path.join(data_dir, 'table7_privacy_utility.csv'))
    # 过滤掉 inf 以便在对数/线性坐标中更好展示，或单独处理
    df_plot = df[df['Privacy Budget (eps)'] != 'inf'].copy()
    df_plot['eps_val'] = df_plot['Privacy Budget (eps)'].astype(float)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_plot['eps_val'], df_plot['Global Reward Convergence (%)'], 'o-', label='Global Reward Convergence (%)', color='green')
    plt.plot(df_plot['eps_val'], df_plot['Accuracy Degradation (%)'], 's-', label='Accuracy Degradation (%)', color='red')
    
    plt.title('Table 7. Privacy-Utility Trade-off under Differential Privacy Constraints')
    plt.xlabel('Privacy Budget ($\epsilon$)')
    plt.ylabel('Percentage (%)')
    plt.xscale('log', base=2)
    plt.xticks([1, 2, 4, 8], ['1', '2', '4', '8'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'table7_privacy_utility.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_gnn_latent_space():
    """
    复现 Figure 3: GNN学习的设施潜在空间表示 (t-SNE 降维散点图)
    """
    df = pd.read_csv(os.path.join(data_dir, 'fig3_gnn_embeddings.csv'))
    
    features = df.drop(columns=['Facility']).values
    labels = df['Facility'].values
    
    # 执行 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    facilities = np.unique(labels)
    colors = sns.color_palette("husl", len(facilities))
    
    for i, fac in enumerate(facilities):
        mask = labels == fac
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                    c=[colors[i]], label=fac, alpha=0.7, edgecolors='w', s=60)
        
    plt.title('Figure 3. Latent Space Representation of Facilities Learned by the GNN (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_latent_space.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_training_convergence():
    """
    复现 Figure 6: 三阶段训练协议下的训练收敛和样本效率
    """
    df = pd.read_csv(os.path.join(data_dir, 'fig6_training_convergence.csv'))
    
    plt.figure(figsize=(12, 6))
    
    # 绘制三阶段背景阴影
    plt.axvspan(0, 50, color='gray', alpha=0.1, label='Warm-up (0-50)')
    plt.axvspan(50, 350, color='blue', alpha=0.05, label='Coordinated Learning (50-350)')
    plt.axvspan(350, 400, color='red', alpha=0.05, label='Fine-tuning (350-400)')
    
    methods = ['Proposed', 'MAPPO', 'SingleAgent']
    colors = ['#2ecc71', '#3498db', '#e67e22']
    labels = ['Proposed MARL-GNN', 'MAPPO', 'Single-Agent PPO']
    
    for i, method in enumerate(methods):
        mean_col = f'Reward_{method}'
        var_col = f'Reward_{method}_Var'
        
        # 绘制方差阴影
        plt.fill_between(df['episode'], 
                         df[mean_col] - df[var_col], 
                         df[mean_col] + df[var_col], 
                         color=colors[i], alpha=0.2)
        # 绘制均值线
        plt.plot(df['episode'], df[mean_col], color=colors[i], label=labels[i], linewidth=2)
        
    plt.title('Figure 6. Training Convergence and Sample Efficiency under Three-Phase Protocol')
    plt.xlabel('Training Episodes')
    plt.ylabel('Normalized Average Episode Reward')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_training_convergence.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_spatiotemporal_occupancy()
    plot_facility_improvements()
    plot_scalability_analysis()
    plot_privacy_utility_tradeoff()
    plot_gnn_latent_space()
    plot_training_convergence()