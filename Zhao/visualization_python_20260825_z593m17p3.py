import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置期刊风格
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

def plot_performance_comparison(data_dir='data', output_dir='figures'):
    """
    复现 Figure/Table 2: 各方法的性能对比条形图。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(os.path.join(data_dir, 'performance_comparison.csv'))
    
    # 计算均值和标准差
    summary = df.groupby('method').agg({
        'occupancy_utilization': ['mean', 'std'],
        'energy_efficiency': ['mean', 'std'],
        'user_satisfaction': ['mean', 'std']
    }).reset_index()
    
    # 扁平化列名
    summary.columns = ['method', 'occ_mean', 'occ_std', 'eff_mean', 'eff_std', 'sat_mean', 'sat_std']
    
    # 排序以便展示
    method_order = ['Proposed MARL-GNN', 'QGNN', 'MAPPO', 'Single-Agent PPO', 'Integer Programming', 'Greedy Heuristic']
    summary['method'] = pd.Categorical(summary['method'], categories=method_order, ordered=True)
    summary = summary.sort_values('method')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('occ_mean', 'occ_std', 'Occupancy Utilization', axes[0]),
        ('eff_mean', 'eff_std', 'Energy Efficiency', axes[1]),
        ('sat_mean', 'sat_std', 'User Satisfaction', axes[2])
    ]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#95a5a6']
    
    for i, (mean_col, std_col, title, ax) in enumerate(metrics):
        bars = ax.bar(summary['method'], summary[mean_col], yerr=summary[std_col], capsize=5, color=colors, edgecolor='black')
        ax.set_title(title)
        ax.set_xticklabels(summary['method'], rotation=45, ha='right')
        ax.set_ylim(0, 1.1 if i < 2 else 5.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_performance_comparison.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_ablation_recovery(data_dir='data', output_dir='figures'):
    """
    复现 Table 5/6: 消融实验恢复时间箱线图。
    """
    df = pd.read_csv(os.path.join(data_dir, 'ablation_recovery.csv'))
    
    # 排序
    config_order = ['Full Framework', 'w/o ViT', 'w/o Meta-Learning', 'w/o GNN', 'w/o PER']
    df['configuration'] = pd.Categorical(df['configuration'], categories=config_order, ordered=True)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='configuration', y='recovery_time_steps', data=df, palette='viridis')
    plt.title('Ablation Study: Recovery Time under Equipment Failure')
    plt.xlabel('Configuration')
    plt.ylabel('Recovery Time (Time Steps)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_ablation_recovery.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_meta_learning_weights(data_dir='data', output_dir='figures'):
    """
    复现 Figure 4: 元学习自适应奖励权重的动态演化。
    """
    df = pd.read_csv(os.path.join(data_dir, 'meta_learning_weights.csv'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['time_step'], df['w_occupancy'], label='w1: Occupancy Utilization', linewidth=2)
    plt.plot(df['time_step'], df['w_energy'], label='w2: Energy Efficiency', linewidth=2)
    plt.plot(df['time_step'], df['w_satisfaction'], label='w3: User Satisfaction', linewidth=2)
    
    # 标注 Peak Hours
    plt.axvspan(40, 56, color='yellow', alpha=0.1, label='Peak Hours (Approx)')
    plt.axvspan(72, 88, color='yellow', alpha=0.1)
    
    plt.title('Dynamic Evolution of Meta-Learning Adaptive Reward Weights')
    plt.xlabel('Time Step (15-min intervals)')
    plt.ylabel('Adaptive Weight')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_meta_learning_weights.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def plot_resilience_trajectory(data_dir='data', output_dir='figures'):
    """
    复现 Figure 5: 设备故障下的韧性与恢复轨迹。
    """
    df = pd.read_csv(os.path.join(data_dir, 'resilience_trajectory.csv'))
    
    plt.figure(figsize=(12, 6))
    
    # 定义颜色和线型
    styles = {
        'Proposed MARL-GNN': {'color': '#2ecc71', 'linestyle': '-', 'linewidth': 2},
        'Single-Agent PPO': {'color': '#e67e22', 'linestyle': '--', 'linewidth': 2},
        'Integer Programming': {'color': '#e74c3c', 'linestyle': ':', 'linewidth': 2}
    }
    
    for method, style in styles.items():
        subset = df[df['method'] == method]
        plt.plot(subset['time_step'], subset['occupancy_utilization'], 
                 label=method, color=style['color'], linestyle=style['linestyle'], linewidth=style['linewidth'])
        
    # 标注故障发生点
    plt.axvline(x=31, color='red', linestyle='-', alpha=0.5, label='Equipment Failure Event')
    
    plt.title('Resilience and Recovery Trajectories under Equipment Failure')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Occupancy Utilization')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_resilience_trajectory.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_performance_comparison()
    plot_ablation_recovery()
    plot_meta_learning_weights()
    plot_resilience_trajectory()