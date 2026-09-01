import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置期刊排版风格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

def compute_dynamic_adjacency_matrix():
    """
    复现论文 Eq.4: A_ij(t) = exp(-d_ij / delta) + alpha * Flow_ij(t)
    生成并可视化静态基础邻接矩阵与动态时变邻接矩阵的对比。
    """
    # 1. 设施定义与物理距离矩阵 (Section 5.1)
    facilities = ['Swimming Pool', 'Basketball Court', 'Fitness Center', 
                  'Squash Court', 'Climbing Wall', 'Multipurpose Hall']
    n_facilities = len(facilities)
    short_names = [f.split()[0] for f in facilities] # 用于图表标签简化

    # 物理距离矩阵 D (单位: 米)
    # 注意: Squash Court 和 Swimming Pool 物理距离极近 (共享 HVAC loop)
    distance_matrix = np.array([
        [0,  50, 40, 10, 80, 60],  # Pool
        [50, 0,  30, 60, 20, 40],  # Basketball
        [40, 30, 0,  50, 70, 30],  # Fitness
        [10, 60, 50, 0,  90, 70],  # Squash (Close to Pool)
        [80, 20, 70, 90, 0,  50],  # Climbing
        [60, 40, 30, 70, 50, 0]    # Multipurpose
    ])

    # 2. 公式 (4) 参数设置
    delta = 30.0  # 距离衰减阈值 (proximity threshold)
    alpha = 0.05  # 用户流动权重系数

    def get_static_adjacency(D, delta):
        """计算仅基于物理距离的静态基础邻接矩阵 (Eq.4 第一项)"""
        A_static = np.exp(-D / delta)
        np.fill_diagonal(A_static, 0) # 移除自环
        return A_static

    def simulate_user_flows(t, D):
        """
        模拟时间步 t 的实时用户流动矩阵 Flow_ij(t)
        基于非齐次泊松过程和设施间的转移概率 (Section 4.1 & 5.1)
        """
        np.random.seed(t) # 保证同一时间步可复现
        flow_matrix = np.zeros((n_facilities, n_facilities))
        
        hour = (t % 96) / 4 # 将15分钟步长转换为小时 (0-24)
        
        # 设施间转移概率矩阵 (基于功能关联性与共享基础设施)
        # 例如: Pool <-> Squash 转移率高 (共享HVAC/用户习惯)
        transfer_prob = np.array([
            [0.00, 0.05, 0.10, 0.25, 0.02, 0.05], # Pool -> others
            [0.05, 0.00, 0.15, 0.02, 0.10, 0.10], # Basketball
            [0.15, 0.10, 0.00, 0.05, 0.05, 0.10], # Fitness
            [0.20, 0.02, 0.05, 0.00, 0.01, 0.05], # Squash -> others
            [0.02, 0.10, 0.05, 0.01, 0.00, 0.15], # Climbing
            [0.05, 0.10, 0.10, 0.05, 0.15, 0.00]  # Multipurpose
        ])
        
        # 模拟各设施当前人数 (基于非齐次泊松分布的期望)
        expected_occupancy = np.array([50, 30, 60, 10, 15, 80]) 
        if (10 <= hour <= 14) or (18 <= hour <= 21): # 高峰期放大
            expected_occupancy *= 1.5
            
        current_occupancy = np.random.poisson(expected_occupancy)
        
        # 计算流动量 Flow_ij(t)
        for i in range(n_facilities):
            for j in range(n_facilities):
                if i != j:
                    # 流动量 = 当前人数 * 转移概率 * 随机环境扰动
                    flow_matrix[i, j] = current_occupancy[i] * transfer_prob[i, j] * np.random.uniform(0.8, 1.2)
                    
        return flow_matrix

    def get_dynamic_adjacency(t, D, delta, alpha):
        """计算 Eq.4 的完整动态邻接矩阵 A_ij(t)"""
        A_static = get_static_adjacency(D, delta)
        Flow_t = simulate_user_flows(t, D)
        
        A_dynamic = A_static + alpha * Flow_t
        
        # 归一化到 [0, 1] 以便于GNN输入和可视化
        max_val = np.max(A_dynamic)
        A_dynamic = A_dynamic / max_val if max_val > 0 else A_dynamic
        return A_dynamic

    # 3. 生成不同场景下的矩阵
    t_off_peak = 24  # 06:00 AM (非高峰期)
    t_peak = 72      # 06:00 PM (晚高峰期)

    A_static = get_static_adjacency(distance_matrix, delta)
    A_off_peak = get_dynamic_adjacency(t_off_peak, distance_matrix, delta, alpha)
    A_peak = get_dynamic_adjacency(t_peak, distance_matrix, delta, alpha)

    # 归一化静态矩阵以便对比
    A_static_norm = A_static / np.max(A_static) 

    # 4. 可视化对比 (输出高清 PDF)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin, vmax = 0, 1
    cbar_kws = {'label': 'Edge Weight $A_{ij}(t)$'}

    # 图1: 静态基础邻接矩阵
    sns.heatmap(A_static_norm, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0], 
                vmin=vmin, vmax=vmax, cbar_kws=cbar_kws,
                xticklabels=short_names, yticklabels=short_names)
    axes[0].set_title("(a) Static Adjacency\n(Physical Proximity Only, Eq.4 Base Term)")

    # 图2: 非高峰期动态邻接矩阵
    sns.heatmap(A_off_peak, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1], 
                vmin=vmin, vmax=vmax, cbar_kws=cbar_kws,
                xticklabels=short_names, yticklabels=short_names)
    axes[1].set_title(f"(b) Dynamic Adjacency at Off-Peak\n(t={t_off_peak}, 06:00)")

    # 图3: 高峰期动态邻接矩阵
    sns.heatmap(A_peak, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[2], 
                vmin=vmin, vmax=vmax, cbar_kws=cbar_kws,
                xticklabels=short_names, yticklabels=short_names)
    axes[2].set_title(f"(c) Dynamic Adjacency at Peak\n(t={t_peak}, 18:00)")

    plt.tight_layout()
    
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'eq4_dynamic_adjacency_matrix.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    # 打印关键发现 (对应论文 Section 6.1 中关于 Pool 和 Squash 耦合的讨论)
    pool_idx = facilities.index('Swimming Pool')
    squash_idx = facilities.index('Squash Court')
    print(f"--- Key Topological Insights (Section 6.1) ---")
    print(f"Static Weight (Pool <-> Squash): {A_static_norm[pool_idx, squash_idx]:.3f}")
    print(f"Peak Dynamic Weight (Pool <-> Squash): {A_peak[pool_idx, squash_idx]:.3f} (Amplified by user flow)")
    print(f"Off-Peak Dynamic Weight (Pool <-> Squash): {A_off_peak[pool_idx, squash_idx]:.3f}")

if __name__ == "__main__":
    compute_dynamic_adjacency_matrix()