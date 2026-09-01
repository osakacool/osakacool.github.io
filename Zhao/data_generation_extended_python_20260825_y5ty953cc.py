import numpy as np
import pandas as pd
import os

def generate_extended_simulated_data(output_dir='data'):
    """
    生成论文中剩余图表和表格所需的模拟数据集。
    真实数据替换方法：
    1. Fig 2 (时空占用): 替换为实际传感器按15分钟粒度采集的各设施占用率CSV。
    2. Fig 3 (潜在空间): 替换为实际GNN模型输出的节点Embedding矩阵 (N_nodes x 128)。
    3. Fig 6 (训练收敛): 替换为实际训练日志中每个episode的平均Reward记录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(42)
    facilities = ['Swimming Pool', 'Basketball Court', 'Fitness Center', 
                  'Squash Court', 'Climbing Wall', 'Multipurpose Hall']

    # ==========================================
    # 1. 生成 Figure 2: 时空占用分布数据 (24小时 x 6设施)
    # ==========================================
    time_steps = np.arange(96) # 96 steps = 24 hours
    spatial_temporal_data = np.zeros((96, 6))
    
    for i, fac in enumerate(facilities):
        # 模拟不同设施的高峰期
        if fac == 'Swimming Pool':
            peak1, peak2 = (10, 14), (18, 21)
        elif fac == 'Basketball Court':
            peak1, peak2 = (16, 20), (19, 22)
        elif fac == 'Fitness Center':
            peak1, peak2 = (6, 9), (17, 21)
        else:
            peak1, peak2 = (12, 14), (18, 20)
            
        for t in time_steps:
            hour = t * 15 / 60
            base = 0.2
            if (peak1[0] <= hour <= peak1[1]) or (peak2[0] <= hour <= peak2[1]):
                base = 0.85
            spatial_temporal_data[t, i] = np.clip(base + np.random.normal(0, 0.05), 0, 1)

    df_spatiotemporal = pd.DataFrame(spatial_temporal_data, columns=facilities)
    df_spatiotemporal['time_step'] = time_steps
    df_spatiotemporal.to_csv(os.path.join(output_dir, 'spatiotemporal_occupancy.csv'), index=False)

    # ==========================================
    # 2. 生成 Table 4: 异构设施类型的性能改进
    # ==========================================
    table4_data = {
        'Facility Type': facilities,
        'Energy Reduction vs IP (%)': [24.5, 15.3, 18.7, 28.2, 11.4, 19.6],
        'Utilization Gain vs Greedy (%)': [18.2, 22.6, 14.5, 12.3, 16.8, 25.4],
        'Conflict Rate Reduction (%)': [85.4, 72.1, 68.9, 88.6, 65.2, 79.5]
    }
    df_table4 = pd.DataFrame(table4_data)
    df_table4.to_csv(os.path.join(output_dir, 'table4_facility_improvements.csv'), index=False)

    # ==========================================
    # 3. 生成 Table 6: 可扩展性分析
    # ==========================================
    table6_data = {
        'Configuration': ['6 Facilities', '12 Facilities', '24 Facilities'],
        'Node Count': [6, 12, 24],
        'Latency_Proposed (ms)': [120, 185, 290],
        'Latency_SingleAgent (ms)': [210, 480, 1150],
        'Latency_MAPPO (ms)': [135, 175, 240],
        'Comm_Overhead (MB)': [14.2, 28.6, 57.3],
        'Utilization_Proposed': [0.82, 0.80, 0.79],
        'Utilization_MAPPO': [0.78, 0.74, 0.69]
    }
    df_table6 = pd.DataFrame(table6_data)
    df_table6.to_csv(os.path.join(output_dir, 'table6_scalability.csv'), index=False)

    # ==========================================
    # 4. 生成 Table 7: 隐私-效用权衡
    # ==========================================
    table7_data = {
        'Privacy Budget (eps)': ['inf', 8, 4, 2, 1],
        'Noise Scale (sigma)': [0, 0.5, 1.2, 2.5, 4.8],
        'Global Reward Convergence (%)': [100.0, 98.6, 96.1, 92.4, 87.5],
        'Accuracy Degradation (%)': [0, 1.4, 3.9, 7.6, 12.5],
        'Comm Overhead (MB)': [14.2, 14.5, 15.1, 16.8, 19.4]
    }
    df_table7 = pd.DataFrame(table7_data)
    df_table7.to_csv(os.path.join(output_dir, 'table7_privacy_utility.csv'), index=False)

    # ==========================================
    # 5. 生成 Figure 3: GNN潜在空间表示 (用于t-SNE)
    # ==========================================
    # 模拟 128 维的 GNN embedding，每个设施生成 50 个时间步的样本
    n_samples_per_fac = 50
    embedding_dim = 128
    embeddings = []
    labels = []
    
    for i, fac in enumerate(facilities):
        # 为每个设施生成一个中心点，并添加高斯噪声
        center = np.random.randn(embedding_dim) * 2 + i * 1.5 
        samples = center + np.random.randn(n_samples_per_fac, embedding_dim) * 0.5
        embeddings.append(samples)
        labels.extend([fac] * n_samples_per_fac)
        
    embeddings = np.vstack(embeddings)
    df_embeddings = pd.DataFrame(embeddings, columns=[f'dim_{j}' for j in range(embedding_dim)])
    df_embeddings['Facility'] = labels
    df_embeddings.to_csv(os.path.join(output_dir, 'fig3_gnn_embeddings.csv'), index=False)

    # ==========================================
    # 6. 生成 Figure 6: 训练收敛曲线
    # ==========================================
    episodes = np.arange(400)
    
    def generate_reward_curve(base_reward, growth_rate, noise_std, seed):
        np.random.seed(seed)
        # 使用对数增长模拟收敛
        reward = base_reward + growth_rate * np.log1p(episodes) + np.random.normal(0, noise_std, 400)
        # 添加阶段特征
        reward[50:350] += 0.5 # Coordinated learning boost
        reward[350:] += 0.8   # Fine-tuning boost
        return np.clip(reward, 0, 10)

    df_training = pd.DataFrame({'episode': episodes})
    # Proposed MARL-GNN
    df_training['Reward_Proposed'] = generate_reward_curve(1.0, 2.5, 0.3, 10)
    df_training['Reward_Proposed_Var'] = np.random.uniform(0.1, 0.4, 400)
    # MAPPO
    df_training['Reward_MAPPO'] = generate_reward_curve(0.8, 2.0, 0.4, 20)
    df_training['Reward_MAPPO_Var'] = np.random.uniform(0.2, 0.6, 400)
    # Single-Agent PPO
    df_training['Reward_SingleAgent'] = generate_reward_curve(0.5, 1.5, 0.5, 30)
    df_training['Reward_SingleAgent_Var'] = np.random.uniform(0.3, 0.8, 400)
    
    df_training.to_csv(os.path.join(output_dir, 'fig6_training_convergence.csv'), index=False)
    print("Extended simulated data generated successfully.")

if __name__ == "__main__":
    generate_extended_simulated_data()