import numpy as np
import pandas as pd
import os

def generate_simulated_data(output_dir='data'):
    """
    生成模拟数据集以复现论文中的实验结果。
    真实数据替换方法：将生成的 CSV 文件替换为实际传感器日志和调度记录，
    确保列名与 DataFrame 结构一致。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置随机种子以保证可复现性
    np.random.seed(42)

    # 1. 设施基础信息 (6种设施类型)
    facilities = ['Swimming Pool', 'Basketball Court', 'Fitness Center', 
                  'Squash Court', 'Climbing Wall', 'Multipurpose Hall']
    
    # 2. 生成时间序列数据 (400 episodes, 每个 episode 96 steps, 15分钟间隔)
    # 为了简化演示，我们生成聚合后的评估指标数据，而非原始的每一步传感器数据
    # 原始数据通常包含: timestamp, facility_id, occupancy, energy_kwh, user_rating
    
    n_eval_steps = 100 # 对应论文中 N=100 的评估步数
    methods = ['Proposed MARL-GNN', 'QGNN', 'MAPPO', 'Single-Agent PPO', 'Integer Programming', 'Greedy Heuristic']
    
    # 表2: 性能对比数据 (均值 ± 标准差)
    # 这里我们生成符合这些分布的原始评分数据以便进行 t-test
    data_records = []
    
    for method in methods:
        for i in range(n_eval_steps):
            # 根据论文 Table 2 的均值和方差生成模拟观测值
            if method == 'Proposed MARL-GNN':
                occ = np.random.normal(0.82, 0.03)
                eff = np.random.normal(0.91, 0.02)
                sat = np.random.normal(4.3, 0.2)
                robust = 0.97 + np.random.normal(0, 0.01)
            elif method == 'QGNN':
                occ = np.random.normal(0.79, 0.04)
                eff = np.random.normal(0.89, 0.03)
                sat = np.random.normal(4.1, 0.3)
                robust = 0.95 + np.random.normal(0, 0.01)
            elif method == 'MAPPO':
                occ = np.random.normal(0.78, 0.04)
                eff = np.random.normal(0.88, 0.03)
                sat = np.random.normal(4.1, 0.3)
                robust = 0.94 + np.random.normal(0, 0.01)
            elif method == 'Single-Agent PPO':
                occ = np.random.normal(0.76, 0.05)
                eff = np.random.normal(0.87, 0.03)
                sat = np.random.normal(4.0, 0.3)
                robust = 0.92 + np.random.normal(0, 0.01)
            elif method == 'Integer Programming':
                occ = np.random.normal(0.79, 0.04)
                eff = np.random.normal(0.85, 0.04)
                sat = np.random.normal(3.8, 0.4)
                robust = 0.89 + np.random.normal(0, 0.01)
            else: # Greedy
                occ = np.random.normal(0.68, 0.07)
                eff = np.random.normal(0.78, 0.06)
                sat = np.random.normal(3.5, 0.5)
                robust = 0.82 + np.random.normal(0, 0.01)
            
            data_records.append({
                'method': method,
                'step_id': i,
                'occupancy_utilization': np.clip(occ, 0, 1),
                'energy_efficiency': np.clip(eff, 0, 1),
                'user_satisfaction': np.clip(sat, 1, 5),
                'robustness_score': np.clip(robust, 0, 1)
            })

    df_performance = pd.DataFrame(data_records)
    df_performance.to_csv(os.path.join(output_dir, 'performance_comparison.csv'), index=False)
    print(f"Generated performance data: {df_performance.shape}")

    # 3. 生成消融实验数据 (Table 5)
    ablation_configs = ['Full Framework', 'w/o GNN', 'w/o ViT', 'w/o Meta-Learning', 'w/o PER']
    ablation_records = []
    for config in ablation_configs:
        for i in range(n_eval_steps):
            if config == 'Full Framework':
                rec_time = np.random.normal(8, 1.5) # 8 steps
            elif config == 'w/o GNN':
                rec_time = np.random.normal(14.5, 2.0)
            elif config == 'w/o ViT':
                rec_time = np.random.normal(9.5, 1.8)
            elif config == 'w/o Meta-Learning':
                rec_time = np.random.normal(11.2, 1.9)
            else: # w/o PER
                rec_time = np.random.normal(22, 3.0)
            
            ablation_records.append({
                'configuration': config,
                'step_id': i,
                'recovery_time_steps': max(1, rec_time)
            })
            
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(os.path.join(output_dir, 'ablation_recovery.csv'), index=False)
    print(f"Generated ablation data: {df_ablation.shape}")

    # 4. 生成元学习权重演化数据 (Figure 4)
    # 24小时周期, 96 steps
    time_steps = np.arange(96)
    # 模拟权重变化: Peak hours (e.g., 10-14, 18-22) w1/w3 high, Off-peak w2 high
    w1 = [] # Occupancy
    w2 = [] # Energy
    w3 = [] # Satisfaction
    
    for t in time_steps:
        hour = t * 15 / 60 # 转换为小时
        if (10 <= hour <= 14) or (18 <= hour <= 22):
            # Peak
            w1.append(np.random.normal(0.5, 0.05))
            w3.append(np.random.normal(0.4, 0.05))
            w2.append(np.random.normal(0.1, 0.02))
        else:
            # Off-peak
            w1.append(np.random.normal(0.2, 0.05))
            w3.append(np.random.normal(0.2, 0.05))
            w2.append(np.random.normal(0.6, 0.05))
            
    df_weights = pd.DataFrame({
        'time_step': time_steps,
        'w_occupancy': w1,
        'w_energy': w2,
        'w_satisfaction': w3
    })
    df_weights.to_csv(os.path.join(output_dir, 'meta_learning_weights.csv'), index=False)
    print(f"Generated meta-learning weights data: {df_weights.shape}")

    # 5. 生成韧性恢复轨迹数据 (Figure 5)
    # 假设 disruption 发生在 step 31
    disaster_step = 31
    recovery_steps = 100
    timeline = np.arange(disaster_step - 10, disaster_step + recovery_steps)
    
    def generate_trajectory(method_name, drop_pct, recovery_duration):
        base_val = 0.82 if 'MARL' in method_name else 0.76
        values = []
        for t in timeline:
            if t < disaster_step:
                values.append(base_val + np.random.normal(0, 0.01))
            elif t <= disaster_step + recovery_duration:
                # Linear recovery
                progress = (t - disaster_step) / recovery_duration
                current_val = base_val * (1 - drop_pct) + (base_val * drop_pct) * progress
                values.append(current_val + np.random.normal(0, 0.01))
            else:
                values.append(base_val + np.random.normal(0, 0.01))
        return values

    df_resilience = pd.DataFrame({
        'time_step': np.tile(timeline, 3),
        'method': ['Proposed MARL-GNN'] * len(timeline) + 
                  ['Single-Agent PPO'] * len(timeline) + 
                  ['Integer Programming'] * len(timeline),
        'occupancy_utilization': generate_trajectory('Proposed MARL-GNN', 0.126, 8) +
                                 generate_trajectory('Single-Agent PPO', 0.342, 45) +
                                 generate_trajectory('Integer Programming', 0.415, 100) # IP doesn't recover well
    })
    # Clip IP to show static failure after drop
    ip_mask = df_resilience['method'] == 'Integer Programming'
    ip_indices = df_resilience[ip_mask].index
    drop_point_idx = list(ip_indices).index(ip_indices[0]) + disaster_step - (disaster_step - 10)
    df_resilience.loc[ip_indices[drop_point_idx:], 'occupancy_utilization'] = 0.82 * (1 - 0.415)

    df_resilience.to_csv(os.path.join(output_dir, 'resilience_trajectory.csv'), index=False)
    print(f"Generated resilience trajectory data: {df_resilience.shape}")

if __name__ == "__main__":
    generate_simulated_data()