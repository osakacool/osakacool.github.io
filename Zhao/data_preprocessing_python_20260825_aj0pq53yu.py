import pandas as pd
import numpy as np
import os

def preprocess_raw_data(raw_dir='raw_data', processed_dir='processed_data'):
    """
    将原始传感器日志聚合为 15分钟粒度的 MARL 状态矩阵与图结构
    对应论文 Section 4.1 和 5.3 的输入定义
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. 加载原始数据
    df_energy = pd.read_csv(os.path.join(raw_dir, 'raw_energy_environment.csv'))
    df_failures = pd.read_csv(os.path.join(raw_dir, 'raw_equipment_failures.csv'))
    
    facilities = ['Swimming Pool', 'Basketball Court', 'Fitness Center', 
                  'Squash Court', 'Climbing Wall', 'Multipurpose Hall']
    
    # 2. 构建状态特征矩阵 (N_nodes x Features)
    # 特征包括: [occupancy_ratio, energy_kwh, equipment_status, external_temp]
    state_records = []
    
    for t in df_energy['timestamp_step'].unique():
        step_data = df_energy[df_energy['timestamp_step'] == t]
        node_features = []
        for f in facilities:
            row = step_data[step_data['facility_id'] == f]
            if not row.empty:
                node_features.append([
                    row['occupancy_ratio'].values[0],
                    row['energy_kwh'].values[0],
                    row['equipment_status'].values[0],
                    row['external_temp'].values[0]
                ])
            else:
                node_features.append([0, 0, 1.0, 20.0]) # 默认值
        
        state_records.append({
            'timestamp_step': t,
            'node_features': np.array(node_features).flatten().tolist()
        })
        
    df_states = pd.DataFrame(state_records)
    df_states.to_csv(os.path.join(processed_dir, 'marl_state_features.csv'), index=False)
    
    # 3. 构建奖励信号 (对应 Eq. 10)
    # R = w1*Utilization + w2*Efficiency + w3*Satisfaction
    reward_records = []
    for t in df_energy['timestamp_step'].unique():
        step_data = df_energy[df_energy['timestamp_step'] == t]
        
        # 计算全局占用率
        avg_occ = step_data['occupancy_ratio'].mean()
        # 计算能耗效率 (假设越低越好，这里做归一化反转)
        total_energy = step_data['energy_kwh'].sum()
        eff = 1.0 / (1.0 + total_energy / 500.0) 
        # 模拟用户满意度 (与拥挤度负相关，与设备状态正相关)
        avg_status = step_data['equipment_status'].mean()
        sat = 5.0 - (avg_occ * 2.0) + (avg_status * 1.5)
        sat = np.clip(sat, 1.0, 5.0)
        
        reward_records.append({
            'timestamp_step': t,
            'utilization': avg_occ,
            'efficiency': eff,
            'satisfaction': sat,
            'global_reward': 0.4*avg_occ + 0.3*eff + 0.3*(sat/5.0)
        })
        
    df_rewards = pd.DataFrame(reward_records)
    df_rewards.to_csv(os.path.join(processed_dir, 'marl_reward_signals.csv'), index=False)
    
    # 4. 提取故障事件标签 (用于 PER 优先级计算)
    failure_labels = np.zeros(len(df_rewards))
    for _, row in df_failures.iterrows():
        t = int(row['timestamp_step'])
        if t < len(failure_labels):
            failure_labels[t] = row['severity']
            
    df_rewards['failure_severity'] = failure_labels
    df_rewards.to_csv(os.path.join(processed_dir, 'marl_reward_with_faults.csv'), index=False)
    
    print(f"Processed MARL states and rewards saved to {processed_dir}/")

if __name__ == "__main__":
    preprocess_raw_data()