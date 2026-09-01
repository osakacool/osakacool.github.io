import numpy as np
import pandas as pd
from scipy.stats import weibull_min
import os

class SportsFacilitySimulator:
    """
    论文 Section 5.1 原始环境模拟器
    生成 15分钟粒度的底层原始数据，包括：
    1. 非齐次泊松分布的用户到达 (User arrival processes)
    2. 基于 Eq.14 的能耗模型 (Energy consumption models)
    3. Weibull 分布的设备故障 (Equipment failure probabilities)
    4. 基于 Eq.4 的动态邻接矩阵 (Dynamic adjacency matrix)
    """
    def __init__(self, output_dir='raw_data', seed=42):
        np.random.seed(seed)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 设施基础参数定义 (对应 Section 5.1 的 6 种设施)
        self.facilities = {
            'Swimming Pool':     {'capacity': 150, 'base_load': 45.0, 'hvac_factor': 1.2, 'shared_infra': ['HVAC', 'Water Heating']},
            'Basketball Court':  {'capacity': 100, 'base_load': 15.0, 'hvac_factor': 0.8, 'shared_infra': ['Lighting', 'HVAC']},
            'Fitness Center':    {'capacity': 120, 'base_load': 30.0, 'hvac_factor': 1.0, 'shared_infra': ['HVAC', 'Ventilation']},
            'Squash Court':      {'capacity': 20,  'base_load': 10.0, 'hvac_factor': 1.5, 'shared_infra': ['HVAC']}, # 与 Pool 共享 HVAC
            'Climbing Wall':     {'capacity': 40,  'base_load': 12.0, 'hvac_factor': 0.6, 'shared_infra': ['Lighting', 'Safety']},
            'Multipurpose Hall': {'capacity': 200, 'base_load': 50.0, 'hvac_factor': 1.1, 'shared_infra': ['HVAC', 'Lighting']}
        }
        self.facility_names = list(self.facilities.keys())
        self.n_facilities = len(self.facility_names)
        
        # 2. 物理距离矩阵 (单位: 米, 用于 Eq.4 的基础权重)
        # 假设 Squash Court 和 Swimming Pool 物理距离极近 (共享 HVAC loop)
        self.distance_matrix = np.array([
            [0,  50, 40, 10, 80, 60],  # Pool
            [50, 0,  30, 60, 20, 40],  # Basketball
            [40, 30, 0,  50, 70, 30],  # Fitness
            [10, 60, 50, 0,  90, 70],  # Squash (Close to Pool)
            [80, 20, 70, 90, 0,  50],  # Climbing
            [60, 40, 30, 70, 50, 0]    # Multipurpose
        ])
        
        # 3. Weibull 故障分布参数 (形状参数 k, 尺度参数 lambda)
        # 不同设施的故障率不同，例如 HVAC 相关的设施故障率略高
        self.weibull_params = {
            'Swimming Pool': (2.5, 1500), 'Basketball Court': (3.0, 2000),
            'Fitness Center': (2.8, 1800), 'Squash Court': (2.2, 1200),
            'Climbing Wall': (3.5, 2500), 'Multipurpose Hall': (2.6, 1600)
        }

    def _generate_weather(self, n_steps):
        """生成外部温度序列 (对应 Eq.14 中的 weather-dependent HVAC demands)"""
        hours = np.arange(n_steps) * 0.25 # 15-min intervals to hours
        # 模拟昼夜温差 (正弦波) + 随机天气扰动
        base_temp = 20 + 8 * np.sin((hours - 6) * np.pi / 12) 
        noise = np.random.normal(0, 1.5, n_steps)
        return np.clip(base_temp + noise, 5, 35)

    def _get_arrival_rate(self, hour, facility_name):
        """
        计算非齐次泊松过程的瞬时到达率 lambda(t)
        根据设施类型和一天中的时间动态调整
        """
        # 基础到达率
        base_rate = 2.0 
        if facility_name == 'Swimming Pool':
            if 10 <= hour <= 14 or 18 <= hour <= 21: base_rate = 15.0
            else: base_rate = 3.0
        elif facility_name == 'Basketball Court':
            if 16 <= hour <= 22: base_rate = 12.0
            else: base_rate = 2.0
        elif facility_name == 'Fitness Center':
            if 6 <= hour <= 9 or 17 <= hour <= 21: base_rate = 18.0
            else: base_rate = 4.0
        elif facility_name == 'Squash Court':
            if 18 <= hour <= 22: base_rate = 8.0
            else: base_rate = 1.5
        elif facility_name == 'Climbing Wall':
            if 14 <= hour <= 20: base_rate = 6.0
            else: base_rate = 1.0
        else: # Multipurpose Hall
            if 17 <= hour <= 21: base_rate = 20.0
            else: base_rate = 5.0
        return base_rate

    def simulate(self, n_days=10, inject_failure=True):
        """
        运行完整模拟，生成原始数据集
        :param n_days: 模拟天数 (论文中为 400 days，此处为演示生成 10 天)
        :param inject_failure: 是否注入 HVAC 故障 (对应 Figure 5 / Table 8)
        """
        steps_per_day = 96
        total_steps = n_days * steps_per_day
        
        # 初始化存储列表
        records_arrivals = []
        records_energy = []
        records_failures = []
        
        # 状态初始化
        occupancy = {f: 0 for f in self.facility_names}
        equipment_status = {f: 1.0 for f in self.facility_names} # 1.0 = normal, <1.0 = degraded
        weather = self._generate_weather(total_steps)
        
        # 故障注入时间点 (对应 Figure 5: step 31 注入故障)
        failure_step = 31 if inject_failure else -1

        print(f"Starting simulation for {n_days} days ({total_steps} steps)...")
        
        for t in range(total_steps):
            hour = (t % 96) * 0.25
            temp_ext = weather[t]
            
            # 记录用户流动 (用于构建动态邻接矩阵 Eq.4)
            user_flows = np.zeros((self.n_facilities, self.n_facilities))
            
            for i, f_name in enumerate(self.facility_names):
                params = self.facilities[f_name]
                
                # 1. 非齐次泊松到达 (User arrivals)
                lam = self._get_arrival_rate(hour, f_name) * equipment_status[f_name]
                arrivals = np.random.poisson(lam)
                departures = int(occupancy[f_name] * np.random.uniform(0.1, 0.3)) # 随机离开
                
                # 更新占用率
                occupancy[f_name] = max(0, min(params['capacity'], occupancy[f_name] + arrivals - departures))
                
                # 模拟用户在不同设施间的流动 (用于 Eq.4 的 Flow_ij)
                if arrivals > 5:
                    # 假设部分用户从 Pool 流向 Squash (共享 HVAC)
                    if f_name == 'Swimming Pool' and i < self.n_facilities:
                        flow_to_squash = int(arrivals * 0.15)
                        user_flows[i, self.facility_names.index('Squash Court')] = flow_to_squash
                
                # 2. 能耗计算 (Eq. 14: E(t) = f(O(t)) + E_base + E_weather)
                occ_ratio = occupancy[f_name] / params['capacity']
                # f(O(t)) 假设为二次增长
                load_dynamic = params['base_load'] * (occ_ratio ** 1.5) 
                # E_weather: 温差越大，HVAC 负载越高
                temp_diff = abs(temp_ext - 22.0) 
                load_hvac = params['hvac_factor'] * temp_diff * 0.8 * occ_ratio
                
                total_energy = load_dynamic + params['base_load'] * 0.5 + load_hvac
                # 如果设备故障，能耗可能异常飙升或骤降
                if equipment_status[f_name] < 0.5:
                    total_energy *= np.random.uniform(1.5, 2.2) 
                
                records_energy.append({
                    'timestamp_step': t,
                    'hour': hour,
                    'facility_id': f_name,
                    'occupancy_count': occupancy[f_name],
                    'occupancy_ratio': occ_ratio,
                    'energy_kwh': total_energy,
                    'external_temp': temp_ext,
                    'equipment_status': equipment_status[f_name]
                })
                
                records_arrivals.append({
                    'timestamp_step': t,
                    'facility_id': f_name,
                    'arrivals': arrivals,
                    'departures': departures
                })

            # 3. 设备故障生成 (Weibull 分布)
            if t == failure_step and inject_failure:
                # 注入 HVAC 故障 (对应 Figure 5 和 Table 8)
                target_fac = 'Swimming Pool'
                equipment_status[target_fac] = 0.3 # 性能下降 70%
                records_failures.append({
                    'timestamp_step': t,
                    'facility_id': target_fac,
                    'failure_type': 'HVAC Breakdown',
                    'severity': 0.7
                })
                print(f"[!] Injected HVAC Breakdown at step {t} for {target_fac}")
            
            # 随机 Weibull 故障 (日常维护)
            for f_name in self.facility_names:
                if equipment_status[f_name] == 1.0 and np.random.rand() < 0.001:
                    k, lam = self.weibull_params[f_name]
                    # 极小概率发生严重故障
                    if weibull_min.rvs(k, scale=lam) > 1800: 
                        equipment_status[f_name] = np.random.uniform(0.4, 0.8)
                        records_failures.append({
                            'timestamp_step': t,
                            'facility_id': f_name,
                            'failure_type': 'Degradation',
                            'severity': 1.0 - equipment_status[f_name]
                        })

            # 4. 动态邻接矩阵计算 (Eq. 4)
            # A_ij(t) = exp(-d_ij / delta) + alpha * Flow_ij(t)
            delta = 30.0 # 距离衰减阈值
            alpha = 0.05 # 用户流动权重
            A_t = np.exp(-self.distance_matrix / delta) + alpha * user_flows
            # 归一化
            A_t = A_t / A_t.max() if A_t.max() > 0 else A_t
            
            # 每 96 步 (1天) 保存一次完整的动态图快照
            if (t + 1) % 96 == 0:
                day_id = (t + 1) // 96
                for i in range(self.n_facilities):
                    for j in range(self.n_facilities):
                        records_arrivals.append({ # 复用列表结构或单独建表，此处为简化直接写入图快照
                            'timestamp_step': t,
                            'facility_id': f"Graph_Edge_{self.facility_names[i]}_to_{self.facility_names[j]}",
                            'arrivals': A_t[i, j], # 借用字段存储权重
                            'departures': 0
                        })

        # 转换为 DataFrame 并保存
        df_energy = pd.DataFrame(records_energy)
        df_arrivals = pd.DataFrame(records_arrivals)
        df_failures = pd.DataFrame(records_failures)
        
        df_energy.to_csv(os.path.join(self.output_dir, 'raw_energy_environment.csv'), index=False)
        df_arrivals.to_csv(os.path.join(self.output_dir, 'raw_user_arrivals_flows.csv'), index=False)
        df_failures.to_csv(os.path.join(self.output_dir, 'raw_equipment_failures.csv'), index=False)
        
        print(f"Raw data saved to {self.output_dir}/")
        return df_energy, df_arrivals, df_failures

if __name__ == "__main__":
    # 运行模拟器，生成 10 天的原始高频数据
    # 若需复现论文 400 天训练集，请将 n_days 修改为 400
    simulator = SportsFacilitySimulator(output_dir='raw_data', seed=42)
    df_energy, df_arrivals, df_failures = simulator.simulate(n_days=10, inject_failure=True)
    
    print("\n--- Data Preview ---")
    print(df_energy.head())
    print("\nFailure Logs:")
    print(df_failures)