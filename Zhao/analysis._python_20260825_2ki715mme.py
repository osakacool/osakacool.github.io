import pandas as pd
import numpy as np
from scipy import stats

def perform_statistical_analysis(data_dir='data'):
    """
    执行论文中描述的统计检验：配对 t-test 和 Bonferroni 校正。
    对应论文 Section 6.1 和 Table 3。
    """
    df = pd.read_csv(os.path.join(data_dir, 'performance_comparison.csv'))
    
    # 提取 Proposed MARL-GNN 数据作为基准
    proposed = df[df['method'] == 'Proposed MARL-GNN'].reset_index(drop=True)
    baselines = ['MAPPO', 'Single-Agent PPO', 'Integer Programming']
    
    results = []
    
    for metric in ['occupancy_utilization', 'energy_efficiency', 'user_satisfaction']:
        for baseline in baselines:
            baseline_data = df[df['method'] == baseline].reset_index(drop=True)
            
            # 配对 t-test (Paired t-test)
            # 假设 step_id 是对应的配对键
            merged = pd.merge(proposed[['step_id', metric]], 
                              baseline_data[['step_id', metric]], 
                              on='step_id', 
                              suffixes=('_prop', '_base'))
            
            diff = merged[f'{metric}_prop'] - merged[f'{metric}_base']
            t_stat, p_value = stats.ttest_rel(merged[f'{metric}_prop'], merged[f'{metric}_base'])
            
            # Cohen's d
            cohen_d = diff.mean() / diff.std()
            
            # 95% Confidence Interval
            ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))
            
            results.append({
                'Metric': metric,
                'Comparison': f'Proposed vs {baseline}',
                'Mean Diff': diff.mean(),
                't-statistic': t_stat,
                'df': len(diff) - 1,
                'p-value': p_value,
                'Cohen_d': cohen_d,
                'CI_Lower': ci[0],
                'CI_Upper': ci[1]
            })
            
    results_df = pd.DataFrame(results)
    
    # Bonferroni Correction
    # 论文中提到 alpha_adj = 0.01 for multiple comparisons
    # 这里我们简单标记显著性
    results_df['Significant_Bonferroni'] = results_df['p-value'] < 0.01
    
    print("Statistical Analysis Results (Table 3 Reproduction):")
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    import os
    perform_statistical_analysis()