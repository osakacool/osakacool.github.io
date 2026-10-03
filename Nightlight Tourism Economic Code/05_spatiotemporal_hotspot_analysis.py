import numpy as np
import pandas as pd
from esda.getisord import G_Local
import libpysal

def extract_temporal_phases(hourly_intensity_df):
    """
    Section 5.3: Temporal analysis (Ignition, Peak, Decay phases)
    """
    phases = {
        'Ignition (18:00-20:00)': hourly_intensity_df.between_time('18:00', '20:00').mean(),
        'Peak (20:00-22:00)': hourly_intensity_df.between_time('20:00', '22:00').mean(),
        'Decay (22:00-02:00)': hourly_intensity_df.between_time('22:00', '02:00').mean()
    }
    return phases

def getis_ord_gi_star(values, w_matrix):
    """
    Section 5.4: Hotspot analysis using Getis-Ord Gi*
    Identifies significant clusters (p < 0.01)
    """
    gstat = G_Local(values, w_matrix)
    
    results = pd.DataFrame({
        'Gi': gstat.Gs,
        'p_value': gstat.p_sim,
        'z_score': gstat.Zs
    })
    
    # 提取 99% 置信度的热点 (p < 0.01, z > 2.58)
    hotspots = results[(results['p_value'] < 0.01) & (results['z_score'] > 0)]
    return hotspots

# 示例：验证周末与工作日差异 (Section 5.3)
def weekend_vs_weekday_analysis(df):
    weekend_extent = df[df['is_weekend']]['high_intensity_area'].sum()
    weekday_extent = df[~df['is_weekend']]['high_intensity_area'].sum()
    overflow_ratio = (weekend_extent - weekday_extent) / weekday_extent
    print(f"Weekend spatial overflow: {overflow_ratio * 100:.1f}%") # 论文结果: 42%