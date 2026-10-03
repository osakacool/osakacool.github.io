import geopandas as gpd
import pandas as pd
import numpy as np

class DynamicPOIValidator:
    """
    Filters out 'ghost' or inactive POIs using multi-source dynamic validation.
    Corresponds to Section 6.1.1 Discussion.
    """
    def __init__(self, core_hours=('18:00', '02:00'), activity_threshold=0.2):
        self.core_hours = core_hours
        self.threshold = activity_threshold

    def extract_nocturnal_signals(self, mobility_gdf, social_gdf):
        """提取核心夜间时段 (18:00-02:00) 的动态信号"""
        # 过滤时间
        mob_night = mobility_gdf[mobility_gdf['time'].dt.strftime('%H:%M').between(*self.core_hours)]
        soc_night = social_gdf[social_gdf['time'].dt.strftime('%H:%M').between(*self.core_hours)]
        
        # 聚合到 500m 网格
        mob_grid = mob_night.groupby('grid_id')['flow_count'].sum().reset_index()
        soc_grid = soc_night.groupby('grid_id')['check_in_count'].sum().reset_index()
        
        return mob_grid, soc_grid

    def validate_poi_activity(self, poi_gdf, mob_grid, soc_grid):
        """
        交叉验证：只有当 POI 所在的 500m 网格在夜间有足够的人流和打卡时，
        该 POI 才被保留为 'Active'。
        """
        # 合并数据
        poi_validated = poi_gdf.merge(mob_grid, on='grid_id', how='left')
        poi_validated = poi_validated.merge(soc_grid, on='grid_id', how='left')
        
        # 填充空值
        poi_validated['flow_count'] = poi_validated['flow_count'].fillna(0)
        poi_validated['check_in_count'] = poi_validated['check_in_count'].fillna(0)
        
        # 归一化动态得分 (0-1)
        max_mob = poi_validated['flow_count'].max()
        max_soc = poi_validated['check_in_count'].max()
        
        poi_validated['mob_score'] = poi_validated['flow_count'] / (max_mob + 1e-5)
        poi_validated['soc_score'] = poi_validated['check_in_count'] / (max_soc + 1e-5)
        
        # 综合活跃度得分 (Mobility 权重 0.6, Social 权重 0.4)
        poi_validated['activity_score'] = 0.6 * poi_validated['mob_score'] + 0.4 * poi_validated['soc_score']
        
        # 过滤 "Ghost Venues" (幽灵/废弃 POI)
        poi_validated['is_active'] = poi_validated['activity_score'] >= self.threshold
        
        active_pois = poi_validated[poi_validated['is_active']]
        ghost_pois = poi_validated[~poi_validated['is_active']]
        
        print(f"Dynamic Validation: Retained {len(active_pois)} active POIs, filtered {len(ghost_pois)} ghost venues.")
        return active_pois

# 使用示例
# validator = DynamicPOIValidator()
# active_pois = validator.validate_poi_activity(raw_pois, mobility_data, social_data)
# # 随后仅使用 active_pois 进行 KDE 计算 (Section 3.3.2)