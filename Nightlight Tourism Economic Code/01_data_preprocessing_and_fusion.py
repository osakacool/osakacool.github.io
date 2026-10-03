import numpy as np
import rasterio
import geopandas as gpd
from scipy.stats import gaussian_kde
from rasterio.transform import from_origin
import logging

logging.basicConfig(level=logging.INFO)

class NocturnalDataProcessor:
    def __init__(self, grid_size=500, bbox=(34.25, 108.93, 34.27, 108.95)):
        self.grid_size = grid_size  # 500m resolution
        self.bbox = bbox
        # 论文 Table 1 中的预处理参数
        self.viirs_gaussian_sigma = 0.8 
        self.mobility_weights = {'weekday': 0.7, 'weekend': 1.3}

    def calibrate_viirs(self, dn_array, gain=1.0, bias=0.0):
        """Eq. 1: Radiometric Calibration & Gaussian Filtering"""
        radiance = gain * dn_array + bias
        # 实际应用中需使用 scipy.ndimage.gaussian_filter(radiance, sigma=self.viirs_gaussian_sigma)
        return radiance

    def compute_poi_kde(self, poi_gdf, bandwidth=None):
        """Eq. 2: POI Kernel Density Estimation (KDE)"""
        x = poi_gdf.geometry.x.values
        y = poi_gdf.geometry.y.values
        kde = gaussian_kde([x, y], bw_method=bandwidth)
        # 生成网格并计算密度，随后进行 Log10 变换 (Eq. 3)
        # ... [此处省略网格生成代码，返回 log10(density + 1)]
        return np.log10(kde.evaluate([x, y]) + 1e-5) 

    def weight_human_mobility(self, mobility_df):
        """Eq. 4: Temporal Weighting for Human Mobility"""
        mobility_df['is_weekend'] = mobility_df['date'].dt.dayofweek >= 5
        weights = mobility_df['is_weekend'].map(self.mobility_weights)
        mobility_df['weighted_flow'] = mobility_df['flow_count'] * weights
        return mobility_df

    def fuse_multisource_tensor(self, viirs_path, poi_path, mobility_path, social_path):
        """
        生成 5-Dimensional Input Tensor (Table 1)
        Channels: [Radiance, POI_Density, Mobility_T1, Mobility_T2, Mobility_T3]
        """
        logging.info("Aligning all datasets to 500m grid...")
        # 1. 读取并对齐 VIIRS (Channel 0)
        # 2. 计算 POI KDE (Channel 1)
        # 3. 聚合 Mobility 数据到 3 个时间段 (Channels 2, 3, 4)
        # 4. Social Media 用于验证，不直接作为输入通道，但可用于生成 Ground Truth 标签
        
        # Mock 返回 5通道 numpy array: shape (H, W, 5)
        H, W = 100, 100 # 示例尺寸
        tensor = np.random.rand(H, W, 5).astype(np.float32)
        return tensor

if __name__ == "__main__":
    processor = NocturnalDataProcessor()
    # tensor_5d = processor.fuse_multisource_tensor(...)