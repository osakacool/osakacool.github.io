import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import libpysal
from esda.moran import Moran
from spreg import ML_Lag # 空间滞后模型 (SAR)

# Table 3: Economic Coefficients (αₖ)
ECONOMIC_WEIGHTS = {
    'Core_Consumption': 1.0,
    'Dining_Cluster': 0.8,
    'Cultural_Performance': 0.8,
    'Transportation_Hub': 0.5,
    'Non_Tourism': 0.0
}

def calculate_tourism_activity_index(segmented_pixels, poi_density):
    """Eq. 7 & 8: Tourism Activity Index & Economic Output"""
    # 统计各类别像素数量并乘以经济系数
    activity_index = 0
    for zone, weight in ECONOMIC_WEIGHTS.items():
        pixel_count = np.sum(segmented_pixels == zone)
        activity_index += pixel_count * weight
    
    # 结合 POI 密度
    economic_output = activity_index * poi_density 
    return economic_output

def spatial_econometrics_analysis(df, w_matrix):
    """
    Table 8: Performance Comparison of Linear and Non-Linear Models
    df must contain: 'merchant_revenue', 'activity_index', 'poi_density', 'geometry'
    """
    X = df[['activity_index', 'poi_density']]
    y = df['merchant_revenue']
    
    # 1. Baseline OLS
    X_ols = sm.add_constant(X)
    ols_model = sm.OLS(y, X_ols).fit()
    
    # 2. Moran's I Test for Spatial Autocorrelation (Section 5.4)
    moran = Moran(y, w_matrix)
    print(f"Moran's I: {moran.I}, p-value: {moran.p_sim}")
    
    # 3. Spatial Autoregressive (SAR) Model (Final Model)
    sar_model = ML_Lag(y, X.values, w=w_matrix, name_y='revenue', name_x=['index', 'poi'])
    
    # 4. Gradient Boosting Regressor (GBR)
    gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    gbr_model.fit(X, y)
    gbr_pred = gbr_model.predict(X)
    
    return {
        'OLS_R2': ols_model.rsquared,
        'SAR_R2': sar_model.r2, # 论文中为 0.87
        'GBR_R2': r2_score(y, gbr_pred),
        'SAR_Model': sar_model
    }

if __name__ == "__main__":
    # 构建空间权重矩阵 (Queen Contiguity)
    # gdf = gpd.read_file("study_area_grid.shp")
    # w = libpysal.weights.Queen.from_dataframe(gdf)
    # results = spatial_econometrics_analysis(gdf, w)