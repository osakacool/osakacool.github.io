"""
01_sim_dataset.py
对应论文章节：5.1 Study Areas and Data Collection；4.2 GIS‑Coupled Multi‑Scale Data Integration
功能：1.生成模拟时空GIS生态数据集；2.空间块划分；【替换真实GIS观测数据入口】
输出：pytorch张量数据集、空间划分索引
"""
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_simulation_dataset(n_spatial:int=1200, n_time:int=120, seed=42):
    """
    生成模拟时空生态数据集
    :param n_spatial:空间网格数量
    :param n_time:时间步(月份)
    :param seed:随机种子保证可复现
    :return: dict 包含X[T,S,D], Y_obs[T,S], 空间坐标,研究区标签
    【替换真实数据】：在此函数内读取GIS栅格、FLUXNET、NOAA监测站数据，对齐输出张量维度
    X维度格式：[Time, Spatial_sample, Feature_dim]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_feature = 8

    # 空间坐标
    x_coord = np.random.uniform(0,100, size=n_spatial)
    y_coord = np.random.uniform(0,80, size=n_spatial)
    # 0=西北温带森林，1=墨西哥湾滨海湿地，2=长三角城市绿地
    study_area_id = np.random.choice([0,1,2], size=n_spatial, p=[0.35,0.30,0.35])

    # 时空特征张量 X [T,S,D]
    X = np.zeros((n_time, n_spatial, n_feature))
    for t in range(n_time):
        season_mod = np.sin(2 * np.pi * t / 12)
        for s in range(n_spatial):
            X[t,s,0] = 0.5 + 0.3*season_mod + np.random.normal(0,0.05)  # NDVI
            X[t,s,1] = 20 + 10*season_mod + np.random.normal(0,1.2)      # 气温
            X[t,s,2] = np.clip(800 + 200*np.random.randn(), 200,1800)    # 降水
            X[t,s,3] = np.random.uniform(0,100)                          # 高程
            X[t,s,4] = np.random.uniform(0,50)                           # 土壤碳
            X[t,s,5] = np.random.uniform(0,1)                            # 土地覆盖
            X[t,s,6] = x_coord[s]
            X[t,s,7] = y_coord[s]

    # 生成模拟非市场估值真值
    Y_obs = np.zeros((n_time, n_spatial))
    for t in range(n_time):
        season = np.sin(2 * np.pi * t / 12)
        for s in range(n_spatial):
            base = 0.2 + 0.4 * X[t,s,0] + 0.2*season
            if study_area_id[s]==0:
                base +=0.12
            elif study_area_id[s]==1:
                base +=0.06
            elif study_area_id[s]==2:
                base -=0.05
            noise = np.random.normal(0,0.035)
            Y_obs[t,s] = np.clip(base+noise, 0.0, 1.0)

    dataset = {
        "X": torch.tensor(X, dtype=torch.float32, device=device),
        "Y_obs": torch.tensor(Y_obs, dtype=torch.float32, device=device),
        "study_area_id": torch.tensor(study_area_id, dtype=torch.long, device=device),
        "x_coord": torch.tensor(x_coord, dtype=torch.float32, device=device),
        "y_coord": torch.tensor(y_coord, dtype=torch.float32, device=device)
    }
    return dataset


def spatial_block_split(dataset, test_size=0.15, random_state=42):
    """
    论文5.4 Spatial block cross‑validation
    空间块交叉验证，避免空间泄露
    :param dataset: 数据集字典
    :param test_size:测试集占比
    :return train_sp_idx, test_sp_idx:空间样本索引
    """
    n_spatial_total = dataset["X"].shape[1]
    spatial_idx = np.arange(n_spatial_total)
    groups = dataset["study_area_id"].cpu().numpy()
    gss = GroupShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
    split_list = list(gss.split(spatial_idx, groups=groups))
    train_sp_idx, test_sp_idx = split_list[0][0], split_list[0][1]
    return train_sp_idx, test_sp_idx


if __name__ == "__main__":
    # 测试运行数据模块
    sim_data = generate_simulation_dataset(n_spatial=1200, n_time=120, seed=42)
    train_idx, test_idx = spatial_block_split(sim_data)
    print(f"总空间样本：{sim_data['X'].shape[1]}")
    print(f"训练空间样本:{len(train_idx)} | 测试空间样本:{len(test_idx)}")
    print(f"X tensor shape [T,S,D] = {sim_data['X'].shape}")
#（注：内容由AI生成）
