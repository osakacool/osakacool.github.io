import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn


class SourceAffineProjection(nn.Module):
    """论文公式(2) source‑specific affine projection W_s @ n + b_s"""
    def __init__(self, src_dim: int, unified_dim: int):
        super().__init__()
        self.W = nn.Linear(src_dim, unified_dim, bias=True) # W_s and b_s

    def forward(self, x_src):
        return self.W(x_src)


def ontology_alignment_semantic(raw_ds_list):
    """语义本体对齐：字符串归一化、同义词匹配（仿真实现）
    raw_ds_list: list[pd.DataFrame] 3个数据源 USDA,AFCD,CoFID
    """
    aligned_list = []
    for df in raw_ds_list:
        df = df.copy()
        # 食品名称归一化
        df["food_name_canon"] = df["food_name"].str.lower().str.replace(r'[^\w]'," ",regex=True)
        aligned_list.append(df)
    return aligned_list


def knn_impute_unified_space(X_unified: np.ndarray, k=10):
    """统一语义空间K‑NN距离加权填补缺失值，论文2.2.1"""
    mask_nan = np.isnan(X_unified)
    X_complete = X_unified.copy()
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(np.nan_to_num(X_unified))

    for i in range(X_unified.shape[0]):
        nan_cols = np.where(mask_nan[i,:])[0]
        if len(nan_cols)==0:
            continue
        dist, idx = nbrs.kneighbors(np.nan_to_num(X_unified[i:i+1]))
        neighbor_data = X_unified[idx[0],:]
        for c in nan_cols:
            vals = neighbor_data[:,c]
            valid_vals = vals[~np.isnan(vals)]
            weights = 1.0/(dist[0][1:]+1e-8)
            weights = weights[:len(valid_vals)]
            X_complete[i,c] = np.average(valid_vals, weights=weights)
    return X_complete


def multi_source_fusion_pipeline(df_usda, df_afcd, df_cofid, unified_kdim=128, k_knn=10):
    """完整多源融合流水线，对应图2
    返回 fused_corpus_df, X_fused_tensor
    """
    raw_datasets = [df_usda, df_afcd, df_cofid]
    aligned_dfs = ontology_alignment_semantic(raw_datasets)

    # 初始化每个数据源的仿射投影层
    src_dims = [df.shape[1]-2 for df in aligned_dfs] #减去name,id列
    proj_layers = [SourceAffineProjection(d, unified_kdim) for d in src_dims]

    unified_all = []
    meta_records = []
    for s_idx, df in enumerate(aligned_dfs):
        nut_raw = df.drop(["food_name","food_name_canon"],axis=1).values.astype(np.float32)
        nut_tensor = torch.from_numpy(nut_raw)
        proj_out = proj_layers[s_idx](nut_tensor).detach().cpu().numpy()
        unified_all.append(proj_out)
        for _,row in df.iterrows():
            meta_records.append({"source_idx":s_idx,"food_canon":row["food_name_canon"]})

    X_unified = np.concatenate(unified_all,axis=0)
    X_filled = knn_impute_unified_space(X_unified, k=k_knn)

    # z‑score标准化，只用训练集统计，这里仿真全部
    mu = np.mean(X_filled,axis=0,keepdims=True)
    sigma = np.std(X_filled,axis=0,keepdims=True)+1e-8
    X_zscore = (X_filled - mu)/sigma

    fused_corpus_df = pd.DataFrame(meta_records)
    return fused_corpus_df, torch.from_numpy(X_zscore.astype(np.float32))


if __name__=="__main__":
    # =========3个数据库，真实项目替换为读取USDA/AFCD/CoFID csv=========
    np.random.seed(42)
    n_usda,n_afcd,n_cofid = 4520,3815,4130
    df_usda = pd.DataFrame({"food_name":[f"food_u{i}" for i in range(n_usda)]})
    for nut in ["protein","sodium","iron","fat","carb"]:
        df_usda[nut] = np.random.randn(n_usda)*5
        mask_miss = np.random.choice([True,False],size=n_usda,p=[0.124,1-0.124])
        df_usda.loc[mask_miss,nut]=np.nan

    df_afcd = pd.DataFrame({"food_name":[f"food_a{i}" for i in range(n_afcd)]})
    for nut in ["protein","sodium","iron","fat","carb"]:
        df_afcd[nut] = np.random.randn(n_afcd)*6
        mask_miss = np.random.choice([True,False],size=n_afcd,p=[0.187,1-0.187])
        df_afcd.loc[mask_miss,nut]=np.nan

    df_cofid = pd.DataFrame({"food_name":[f"food_c{i}" for i in range(n_cofid)]})
    for nut in ["protein","sodium","iron","fat","carb"]:
        df_cofid[nut] = np.random.randn(n_cofid)*5.5
        mask_miss = np.random.choice([True,False],size=n_cofid,p=[0.152,1-0.152])
        df_cofid.loc[mask_miss,nut]=np.nan

    fused_df,X_fused = multi_source_fusion_pipeline(df_usda,df_afcd,df_cofid,unified_kdim=128,k_knn=10)
    print(f"融合后数据集大小 {X_fused.shape[0]} 条食品记录")
