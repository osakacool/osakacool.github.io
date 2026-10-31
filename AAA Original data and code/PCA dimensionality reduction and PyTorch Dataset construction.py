import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MusicTherapyDataset(Dataset):
    """
    对应论文 5.1 & 5.2 节：PCA 降维至 128 维，构建 300 帧 (30秒) 序列。
    """
    def __init__(self, session_features_list, session_targets_list, seq_len=300, n_components=128):
        """
        session_features_list: List of np.array, 每个 shape 为 (N_frames, 6373)
        session_targets_list: List of np.array, 每个 shape 为 (N_frames,) (1Hz 上采样后的 10Hz 标注)
        """
        self.seq_len = seq_len # 论文 5.2 节: 300 frames = 30 seconds
        
        # 1. 全局 PCA 降维 (在训练集上 fit，保持 95% 方差)
        all_features = np.vstack(session_features_list)
        self.scaler = StandardScaler()
        all_features_scaled = self.scaler.fit_transform(all_features)
        
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(all_features_scaled)
        print(f"PCA Explained Variance Ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # 2. 转换所有 session 的特征
        self.processed_sessions = []
        self.processed_targets = []
        
        for feats, targets in zip(session_features_list, session_targets_list):
            feats_scaled = self.scaler.transform(feats)
            feats_pca = self.pca.transform(feats_scaled) # Shape: (N_frames, 128)
            
            # 确保特征和目标长度一致
            min_len = min(len(feats_pca), len(targets))
            self.processed_sessions.append(feats_pca[:min_len])
            self.processed_targets.append(targets[:min_len])

    def __len__(self):
        # 返回所有可能的 300 帧序列总数
        total_sequences = sum([len(s) - self.seq_len + 1 for s in self.processed_sessions])
        return total_sequences

    def __getitem__(self, idx):
        # 定位到具体的 session 和起始帧
        for session_idx, session_data in enumerate(self.processed_sessions):
            seq_count = len(session_data) - self.seq_len + 1
            if idx < seq_count:
                start_frame = idx
                break
            idx -= seq_count
            
        # 截取 300 帧 (30秒) 的序列
        x = self.processed_sessions[session_idx][start_frame : start_frame + self.seq_len]
        y = self.processed_targets[session_idx][start_frame : start_frame + self.seq_len]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)