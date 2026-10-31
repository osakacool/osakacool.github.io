import opensmile
import numpy as np

class AcousticFeatureExtractor:
    """
    对应论文 5.1 节：使用 openSMILE 提取 100ms 帧的 6373 维声学特征。
    """
    def __init__(self):
        # 配置 openSMILE: ComParE_2016 包含 6373 个 LLD (Low-Level Descriptors) 及其函数特征
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors, # 提取帧级别特征
        )
        self.frame_size = 0.1  # 100ms
        self.hop_size = 0.1    # 100ms 无重叠滑动窗口

    def extract(self, audio, sr=44100):
        """
        输入: 原始音频波形
        输出: shape 为 (N_frames, 6373) 的特征矩阵
        """
        # 将音频切分为 100ms 的帧
        n_frames = int(len(audio) / (sr * self.hop_size))
        features_list = []
        
        for i in range(n_frames):
            start_idx = int(i * sr * self.hop_size)
            end_idx = start_idx + int(sr * self.frame_size)
            frame = audio[start_idx:end_idx]
            
            # 提取单帧特征
            frame_features = self.smile.process_signal(frame, sr)
            features_list.append(frame_features.values.flatten())
            
        return np.array(features_list) # Shape: (N_frames, 6373)