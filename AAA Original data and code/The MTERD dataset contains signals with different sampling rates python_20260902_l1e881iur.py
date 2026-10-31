import os
import numpy as np
import pandas as pd
import librosa
import mne
from scipy.signal import resample

class MTERDDataLoader:
    """
    对应论文 5.1 节：多模态数据加载与时间对齐。
    音频: 44.1 kHz | 生理信号(EEG/ECG/GSR): 256 Hz | 情感标注: 1 Hz
    """
    def __init__(self, session_dir):
        self.session_dir = session_dir
        self.audio_sr = 44100
        self.physio_sr = 256
        self.annotation_sr = 1
        self.target_sr = 10  # 对齐到 100ms 帧率 (10Hz)

    def load_session(self, session_id):
        """加载单个 session 的所有模态数据"""
        # 1. 加载音频 (44.1kHz)
        audio_path = os.path.join(self.session_dir, f"{session_id}_audio.wav")
        audio, sr = librosa.load(audio_path, sr=self.audio_sr, mono=True)
        
        # 2. 加载生理信号 (256Hz) - 假设存储为 EDF 或 CSV
        physio_path = os.path.join(self.session_dir, f"{session_id}_physio.csv")
        physio_df = pd.read_csv(physio_path) # 包含 EEG, ECG, GSR 列
        # 实际使用中需使用 mne.io.read_raw_edf() 读取标准 EDF 格式
        
        # 3. 加载情感标注 (1Hz, Valence/Arousal)
        anno_path = os.path.join(self.session_dir, f"{session_id}_annotations.csv")
        anno_df = pd.read_csv(anno_path) # 包含 timestamp, valence, arousal
        
        return audio, physio_df, anno_df

    def align_to_100ms(self, audio, physio_df, anno_df):
        """
        将所有信号重采样/对齐到 10Hz (100ms/帧)
        """
        duration_sec = len(audio) / self.audio_sr
        
        # 音频保持原采样率，后续按 100ms 切分
        # 生理信号从 256Hz 重采样到 10Hz
        physio_resampled = resample(physio_df.values, int(duration_sec * self.target_sr))
        physio_aligned = pd.DataFrame(physio_resampled, columns=physio_df.columns)
        
        # 标注信号从 1Hz 上采样到 10Hz (前向填充或线性插值)
        anno_resampled = np.repeat(anno_df[['valence', 'arousal']].values, 10, axis=0)
        # 截断或填充以匹配精确时长
        target_len = int(duration_sec * self.target_sr)
        anno_aligned = anno_resampled[:target_len]
        
        return audio, physio_aligned, anno_aligned