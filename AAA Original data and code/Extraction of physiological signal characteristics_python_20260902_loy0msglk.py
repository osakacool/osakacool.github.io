import neurokit2 as nk
from scipy.signal import welch

class PhysiologicalFeatureExtractor:
    """
    对应论文 5.1 节：提取生理信号生物标志物 (HRV, Alpha/Beta ratios)。
    """
    def __init__(self, physio_sr=256):
        self.sr = physio_sr

    def extract_ecg_hrv(self, ecg_signal):
        """提取 ECG 的 HRV (心率变异性) 特征"""
        # 使用 NeuroKit2 提取 R 峰值并计算 HRV
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=self.sr)
        hrv_features = nk.hrv_frequency(signals, sampling_rate=self.sr)
        # 返回 LF/HF ratio 等关键 HRV 指标
        return hrv_features['HRV_LFn'] / hrv_features['HRV_HFn'] 

    def extract_eeg_bands(self, eeg_signal):
        """提取 EEG 的 Alpha/Beta 频带能量比"""
        # 使用 Welch 方法计算功率谱密度 (PSD)
        freqs, psd = welch(eeg_signal, fs=self.sr, nperseg=self.sr*2)
        
        # 定义频带 (Hz)
        alpha_band = (freqs >= 8) & (freqs <= 12)
        beta_band = (freqs >= 13) & (freqs <= 30)
        
        alpha_power = np.mean(psd[alpha_band])
        beta_power = np.mean(psd[beta_band])
        
        return alpha_power / (beta_power + 1e-8) # Alpha/Beta ratio

    def process_session(self, physio_df):
        """
        按滑动窗口 (如 2秒窗口, 1秒步进) 提取生理特征，最终对齐到 10Hz
        """
        # 此处为伪代码示意，实际需按时间窗口切片并调用上述函数
        # 最终返回 shape: (N_frames_10Hz, 2) 的特征矩阵 [HRV_ratio, Alpha_Beta_ratio]
        pass