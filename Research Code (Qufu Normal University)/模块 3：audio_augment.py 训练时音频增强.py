# audio_augment.py
import librosa
import numpy as np

def augment_raw_audio(waveform: np.ndarray, sr: int =16000):
    """输入原始音频波形 (T,)，返回增强后波形"""
    # 1. pitch shift ±2 semitones
    n_steps = np.random.uniform(-2, 2)
    wav_aug = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

    # 2. time‑stretch 0.9 ~1.1
    rate = np.random.uniform(0.9, 1.1)
    wav_aug = librosa.effects.time_stretch(wav_aug, rate=rate)

    # 3. add gaussian noise SNR 20‑40 dB
    snr_db = np.random.uniform(20,40)
    signal_power = np.mean(wav_aug ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*wav_aug.shape) * np.sqrt(noise_power)
    wav_aug = wav_aug + noise

    return wav_aug
