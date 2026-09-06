import librosa
import numpy as np
import os

def extract_logmel(wav_path, sr=16000, win_len=25, hop_len=10, n_mels=128):
    y,_ = librosa.load(wav_path, sr=sr)
    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=win_len*sr//1000, hop_length=hop_len*sr//1000, n_mels=n_mels, fmin=125, fmax=7500
    )
    logmel = np.log10(1e-8 + mel_spect.T)
    return logmel

def segment_sample(feature_arr, seg_frames=3000):
    T = feature_arr.shape[0]
    start = np.random.randint(0, T-seg_frames)
    return feature_arr[start:start+seg_frames,:]

def label_binarize(score_arr, thr=3.5):
    return (score_arr >= thr).astype(np.float32)
