"""
predict.py — Deepfake Audio Detection Inference
Harshit Gautam | 24115072

Usage:
    python predict.py path/to/audio.wav
    python predict.py path/to/audio.wav --model best_model.keras --config model_config.json
"""

import sys
import json
import argparse
import numpy as np
import librosa
from scipy.fftpack import dct

# ── Constants (must match training) ─────────────────────────
SR            = 16000
DURATION      = 4.0
TARGET_FRAMES = 128
N_LFCC        = 60
N_MFCC        = 60
N_FFT         = 512
HOP_LENGTH    = 160
N_FILTERS     = 128


def compute_lfcc(audio):
    """Linear Frequency Cepstral Coefficients — anti-spoofing standard."""
    stft  = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    lin_f = np.linspace(0, SR // 2, N_FILTERS + 2)
    fb    = np.zeros((N_FILTERS, len(freqs)))
    for m in range(1, N_FILTERS + 1):
        fl, fc, fr = lin_f[m-1], lin_f[m], lin_f[m+1]
        for k, f in enumerate(freqs):
            if fl <= f <= fc:  fb[m-1, k] = (f  - fl) / (fc - fl + 1e-8)
            elif fc < f <= fr: fb[m-1, k] = (fr - f)  / (fr - fc + 1e-8)
    log_spec = np.log(np.dot(fb, stft) + 1e-8)
    return dct(log_spec, type=2, axis=0, norm='ortho')[:N_LFCC]


def extract_features(file_path):
    """Extract LFCC + MFCC + Delta-LFCC features. Returns (180, 128, 1)."""
    n = int(SR * DURATION)
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")

    audio = np.pad(audio, (0, max(0, n - len(audio))))[:n]
    # Pre-emphasis filter
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)

    lfcc       = compute_lfcc(audio)
    mfcc       = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH)
    lfcc_delta = librosa.feature.delta(lfcc)
    features   = np.concatenate([lfcc, mfcc, lfcc_delta], axis=0)

    T = features.shape[1]
    features = np.pad(features, ((0,0),(0, max(0, TARGET_FRAMES-T))))[:, :TARGET_FRAMES]

    # Per-sample normalization
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1,  keepdims=True) + 1e-6
    features = (features - mean) / std

    return features.astype(np.float32)[np.newaxis, ..., np.newaxis]  # (1, 180, 128, 1)


def predict_audio(file_path, model_path="best_model.keras",
                  config_path="model_config.json"):
    """
    Predict whether an audio file is Genuine or Deepfake.

    Args:
        file_path:   Path to .wav or .flac audio file
        model_path:  Path to trained model (.keras)
        config_path: Path to model config JSON

    Returns:
        label:      'Genuine (Human)' or 'Deepfake (AI-Generated)'
        confidence: float 0-1
    """
    from tensorflow.keras.models import load_model

    # Load model and config
    model = load_model(model_path)
    with open(config_path) as f:
        cfg = json.load(f)
    threshold = cfg["threshold"]

    # Extract features and predict
    features = extract_features(file_path)
    prob     = float(model.predict(features, verbose=0)[0][0])

    label      = "Deepfake (AI-Generated)" if prob > threshold else "Genuine (Human)"
    confidence = prob if prob > threshold else (1 - prob)

    return label, confidence, prob


def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detector")
    parser.add_argument("audio",   type=str, help="Path to audio file (.wav/.flac)")
    parser.add_argument("--model", type=str, default="best_model.keras",
                        help="Path to model file")
    parser.add_argument("--config", type=str, default="model_config.json",
                        help="Path to config JSON")
    args = parser.parse_args()

    print(f"\n{'='*45}")
    print("  Deepfake Audio Detector")
    print(f"{'='*45}")
    print(f"  File: {args.audio}")

    label, confidence, prob = predict_audio(args.audio, args.model, args.config)

    print(f"  Result    : {label}")
    print(f"  Confidence: {confidence*100:.1f}%")
    print(f"  Fake prob : {prob:.4f}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
