"""
app.py — Deepfake Audio Detection Web App
Harshit Gautam | 24115072

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import tempfile
import os
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

# ── Constants ────────────────────────────────────────────────
SR            = 16000
DURATION      = 4.0
TARGET_FRAMES = 128
N_LFCC        = 60
N_MFCC        = 60
N_FFT         = 512
HOP_LENGTH    = 160
N_FILTERS     = 128
MODEL_PATH    = "best_model.keras"
CONFIG_PATH   = "model_config.json"


# ── Feature extraction ───────────────────────────────────────
def compute_lfcc(audio):
    stft  = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    lin_f = np.linspace(0, SR // 2, N_FILTERS + 2)
    fb    = np.zeros((N_FILTERS, len(freqs)))
    for m in range(1, N_FILTERS + 1):
        fl, fc, fr = lin_f[m-1], lin_f[m], lin_f[m+1]
        for k, f in enumerate(freqs):
            if fl <= f <= fc:  fb[m-1, k] = (f  - fl) / (fc - fl + 1e-8)
            elif fc < f <= fr: fb[m-1, k] = (fr - f)  / (fr - fc + 1e-8)
    return dct(np.log(np.dot(fb, stft) + 1e-8), type=2, axis=0, norm='ortho')[:N_LFCC]


def extract_features(file_path):
    n = int(SR * DURATION)
    audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    audio = np.pad(audio, (0, max(0, n - len(audio))))[:n]
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)
    lfcc       = compute_lfcc(audio)
    mfcc       = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH)
    lfcc_delta = librosa.feature.delta(lfcc)
    features   = np.concatenate([lfcc, mfcc, lfcc_delta], axis=0)
    T = features.shape[1]
    features = np.pad(features, ((0,0),(0, max(0, TARGET_FRAMES-T))))[:, :TARGET_FRAMES]
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1,  keepdims=True) + 1e-6
    features = (features - mean) / std
    return features.astype(np.float32)[np.newaxis, ..., np.newaxis], audio


@st.cache_resource
def load_model_and_config():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return model, cfg


# ── UI ───────────────────────────────────────────────────────
st.title("🎙️ Deepfake Audio Detector")
st.markdown("**Harshit Gautam | 24115072**")
st.markdown("Upload a speech recording to detect whether it is **Genuine (Human)** or **Deepfake (AI-Generated)**.")

st.divider()

# Model loading
model_ok = os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH)
if not model_ok:
    st.error(
        " Model files not found. Place `best_model.keras` and `model_config.json` "
        "in the same directory as `app.py`."
    )
    st.stop()

model, cfg = load_model_and_config()
threshold  = cfg["threshold"]

st.success(" Model loaded successfully")

with st.expander(" Model Performance"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", "88.93%")
    col2.metric("EER", "6.78%")
    col3.metric("ROC AUC", "96.36%")
    col1.metric("F1 Score", "88.15%")
    col2.metric("Real Class Acc", "97.75%")
    col3.metric("Fake Class Acc", "80.51%")

st.divider()

# File upload
uploaded = st.file_uploader(
    "Upload Audio File",
    type=["wav", "flac", "mp3", "ogg"],
    help="Supported formats: WAV, FLAC, MP3, OGG"
)

if uploaded:
    st.audio(uploaded, format="audio/wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing audio..."):
        try:
            features, audio = extract_features(tmp_path)
            prob     = float(model.predict(features, verbose=0)[0][0])
            threshold = cfg["threshold"]  # should be 0.05

            is_fake    = prob > threshold
            label      = "Deepfake (AI-Generated)" if is_fake else "Genuine (Human)"
            # Confidence = how sure we are about the prediction
            confidence = prob if is_fake else (1 - prob)

            # ── Result ──────────────────────────────────────
            st.divider()
            if is_fake:
                st.error(f"## {label}")
            else:
                st.success(f"## {label}")

            col1, col2 = st.columns(2)
            col1.metric("Fake Probability", f"{prob*100:.1f}%")
            col2.metric("Result", "Deepfake" if is_fake else "Genuine")

            # Confidence bar
            st.markdown("**Detection Confidence:**")
            bar_color = "red" if is_fake else "green"
            st.progress(confidence)

            # ── Waveform ─────────────────────────────────────
            st.divider()
            st.markdown("**Waveform**")
            fig, ax = plt.subplots(figsize=(10, 2))
            times = np.linspace(0, len(audio)/SR, len(audio))
            ax.plot(times, audio, linewidth=0.5,
                    color="#e74c3c" if is_fake else "#2ecc71")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Audio Waveform — {label}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Mel Spectrogram ──────────────────────────────
            st.markdown("**Mel Spectrogram**")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH,
                                           x_axis="time", y_axis="mel", ax=ax2)
            fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
            ax2.set_title("Mel Spectrogram")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            os.unlink(tmp_path)

st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "CNN + BiLSTM | LFCC + MFCC Features | ASVspoof 2019 + Fake-or-Real Dataset"
    "</div>",
    unsafe_allow_html=True
)
