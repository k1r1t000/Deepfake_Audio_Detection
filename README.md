# Deepfake Audio Detection
**Harshit Gautam | 24115072**

A deep learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** using a CNN + BiLSTM architecture trained on anti-spoofing features.

---

## Results

| Metric | Score | Required | Status |
|--------|-------|----------|--------|
| Overall Accuracy | 88.93% | ≥ 80% | Pass |
| Equal Error Rate (EER) | 6.78% | ≤ 12% | Pass |
| F1 Score | 88.15% | ≥ 80% | Pass |
| Real Class Accuracy | 97.75% | ≥ 75% | Pass |
| Fake Class Accuracy | 80.51% | ≥ 75% | Pass |
| ROC AUC | 96.36% | -- | --|

---

### Live Demo
[Try the Web App](YOUR_STREAMLIT_LINK)


**Kaggle Notebook:** [View on Kaggle](https://www.kaggle.com/code/k1r1t00/deepfake-audio-detection)

## Installation

```bash
git clone https://github.com/k1r1t000/Deepfake_Audio_Detection
cd Deepfake_Audio_Detection
pip install -r requirements.txt
```


## Methodology

### 1. Dataset
- **Primary:** [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) — `for-norm` split (Train: 53,868 | Val: 10,798 | Test: 4,634)
- **Supplementary:** [ASVspoof 2019 LA](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) — 22,800 additional fake samples for cross-dataset generalization

### 2. Feature Extraction
Standard Mel spectrograms fail on this task because they compress high-frequency information where TTS artifacts reside. Instead, we use:

| Feature | Coefficients | Why |
|---------|-------------|-----|
| LFCC (Linear Frequency Cepstral Coefficients) | 60 | Linear filterbank preserves high-freq TTS artifacts — ASVspoof competition standard |
| MFCC | 60 | Complementary phonetic features |
| Delta-LFCC | 60 | Temporal dynamics (rate of change) |
| **Total** | **180** | Combined feature map: (180, 128, 1) |

**Preprocessing pipeline:**
1. Load audio at 16kHz, pad/trim to 4 seconds
2. Pre-emphasis filter (α=0.97) — boosts high frequencies
3. Extract LFCC + MFCC + Delta-LFCC
4. Per-sample normalization (mean/std per feature dimension)

### 3. Model Architecture — CNN + BiLSTM

```
Input (180, 128, 1)
    ↓
Conv2D(32) → BN → Swish → MaxPool(2,2) → SpatialDropout
    ↓
Conv2D(64) → BN → Swish → MaxPool(2,2) → SpatialDropout
    ↓
Conv2D(128) → BN → Swish → MaxPool(3,2) → SpatialDropout
    ↓
Conv2D(128) → BN → Swish → MaxPool(1,2) → SpatialDropout
    ↓
Reshape → (15, 1024)
    ↓
BiLSTM(128, return_sequences=True)
    ↓
BiLSTM(64)
    ↓
Dense(128, swish) → Dropout(0.4)
    ↓
Dense(1, sigmoid)
```

**Total parameters:** 1,602,977

### 4. Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=3e-4, clipnorm=1.0) |
| Loss | Binary Crossentropy (label_smoothing=0.05) |
| Epochs | 15 (EarlyStopping on val_loss, patience=5) |
| Batch size | 32 |
| Augmentation | SpecAugment (freq mask=20, time mask=20) |
| Class weights | Balanced (real=1.42, fake=0.77) |

### 5. Key Design Decisions

**Why LFCC over Mel?**
The test set contains TTS systems not seen during training. Mel spectrogram memorizes system-specific artifacts. LFCC's linear filterbank captures universal high-frequency artifacts present across all TTS/VC systems.

**Why ASVspoof 2019?**
The for-norm test set uses different TTS systems than training. Adding 22,800 fake samples from 8 diverse TTS/VC systems taught the model to detect generic deepfake artifacts rather than system-specific fingerprints.

**Why label smoothing?**
Prevents overconfident predictions (99.99% probability) that don't generalize to unseen data.

**Why SpecAugment?**
Random time and frequency masking during training prevents memorization of TTS-system-specific spectral patterns.

---

## Repository Structure

```
DeepfakeAudioDetection/
  README.md               -> This file
  notebook.ipynb          -> Kaggle notebook (reference only, run on Kaggle)
  predict.py              -> Inference script for new audio files
  app.py                  -> Streamlit web app
  report.md               -> Performance report with metrics
  requirements.txt        -> Dependencies
```


---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run inference on a single file
```bash
python predict.py path/to/audio.wav
```

### Launch Streamlit web app
```bash
streamlit run app.py
```

---

## Datasets Used
- [The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- [ASVspoof 2019 Dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)

