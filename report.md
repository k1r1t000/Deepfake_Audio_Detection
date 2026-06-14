# Performance Report — Deepfake Audio Detection
**Harshit Gautam | 24115072**

---

## 1. Final Test Set Results

### Primary Metrics (Verification Criteria)

| Metric | Score | Required | Status |
|--------|-------|----------|--------|
| Overall Accuracy | **88.93%** | ≥ 80% |  PASS |
| Equal Error Rate (EER) | **6.78%** | ≤ 12% |  PASS |

### Secondary Metrics

| Metric | Score | Required | Status |
|--------|-------|----------|--------|
| F1 Score | **88.15%** | ≥ 80% |  PASS |
| Real (Genuine) Class Accuracy | **97.75%** | ≥ 75% |  PASS |
| Fake (Deepfake) Class Accuracy | **80.51%** | ≥ 75% |  PASS |
| ROC AUC | **96.36%** | — | — |

**Decision Threshold:** 0.05 (selected by maximizing balanced per-class accuracy on validation set)

---

## 2. Confusion Matrix

```
                  Predicted Real    Predicted Fake
Actual Real           2212               52
Actual Fake            461             1909
```

- True Positives (Fake correctly detected): 1909
- True Negatives (Real correctly detected): 2212
- False Positives (Real misclassified as Fake): 52
- False Negatives (Fake misclassified as Real): 461

---

## 3. Classification Report

```
              precision    recall  f1-score   support

        Real       0.83      0.98      0.90      2264
        Fake       0.97      0.81      0.88      2370

    accuracy                           0.89      4634
   macro avg       0.90      0.89      0.89      4634
weighted avg       0.90      0.89      0.89      4634
```

---

## 4. Validation Set Results

| Metric | Score |
|--------|-------|
| Val Accuracy | 99.94% |
| Val F1 Score | 99.94% |
| Val Real Class Accuracy | 99.91% |
| Val Fake Class Accuracy | 99.96% |

---

## 5. Training Summary

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1 | 86.62% | 98.19% | 0.1720 |
| 2 | 97.38% | 98.82% | 0.1417 |
| Best | **99.85%** | **99.89%** | **0.1199** |

Training stopped at best val_loss via EarlyStopping (patience=5).

---

## 6. Preprocessing Pipeline

1. **Audio loading:** 16kHz sample rate, 4-second fixed duration (pad/trim)
2. **Pre-emphasis filter:** `y[t] = y[t] - 0.97 * y[t-1]` — boosts high frequencies
3. **LFCC extraction:** 60 coefficients, linear filterbank (128 filters), FFT=512, hop=160
4. **MFCC extraction:** 60 coefficients, FFT=512, hop=160
5. **Delta-LFCC:** First-order temporal derivatives of LFCC
6. **Feature stacking:** Concatenate → (180, T) feature matrix
7. **Time normalization:** Pad/trim to 128 frames
8. **Per-sample normalization:** `(x - mean) / std` per feature dimension

---

## 7. Model Architecture

```
Input Shape: (180, 128, 1)

CNN Block 1: Conv2D(32, 3×3) → BN → Swish → MaxPool(2,2) → SpatialDropout(0.1)
CNN Block 2: Conv2D(64, 3×3) → BN → Swish → MaxPool(2,2) → SpatialDropout(0.1)
CNN Block 3: Conv2D(128,3×3) → BN → Swish → MaxPool(3,2) → SpatialDropout(0.1)
CNN Block 4: Conv2D(128,3×3) → BN → Swish → MaxPool(1,2) → SpatialDropout(0.1)

Reshape: (15, 1024)

BiLSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
BiLSTM(64,  return_sequences=False, dropout=0.2)

Dense(128, swish, L2=1e-3)
Dropout(0.4)
Dense(1, sigmoid)

Total Parameters: 1,602,977
```

---

## 8. Why This Approach Works

**The core challenge:** The test set contains TTS systems not present in training data. Standard Mel spectrogram models memorize system-specific artifacts and fail on unseen systems (we observed this — val accuracy was 99.9% but test accuracy was only 51% with Mel features).

**Solution:**
- **LFCC** captures universal artifacts in the linear frequency domain that all TTS systems produce
- **ASVspoof 2019 supplementary data** (22,800 fake samples from 8 diverse TTS/VC systems) taught the model to generalize
- **SpecAugment** prevented overfitting to specific TTS signatures
- **Label smoothing** prevented overconfident predictions that fail on distribution shift
