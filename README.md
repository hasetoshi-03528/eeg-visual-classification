# eeg-visual-classification

EEG-based visual stimulus classification using EEGNet (PyTorch)  
Graduation thesis project — Tongji University, 2023

## Overview

This project implements EEG signal classification for a non-invasive 
Brain-Computer Interface (BCI) system. EEG data was recorded from a 
single subject while viewing visual stimuli (images and text), then 
classified using EEGNet, a compact CNN designed for EEG-based BCI tasks.

## 日本語概要

同済大学（中国）卒業論文プロジェクト（2023年）。

64チャンネル脳波キャップを用いて自身のEEG信号を計測・記録し、
EEGNet（PyTorch）による視覚刺激分類システムを構築。

視覚刺激（画像・テキスト）を見ているときの脳波から
カテゴリを分類するBCI（Brain-Computer Interface）研究。

2クラス分類で最高81.0%の精度を達成。

### 技術スタック
Python / PyTorch / MNE / NumPy / SciPy

## Experiment Design

- **Subject**: Single participant (self-recorded)
- **Stimuli**: 4 categories × 3 classes = 12 classes
  - Buildings (stadium, Tokyo Tower, laboratory)
  - People (Einstein, Messi, Jackie Chan)
  - Fruits (banana, orange, watermelon)
  - Sports (badminton, swimming, basketball)
- **Text modalities**:
  - Nouns: 12 classes (same as above, displayed for 4s)
  - Scripts: 3 sentences combining multiple classes (displayed for 8s)
    - "Messi plays soccer in the stadium"
    - "Einstein eats a banana in the laboratory"
    - "I watch Tokyo Tower with Jackie Chan"
- **EEG device**: 64-channel electrode cap
- **Sessions**: 14 recording sessions

## Methods

- Preprocessing: Bandpass filter (1–60 Hz), notch filter (50 Hz), ICA
  (artifact removal for EOG/ECG)
- Model: EEGNet (PyTorch)
- Train/test split: 6:1, Epoch: 200, Batch size: 32, LR: 3e-4

## Results

All results are from a single subject (n = 1, self-recorded, 14 sessions).

| Task | Accuracy | Chance level |
|---|---|---|
| Image, 2-class (mean of 66 pairs) | 51.3% (best 75.0%: Jackie Chan vs. Watermelon; worst 20.5%) | 50% |
| Noun, 2-class (mean of 66 pairs) | 62.8% (best 81.0%: Stadium vs. Tokyo Tower; worst 32.5%) | 50% |
| Image, 4-category | 26.3% | 25% |
| Noun, 4-category | 40.7% | 25% |
| Image + Noun, 24-class | 63.2% | ~4.2% |

The "best" values are the maximum over 66 pairwise comparisons and are
not representative of typical performance.

## Limitations

- Single subject; results do not generalize.
- Image 2-class accuracy is near chance on average, and 27 of 66 pairs
  are below 50%, which suggests unstable evaluation (e.g., distribution
  shift between train/test sessions). The train/test split unit
  (session-wise or epoch-wise) is [The split is session-wise (12 sessions for training, 2 for testing).].
- Text stimuli were cut into overlapping windows (0.25 s stride); if the
  split was done at the window level, train/test leakage is possible.

## Tech Stack

- Python, PyTorch, NumPy, SciPy, MNE, Matplotlib

## Data

EEG data is not publicly available.  
Thesis document available upon request.
