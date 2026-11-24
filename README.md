# eeg-visual-classification

EEG-based visual stimulus classification using EEGNet (PyTorch)  
Graduation thesis project — Tongji University, 2023

## Overview

This project implements EEG signal classification for a non-invasive
Brain-Computer Interface (BCI) system. EEG data was recorded from a
single subject while viewing visual stimuli (images and text), then
classified using EEGNet, a compact CNN designed for EEG-based BCI tasks.
This is an exploratory single-subject study; see Limitations.

## 日本語概要

同済大学（中国）卒業論文プロジェクト（2023年）。

64チャンネル脳波キャップで自身のEEG信号を計測・記録し、
EEGNet（PyTorch）で視覚刺激（画像・テキスト）のカテゴリ分類を行った、
非侵襲BCIを念頭に置いた探索的な研究です。

結果：2クラス分類（66組の平均）は、名詞刺激で62.8%（最高81.0%）、
画像刺激で51.3%（偶然水準に近い）でした。被験者は1名のため、
結果は一般化できません。詳細は Results / Limitations を参照してください。

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
  (artifact removal for EOG/ECG), epoching (images: 1 s = 500 samples;
  nouns: 500-sample windows, 0.25 s stride)
- Model: EEGNet (PyTorch)
- Training: Epoch 200, Batch size 32, LR 3e-4
- Train/test split: 6:1. For the combined image + noun tasks,
  300 training / 50 test samples per type.

## Results

All results are from a single subject (n = 1, self-recorded, 14 sessions).

| Task | Accuracy | Chance level |
|---|---|---|
| Image, 2-class (mean of 66 pairs) | 51.3% (best 75.0%: Jackie Chan vs. Watermelon; worst 20.5%) | 50% |
| Noun, 2-class (mean of 66 pairs) | 62.8% (best 81.0%: Stadium vs. Tokyo Tower; worst 32.5%) | 50% |
| Image, 4-category | 26.3% | 25% |
| Noun, 4-category | 40.7% | 25% |
| Image, 12-class | 11.2% | ~8.3% |
| Noun, 12-class | 18.5% | ~8.3% |
| Image + Noun, 24-class | 63.2% | ~4.2% |
| Image + Noun merged by class, 12-class | 63.7% | ~8.3% |

The "best" values are the maximum over 66 pairwise comparisons and are
not representative of typical performance.

In this single-subject setting, noun (text) stimuli were easier to
classify than image stimuli, whose accuracy is close to chance on average.

## Limitations

- Single subject; results do not generalize.
- Image 2-class accuracy is near chance on average, and 27 of 66 pairs
  are below 50%, which suggests unstable evaluation.
- Noun stimuli use overlapping windows (0.25 s stride). The unit of the
  train/test split is not documented here.
- The combined image + noun results (63.2%, 63.7%) used a different
  sampling scheme (300 training / 50 test samples per type) and are far
  higher than the single-modality results (11–41%). They are not directly
  comparable.

## Repository Structure

- `preprocessing.ipynb`: preprocessing notebook
- `eegnet_<input>_<output>_<scheme>.py` (10 scripts): training scripts
  - image_image: 2class / 3class / 4category / 12class
  - image_noun: 12class / 24class
  - noun_noun: 2class / 3class / 4category / 12class

## Tech Stack

- Python, PyTorch, NumPy, SciPy, MNE, Matplotlib, Pandas

## Data

EEG data is not publicly available.  
Thesis document available upon request.
