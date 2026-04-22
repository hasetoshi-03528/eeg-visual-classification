# eeg-visual-classification

EEG-based visual stimulus classification using EEGNet (PyTorch)  
Graduation thesis project — Tongji University, 2023

## Overview

This project implements EEG signal classification for a non-invasive 
Brain-Computer Interface (BCI) system. EEG data was recorded from a 
single subject while viewing visual stimuli (images and text), then 
classified using EEGNet, a compact CNN designed for EEG-based BCI tasks.

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

## Results (highlights)

| Task | Accuracy |
|------|----------|
| 2-class image classification | up to 81.0% (best pair) |
| 4-class (category-level) image | 26.3% |
| 2-class noun classification | up to 81.0% (best pair) |
| 4-class (category-level) noun | 40.7% |
| 24-class image+noun combined | 63.2% |

## Tech Stack

- Python, PyTorch, NumPy, SciPy, MNE, Matplotlib

## Data

EEG data is not publicly available.  
Thesis document available upon request.
