# eeg-classifier

## EEG Classification with Contrastive Learning and Sub-center CosFace

This repository implements an EEG classification pipeline using contrastive learning and a Conformer-based architecture. The model learns a shared representation between EEG signals and visual features extracted from images using EVA-CLIP.

The workflow includes:
-EEG signal analysis
-Image feature extraction
-Contrastive representation learning
-Linear probing evaluation
-Final classification with Sub-center CosFace
-Inference and submission generation

The goal is to improve EEG classification performance by aligning EEG representations with semantic image features.

## Overview

The pipeline follows these steps:
1.EEG data exploration and visualization
2.Image feature extraction using EVA-CLIP
3.Contrastive learning between EEG and image embeddings
4.Linear probing to evaluate learned representations
5.Final classification using Sub-center CosFace
6.Test-time inference and submission generation

## Model Architecture

The model processes EEG signals in the following way:
1.Channel weighting
2.Linear projection
3.Subject-specific adaptation (SubjectBlock)
4.Subject embedding
5.Positional encoding
6.Conformer encoder blocks
7.Attention pooling
8.EEG-to-CLIP feature projection
9.Sub-center CosFace classifier

The architecture is designed to capture both temporal EEG patterns and subject-specific variability.

## EEG Channels

The model uses 17 EEG channels:

```
['Pz','P3','P7','O1','Oz','O2','P4','P8','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2']
```
Channels in the occipital and parieto-occipital regions are given higher importance because they are more relevant for visual processing tasks.

## Installation

Example dependencies:
```
pip install japanize-matplotlib
pip install transformers
pip install open_clip_torch timm
pip install torch torchvision torchaudio
```
The code is designed to run on Google Colab.

## Dataset Structure

Expected directory structure:
```
EEG/
 ├── data/
 │   ├── train/
 │   │   ├── eeg.npy
 │   │   ├── labels.npy
 │   │   ├── subject_idxs.npy
 │   │   ├── image_paths.txt
 │   │   └── train_eva_feats.npy
 │   ├── val/
 │   │   ├── eeg.npy
 │   │   ├── labels.npy
 │   │   ├── subject_idxs.npy
 │   │   └── val_eva_feats.npy
 │   └── test/
 │       ├── eeg.npy
 │       └── subject_idxs.npy
 ├── last_exam2.ipynb
 └── README.md
```

## Data Format

EEG data
```
shape: (N, C, T)
```
-N: number of samples
-C: number of channels (17)
-T: number of time steps

Labels
```
shape: (N,)
```
Subject IDs
```
shape: (N,)
```
## Image Feature Extraction

Image embeddings are extracted using EVA-CLIP.

Example:
```
train_feats, train_paths = extract_eva_clip_features(train_paths)
val_feats, val_paths = extract_eva_clip_features(val_paths)
```
These features are used as supervision signals during contrastive learning.

## Contrastive Learning

EEG features and image features are aligned using an InfoNCE-style contrastive loss.

The model learns a representation where corresponding EEG and image samples are close in embedding space.

Output checkpoint:
```
best_contrastive.pt
```
## Linear Probing

After contrastive training, the backbone is frozen and a linear classifier is trained on top of the learned embeddings.

This step evaluates the quality of the learned EEG representation.

Output checkpoint:
```
best_linear_probe.pt
```
## Final Classification

The final classifier uses Sub-center CosFace to handle multimodal distributions within classes.

Each class is represented by multiple centers in the embedding space, which improves robustness to subject variation.

Output checkpoint:
```
best_subcosface_eeg2clip_K6.pt
```
## Inference

Test predictions are generated as logits.

submission.npy

The repository also generates a zipped submission file containing:

-model weights
-submission file
-notebook

## Key Components

SubjectBlock

Handles subject-specific variations by applying a subject-dependent transformation before the Conformer encoder.

Attention Pooling

Aggregates temporal EEG features using learned attention weights.

Conformer Encoder

Combines:
-feed-forward layers
-multi-head self-attention
-convolution modules

This architecture captures both long-range dependencies and local temporal patterns.

Sub-center CosFace

Extends CosFace by assigning multiple centers to each class, allowing better modeling of intra-class variability.

## Training Strategy

Contrastive Pretraining
-Optimizer: AdamW
-Scheduler: cosine decay with warmup
-Loss: contrastive InfoNCE

Classification Training
-Backbone mostly frozen
-EEG-to-CLIP projection optionally unfrozen
-Sub-center CosFace classifier trained

## Outputs

Training produces:
```
best_contrastive.pt
best_linear_probe.pt
best_subcosface_eeg2clip_K6.pt
submission.npy
submission_last_exam2.zip
```
## Notes

-Designed for execution on Google Colab
-Paths may need adjustment depending on environment
-Dataset is not included in this repository

## Future Work

Possible extensions:
-stronger subject-invariant learning
-EEG data augmentation
-joint contrastive and classification training
-improved temporal feature selection
