# EEG-classifier

## EEG Classification with Contrastive Learning and Sub-center CosFace

This repository implements an EEG classification pipeline using contrastive learning and a Conformer-based neural network architecture.
The model learns a shared representation between EEG signals and visual features extracted from images using EVA-CLIP.

The overall workflow includes:

- EEG signal analysis
- Image feature extraction
- Contrastive representation learning
- Linear probing evaluation
- Final classification with Sub-center CosFace
- Test-time inference and submission generation

The goal is to improve EEG classification performance by aligning EEG representations with semantic image features.

## Overview

The pipeline follows these steps:

1. EEG data exploration and visualization
2. Image feature extraction using EVA-CLIP
3. Contrastive learning between EEG and image embeddings
4. Linear probing to evaluate learned representations
5. Final classification using Sub-center CosFace
6. Test-time inference and submission generation

## Repository Structure
```
eeg-classifier/
│
├── analysis/
│   ├── eeg_visualization.py
│   ├── image_feature_extraction.py
│   └── multimodality_analysis.py
│
├── model/
│   ├── eeg_conformer.py
│   └── subcenter_cosface.py
│
├── training/
│   ├── train_contrastive.py
│   ├── linear_probe.py
│   └── train_classifier.py
│
├── inference/
│   └── predict.py
│
│
├── README.md
└── LICENSE
```

Directory description

**analysis/**

Scripts for EEG data visualization, feature analysis, and multimodality checks.

**model/**

Neural network architecture definitions including the EEG Conformer and Sub-center CosFace classifier.

**training/**

Training scripts for contrastive learning, linear probing, and final classifier training.

**inference/**

Inference script for generating predictions and submission files.


## Model Architecture

The model processes EEG signals through the following pipeline:

1. Channel weighting
2. Linear projection
3. Subject-specific adaptation (SubjectBlock)
4. Subject embedding
5. Positional encoding
6. Conformer encoder blocks
7. Attention pooling
8. EEG-to-CLIP feature projection
9. Sub-center CosFace classifier

This architecture captures both:

- temporal EEG patterns
- subject-specific variability

## EEG Channels

The model uses the following 17 EEG channels:
```
['Pz','P3','P7','O1','Oz','O2','P4','P8','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2']
```
Channels located in occipital and parieto-occipital regions are given higher weights because they are strongly related to visual processing.

## Installation

Example dependencies:
```
pip install torch torchvision torchaudio
pip install transformers
pip install open_clip_torch timm
pip install scikit-learn scipy tqdm pillow matplotlib numpy
```
The project is primarily designed to run on Google Colab.


## Dataset

The dataset used in this project is not included in this repository.  
The dataset cannot be redistributed due to licensing restrictions.

If you want to run the code with your own dataset, please preprocess your data into the following format:
```
data/
├── train/
├── val/
└── test/
```

## Data Format
EEG data
```
shape: (N, C, T)
```
- N: number of samples
- C: number of channels (17)
- T: number of time steps

Labels
```
shape: (N,)
```
Subject IDs
```
shape: (N,)
```

Example:

train_eeg.npy

train_labels.npy

train_subject_idxs.npy

## Image Feature Extraction

Image embeddings are extracted using EVA-CLIP.

Example:
```
train_feats, train_paths = extract_eva_clip_features(train_paths)
val_feats, val_paths = extract_eva_clip_features(val_paths)
```
These features serve as supervision signals for contrastive learning.

## Contrastive Learning

EEG features and image features are aligned using an InfoNCE-style contrastive loss.

The model learns a representation where corresponding EEG and image samples are close in embedding space.

Output checkpoint:
```
best_contrastive.pt
```
## Linear Probing

After contrastive training, the backbone is frozen, and a linear classifier is trained on top of the learned embeddings.

This step evaluates the quality of the learned EEG representation.

Output checkpoint:
```
best_linear_probe.pt
```
## Final Classification

The final classifier uses Sub-center CosFace to handle multimodal distributions within classes.
Each class is represented by multiple centers in the embedding space, improving robustness to:

- subject variation
- intra-class diversity

Output checkpoint:
```
best_subcosface_eeg2clip_K6.pt
```

## Inference

Test predictions are generated as logits.
```
submission.npy
```
The repository can also generate a zipped submission file containing:

- model weights
- submission file
- notebook

## Key Components
**SubjectBlock**
Handles subject-specific variations by applying a subject-dependent transformation before the Conformer encoder.

**Attention Pooling**
Aggregates temporal EEG features using learned attention weights.

**Conformer Encoder**
Combines:

- feed-forward layers
- multi-head self-attention
- convolution modules

This architecture captures both long-range dependencies and local temporal patterns.

**Sub-center CosFace**
Extends CosFace by assigning multiple centers per class, allowing better modeling of intra-class variability.

## Training Strategy
**Contrastive Pretraining**
- Optimizer: AdamW
- Scheduler: cosine decay with warmup
- Loss: contrastive InfoNCE

**Classification Training**
- Backbone mostly frozen
- EEG-to-CLIP projection optionally unfrozen
- Sub-center CosFace classifier trained

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

- Designed for execution on Google Colab
- Paths may require adjustment depending on environment
- Dataset is not included in this repository

## Future Work
Possible extensions:
- stronger subject-invariant learning
- EEG data augmentation
- joint contrastive and classification training
- improved temporal feature selection
