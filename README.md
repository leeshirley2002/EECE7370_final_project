# Deepfake Detection Project

A comprehensive deepfake detection system that uses multiple specialized models to detect manipulated faces in videos. This project implements a multi-region approach combining eye, nose, mouth, and full-face detection models with adaptive weight fusion.

## Overview

This project implements a deepfake detection pipeline that analyzes facial regions (eyes, nose, mouth) and full faces using specialized CNN and CNN-ViT hybrid models. The system combines predictions from multiple models using adaptive weights to achieve robust deepfake detection.

## Features

- **Multi-Region Analysis**: Separate models for eye, nose, mouth, and full-face regions
- **Hybrid Architecture**: Combines CNN and Vision Transformer (ViT) models
- **Adaptive Weight Fusion**: Learns optimal weights for combining model predictions
- **FaceForensics++ Dataset**: Compatible with the FaceForensics++ benchmark dataset
- **Comprehensive Preprocessing**: Frame extraction, face detection, and feature cropping

## Project Structure

```
.
├── data.py                              # Data preprocessing pipeline
├── download-faceforensics.py           # FaceForensics++ dataset downloader
├── FFData.py                           # Alternative dataset downloader
├── model_ab_paper.py                   # Model A and B training script
├── combined_model_adaptive_weights.py   # Combined model with adaptive weights
├── CNN_Deepfacke_Detection_Model_mouth.ipynb      # Mouth region model notebook
├── CNN_ViT_Hybrid_Deepfake_Detection_modelC.ipynb # Full face CNN-ViT hybrid notebook
├── Majority_Voting_Fusion_Model.ipynb  # Majority voting fusion approach
└── README.md                           # This file
```

## Models

### Model A (Eye & Nose Regions)
- **Architecture**: 12-layer CNN with Batch Normalization
- **Input Size**: 50×50×3
- **Purpose**: Detects deepfakes in eye and nose regions
- **Features**: Three convolutional blocks with BatchNorm, ReLU, MaxPool, and Dropout

### Model B (Mouth Region)
- **Architecture**: CNN Model A variant
- **Input Size**: 64×64×3
- **Purpose**: Detects deepfakes in mouth region
- **Features**: Multi-block CNN architecture with BatchNorm

### Model C (Full Face)
- **Architecture**: CNN-ViT Hybrid
- **Input Size**: 224×224×3
- **Purpose**: Full-face deepfake detection
- **Features**: 
  - CNN module for feature extraction
  - Vision Transformer for attention-based processing
  - Global average pooling and classification head

### Combined Model
- **Method**: Adaptive Weight Fusion
- **Approach**: Learns optimal weights for combining predictions from all four models
- **Benefits**: Automatically determines the importance of each region for final prediction

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space (for dataset)

### Required Packages

```bash
pip install torch torchvision
pip install opencv-python
pip install deepface
pip install dlib
pip install tqdm
pip install scikit-learn
pip install pillow
pip install numpy
```

### Additional Requirements

- **dlib shape predictor**: Download `shape_predictor_68_face_landmarks.dat` from [dlib models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project root directory.

## Usage

### 1. Download Dataset

Download the FaceForensics++ dataset:

```bash
# Download original videos
python download-faceforensics.py ./FFData -d original -c c23 -t videos --server EU2

# Download deepfake videos
python download-faceforensics.py ./FFData -d Deepfakes -c c23 -t videos --server EU2
```

**Note**: You must agree to the FaceForensics++ terms of use before downloading.

### 2. Preprocess Data

Extract frames, detect faces, and crop facial features:

```bash
python data.py
```

This script will:
- Extract frames from videos (every 30 frames)
- Detect and crop faces from frames
- Extract facial features (eyes, nose, mouth) using dlib landmarks
- Organize data into `Frames/`, `CroppedFaces/`, and `Features/` directories

### 3. Train Individual Models

#### Train Model A (Eye/Nose Regions)

```bash
# Train eye region model
python model_ab_paper.py --data exported_data --region eyes --model a --epochs 100

# Train nose region model
python model_ab_paper.py --data exported_data --region nose --model a --epochs 100
```

#### Train Model B (Mouth Region)

Use the `CNN_Deepfacke_Detection_Model_mouth.ipynb` notebook to train the mouth region model.

#### Train Model C (Full Face)

Use the `CNN_ViT_Hybrid_Deepfake_Detection_modelC.ipynb` notebook to train the full-face CNN-ViT hybrid model.

### 4. Train Combined Model

Use the `combined_model_adaptive_weights.py` script or the corresponding notebook to:
1. Load all four pre-trained models
2. Create a combined model with adaptive weights
3. Train the weight parameters on validation data
4. Evaluate the combined model performance

## Data Directory Structure

After preprocessing, your data should be organized as follows:

```
FFData/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
└── manipulated_sequences/
    └── Deepfakes/
        └── c23/
            └── videos/

Frames/
├── original/
└── manipulated/

CroppedFaces/
├── original/
└── manipulated/

Features/
├── original/
│   ├── leftEye/
│   ├── rightEye/
│   ├── nose/
│   └── mouth/
└── manipulated/
    ├── leftEye/
    ├── rightEye/
    ├── nose/
    └── mouth/
```

## Model Architecture Details

### Model A Architecture
```
Input (50×50×3)
  ↓
Block 1: Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout(0.3)
  ↓
Block 2: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout(0.3)
  ↓
Block 3: Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Dropout(0.3)
  ↓
Flatten → Linear(4608 → 512) → ReLU → Dropout(0.5) → Linear(512 → 2)
```

### Model C (CNN-ViT Hybrid) Architecture
```
Input (224×224×3)
  ↓
CNN Module: Conv → ReLU → Conv → ReLU → MaxPool
  ↓
Feature Maps (112×112×32)
  ↓
Patch Embedding → Position Embedding
  ↓
Transformer Blocks (6 layers, 8 heads, dim=1024)
  ↓
Global Average Pooling
  ↓
Classifier: Linear(1024 → 2)
```

## Training Configuration

### Model A/B Training
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Input Size**: 50×50×3

### Combined Model Training
- **Batch Size**: 16
- **Learning Rate**: 0.001 (for weight parameters only)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 20

## Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for fake detection
- **Recall**: Recall for fake detection
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Results

The combined model with adaptive weights typically achieves:
- Better performance than individual models
- Automatic weight optimization for different scenarios
- Robust detection across various deepfake generation methods

## Notes

- The preprocessing step requires significant computational resources and time
- GPU acceleration is highly recommended for training
- Model checkpoints should be saved regularly during training
- The adaptive weight approach allows the model to focus on the most discriminative regions

## Citation

If you use this code or dataset, please cite the FaceForensics++ paper:

```bibtex
@inproceedings{rossler2019faceforensics++,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## License

This project is for research purposes. Please ensure compliance with the FaceForensics++ dataset terms of use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository.
