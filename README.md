# Emotion and Action Prediction using Deep Learning

## Introduction

This project classifies emotions and predicts human actions using deep convolutional neural networks (CNNs) and MediaPipe pose dynamics.

## Features

- **Emotion Detection**: Classifies facial expressions into seven emotions.
- **Action Prediction Engine**: Classifies human actions using keypoint dynamics and joint angle math.
- **Multimodal Tracking**: Integrates facial emotions with pose actions real-time.

## Directory Structure
```bash
Emotify/
│── .gitignore
│── README.md
│── requirements.txt
│── imgs/
│── src/
│   │── data/
│   │── action_classifier.py  # Pose keypoint dynamics & joint angle action classifier
│   │── action.py  # Real-time action prediction engine with HUD display
│   │── action_mapping.py  # Bidirectional action-emotion mapping and correlation
│   │── emotions.py  # Multimodal emotion & action tracking script
│   │── dataset_prepare.py  # FER2013 CSV data preprocessing
│   │── load_mpii.py  # MPII dataset annotations parser & summary exporter
│   │── actions.txt  # Cleaned MPII action taxonomy list
│   │── haarcascade_frontalface_default.xml  # Face cascade classifier
│   │── model.h5  # Pre-trained CNN weights
│   │── mpii_human_pose_v1_u12_1.mat
```

## Basic Usage

```bash
git clone https://github.com/miracneroid/Emotify.git
cd Emotify
```

* **Run Real-Time Action Prediction HUD**:
```bash
cd src
python action.py --source 0
```

* **Run Multimodal Tracking**:
```bash
cd src
python emotions.py --mode display
```
