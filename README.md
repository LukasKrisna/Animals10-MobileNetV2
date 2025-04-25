# Image Classification with MobileNetV2 (Animals-10)

A deep learning project for multi-class image classification using transfer learning with MobileNetV2.

## Overview

This project implements an image classifier that can identify 10 different classes: cow, elephant, cat, butterfly, squirrel, chicken, sheep, dog, horse, and spider. It uses transfer learning with a pre-trained MobileNetV2 model and adds custom layers to achieve high accuracy.
[Source](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

## Features

- Transfer learning with MobileNetV2
- Data preprocessing and augmentation
- Training with custom callbacks
- Model evaluation and visualization
- Model export in multiple formats (SavedModel, TF-Lite, TFJS)
- Simple inference code for predictions

## Project Structure

```
├── models/                            # Saved model files
│   ├── saved_model/                   # TensorFlow SavedModel format
│   │   ├── saved_model.pb
│   │   └── variables
│   ├── tflite/                        # TensorFlow Lite model
│   │   ├── model.tflite
│   │   └── label.txt
│   └── tfjs_model/                    # TensorFlow.js model files
│       ├── group1-shard1of4.bin
│       ├── group1-shard2of4.bin
│       ├── group1-shard3of4.bin
│       ├── group1-shard4of4.bin
│       └── model.json
├── notebook.ipynb                                          # Jupyter notebooks
├── submission_akhir_klasifikasi_gambar_lukas_krisna.py     # Python file
├── requirements.txt                                        # Project dependencies
└── README.md                                               # Project documentation
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```python
# Mount Google Drive and extract dataset
drive.mount('/content/drive')
# Extract zip file with dataset
# Run data preprocessing and model training
```

### Inference

To make predictions on new images:

```python
# Run
upload_and_predict()
```

## Model Architecture

- Base model: MobileNetV2 (pre-trained on ImageNet)
- Added layers:
  - Conv2D (128 filters, 3x3 kernel)
  - MaxPooling2D (2x2)
  - GlobalAveragePooling2D
  - Dense (256 units)
  - Dropout (0.5)
  - Dense output layer (10 classes)

## Performance

The model achieves over 95% accuracy on both training and validation sets.
