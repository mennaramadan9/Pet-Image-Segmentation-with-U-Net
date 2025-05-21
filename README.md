# Pet Image Segmentation with U-Net

A deep learning project for semantic segmentation of pets using the U-Net architecture on the Oxford-IIIT Pet Dataset.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🐕 Project Overview

This project implements a U-Net convolutional neural network to perform semantic segmentation on pet images, distinguishing between pets, backgrounds, and outlines. The model is trained on the Oxford-IIIT Pet Dataset and achieves ~85% validation accuracy.

## 📊 Dataset

- **Source**: Oxford-IIIT Pet Dataset (via TensorFlow Datasets)
- **Size**: 7,349 images across 37 pet breeds
- **Classes**: 3 (pet, background, outline)
- **Image Size**: Resized to 128×128 pixels
- **Split**: Training and testing sets provided by TFDS

## 🏗️ Architecture

The model uses the U-Net architecture, which consists of:

- **Encoder Path**: 4 downsampling blocks (64→128→256→512 filters)
- **Bottleneck**: 1024 filters
- **Decoder Path**: 4 upsampling blocks (512→256→128→64 filters)
- **Skip Connections**: Preserve spatial information
- **Output**: 3-channel softmax for pixel-wise classification

### Model Summary
- **Total Parameters**: ~31 million
- **Input Shape**: (128, 128, 3)
- **Output Shape**: (128, 128, 3)

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow
pip install tensorflow-datasets
pip install matplotlib
pip install numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pet-segmentation-unet.git
cd pet-segmentation-unet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. **Run the complete pipeline**:
```bash
python image_segmentation.py
```

2. **Key functions**:
   - `normalize()`: Normalizes images and masks
   - `load_train_images()`: Preprocesses training data with augmentation
   - `build_unet_model()`: Creates the U-Net architecture
   - `display_sample()`: Visualizes results

## 📈 Results

### Training Metrics
- **Training Accuracy**: ~88%
- **Validation Accuracy**: ~87%
- **Training Time**: 20 epochs
- **Loss Function**: Sparse Categorical Crossentropy

### Performance
- Effective segmentation across various pet breeds
- Good boundary detection in most cases
- Some challenges with complex backgrounds

## 🖼️ Sample Results

The model successfully segments pets from backgrounds:

| Input Image | Ground Truth | Prediction |
|-------------|--------------|------------|
| Original pet photo | Manual segmentation mask | Model prediction |

## 🔧 Key Features

- **Data Augmentation**: Random horizontal flipping
- **Regularization**: 30% dropout in encoder/decoder blocks
- **Optimization**: Adam optimizer
- **Preprocessing**: Image normalization and resizing
- **Visualization**: Built-in result display functions

## 📋 Project Structure

```
pet-segmentation-unet/
│
├── image_segmentation.ipynb 
├── README.md               
├── image_segmentation.pptx
```

## 🛠️ Implementation Details

### Data Preprocessing
```python
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = input_mask - 1  # Convert to zero-based indexing
    return input_image, input_mask
```

### U-Net Building Blocks
- **Double Convolution**: Two 3×3 conv layers with ReLU
- **Downsampling**: MaxPooling + Dropout
- **Upsampling**: Conv2DTranspose + Skip connections

### Training Configuration
- **Batch Size**: 64
- **Epochs**: 20
- **Buffer Size**: 1000 (for shuffling)
- **Prefetching**: Enabled for performance

---

⭐ **Found this project helpful? Please consider giving it a star!** ⭐
