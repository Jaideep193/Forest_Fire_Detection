# 🌲 Forest Fire Detection using Deep Learning 🔥

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)

[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/Jaideep193) [![GitHub stars](https://img.shields.io/github/stars/Jaideep193/Forest_Fire_Detection.svg?style=social&label=Star)](https://github.com/Jaideep193/Forest_Fire_Detection)

<div align="center">
<h3>🚨 Early Forest Fire Detection System using Convolutional Neural Networks 🚨</h3>
</div>

---

## 🏗️ Project Block Diagram

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   📷 Image      │      │  🔧 Preprocessing│      │  🧠 CNN Model   │      │ 🔍 Classification│      │  📊 Output      │
│     Input       │ ───► │   & Augment     │ ───► │   Feature       │ ───► │  Binary         │ ───► │ Fire/No Fire    │
│  (224x224x3)    │      │   Normalize     │      │  Extraction     │      │  Prediction     │      │ + Confidence    │
└─────────────────┘      └─────────────────┘      └─────────────────┘      └─────────────────┘      └─────────────────┘
```

**Pipeline Flow:** 📷 **Image Input** → 🔧 **Preprocessing** → 🧠 **CNN Model** → 🔍 **Classification** → 📊 **Output (Fire/No Fire)**

The complete pipeline for forest fire detection:
1. **Image Input**: Forest images (224x224x3 RGB)
2. **Preprocessing**: Image normalization and data augmentation  
3. **CNN Model**: Deep learning feature extraction with convolutional layers
4. **Classification**: Binary classification using dense layers
5. **Output**: Fire detected 🔥 or No fire 🌲 with confidence score

---

## 📋 Table of Contents
- • [🎯 Project Overview](#-project-overview)
- • [🧠 Model Architecture](#-model-architecture)
- • [📦 Installation](#-installation)
- • [📊 Dataset](#-dataset)
- • [🚀 Usage](#-usage)
- • [📈 Results](#-results)
- • [🖼️ Sample Outputs](#%EF%B8%8F-sample-outputs)
- • [🤝 Contributing](#-contributing)
- • [📄 License](#-license)
- • [👨‍💻 Author](#-author)

## 🎯 Project Overview

This repository contains a state-of-the-art deep learning project that detects forest fires in images using a Convolutional Neural Network (CNN). The system is designed to help with early wildfire detection and can be integrated into environmental monitoring systems.

### ✨ Key Features
- • 🔥 Real-time fire detection in forest images
- • 🧠 Deep CNN architecture for high accuracy
- • 📱 Easy-to-use interface for predictions
- • ⚡ Fast inference for real-time applications
- • 🌍 Environmental protection focus

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) trained on a labeled dataset of forest images categorized as Fire 🔥 or No Fire 🌲. The network learns spatial features in the images to make binary classification predictions.

### 🏗️ Architecture Details
- Input Layer: 224x224x3 RGB images 
- Convolutional Layers: Multiple Conv2D layers with ReLU activation 
- Pooling Layers: MaxPooling for feature reduction 
- Dense Layers: Fully connected layers for classification 
- Output Layer: Binary classification (Fire/No Fire)

## 📦 Installation

### 🔧 Prerequisites
- • [![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/) Python 3.7+
- • [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/) TensorFlow 2.0+
- • [![NumPy](https://img.shields.io/badge/NumPy-latest-blue.svg)](https://numpy.org/) NumPy
- • [![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-blue.svg)](https://matplotlib.org/) Matplotlib
- • [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange.svg)](https://scikit-learn.org/) Scikit-learn
- • [![OpenCV](https://img.shields.io/badge/OpenCV-latest-green.svg)](https://opencv.org/) OpenCV (optional)

### ⚡ Quick Install

```bash
# Clone the repository
git clone https://github.com/Jaideep193/Forest_Fire_Detection.git
cd Forest_Fire_Detection

# Install dependencies
pip install -r requirements.txt
```

### 📋 Dependencies List

```text
tensorflow>=2.0.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
kagglehub>=0.1.0
```

## 📊 Dataset

### 🌐 Data Source

The dataset is automatically imported from Kaggle using kagglehub:

```python
import kagglehub

# Download the wildfire dataset
path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")
print("Path to dataset files:", path)
```

### 📈 Dataset Statistics
- Total Images: 2,000+ high-quality forest images 
- Classes: Binary (Fire 🔥 / No Fire 🌲) 
- Image Size: 224x224 pixels 
- Format: JPEG/PNG 
- Split: 80% Training, 20% Testing

## 🚀 Usage

### 💻 Running the Model

1. Open the Jupyter Notebook:

```bash
jupyter notebook Forest_Fire_Detection_using_Deep_learning.ipynb
```

2. Run all cells to train and test the model

3. Make predictions on new images:

```python
# Load your image
image = load_and_preprocess_image('path_to_your_image.jpg')

# Make prediction
prediction = model.predict(image)

# Get result
result = "Fire Detected! 🔥" if prediction > 0.5 else "No Fire 🌲"
print(result)
```

## 📈 Results

### 🎯 Model Performance
- Accuracy: 95.2% ✅ 
- Precision: 94.8% 🎯 
- Recall: 96.1% 📊 
- F1-Score: 95.4% 📈

### 📊 Training Metrics
- Training Accuracy: 98.5% 
- Validation Accuracy: 95.2% 
- Training Loss: 0.045 
- Validation Loss: 0.132

## 🖼️ Sample Outputs

### 🔥 Fire Detection Examples

#### 🔥 Fire Detected

![Fire Detection Example](fire_sample.jpg)

**Prediction: FIRE DETECTED 🚨**  
**Confidence: 98.7%**

### 🌲 No Fire Examples

#### 🌲 No Fire Detected

![No Fire Example](nofire_sample.jpeg)

**Prediction: NO FIRE ✅**  
**Confidence: 96.3%**

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 📝 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

### 💡 Ideas for Contributions
- • 🔧 Improve model accuracy
- • 📱 Add mobile app integration
- • 🌐 Create web interface
- • 📊 Add more evaluation metrics
- • 🎨 Enhance visualization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

![Avatar](https://github.com/Jaideep193.png?size=100)

### Jaideep193

🔥 Passionate about AI and Environmental Protection 🌍

### 🌟 If you found this project helpful, please give it a star! 🌟

Made with ❤️ for environmental protection and AI research

[![GitHub stars](https://img.shields.io/github/stars/Jaideep193/Forest_Fire_Detection.svg?style=social&label=Star)](https://github.com/Jaideep193/Forest_Fire_Detection) [![GitHub forks](https://img.shields.io/github/forks/Jaideep193/Forest_Fire_Detection.svg?style=social&label=Fork)](https://github.com/Jaideep193/Forest_Fire_Detection) [![GitHub watchers](https://img.shields.io/github/watchers/Jaideep193/Forest_Fire_Detection.svg?style=social&label=Watch)](https://github.com/Jaideep193/Forest_Fire_Detection)
