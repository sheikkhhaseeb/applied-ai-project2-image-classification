# Applied AI Project 2: Image Classification Using Transfer Learning

**Author:** Haseeb Bilal  
**Course:** Introduction to Applied AI  
**Project:** Neural Networks, Convolutions, and Transfer Learning  
**Framework:** PyTorch  
**Model:** ResNet18  
**Dataset:** Intel Image Classification Dataset

## 📌 Project Overview

This project explores neural networks, convolutional neural networks (CNNs), and transfer learning for image classification. A pretrained ResNet18 model was trained to classify natural and urban scene images into six categories.

The project demonstrates how pretrained models can be adapted to a new classification task using feature extraction and fine-tuning.

## 🎯 Objectives

- Understand the fundamentals of convolutional neural networks.
- Explore convolution operations, strides, pooling, and parameter sharing.
- Prepare and split an image classification dataset.
- Apply transfer learning using a pretrained ResNet18 model.
- Compare feature extraction with fine-tuning.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Save the trained model for future use.

## 📂 Dataset

The project uses the **Intel Image Classification Dataset**, containing six scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Dataset split used in this project:

| Dataset Split | Number of Images |
|---|---:|
| Training | 11,929 |
| Validation | 2,105 |
| Testing | 3,000 |

Each image represents one of the six scene categories.

## 🧠 Methodology

### 1. Dataset Exploration and Preprocessing

The dataset was explored to understand its structure, image dimensions, data types, and class distribution. Images were prepared for input into the pretrained ResNet18 model.

### 2. Convolutional Neural Networks

Convolution operations were studied to understand how filters extract features from images. The experiments covered convolutional output dimensions, strides, pooling, parameter counts, and weight sharing.

### 3. Transfer Learning: Feature Extraction

A pretrained ResNet18 model was used as a feature extractor. The pretrained layers were frozen, and the final classification layer was adapted for the six target classes.

- Trainable parameters: **3,078**
- Total parameters: **11,179,590**
- Best validation accuracy: **90.1%**

### 4. Fine-Tuning

The final convolutional block (`layer4`) was unfrozen to allow the model to adapt its learned features to the Intel Image Classification dataset.

- Trainable parameters after unfreezing `layer4`: **8,396,806**
- Best validation accuracy: **93.8%**
- Improvement over feature extraction: **3.7 percentage points**

### 5. Model Evaluation

The fine-tuned model was evaluated on the held-out test dataset using accuracy, precision, recall, and F1-score.

## 📊 Results

| Evaluation Metric | Result |
|---|---:|
| Feature extraction — best validation accuracy | 90.1% |
| Fine-tuning — best validation accuracy | 93.8% |
| Final test accuracy | 93.5% |
| Macro-average F1-score | 93.6% |

Fine-tuning improved validation accuracy from 90.1% to 93.8%, showing that adapting pretrained convolutional features can improve performance on a target image classification task.

### Class-Wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Buildings | 92.8% | 93.8% | 93.3% |
| Forest | 99.4% | 99.4% | 99.4% |
| Glacier | 89.4% | 88.8% | 89.1% |
| Mountain | 90.8% | 88.6% | 89.7% |
| Sea | 94.5% | 98.4% | 96.4% |
| Street | 94.7% | 93.0% | 93.9% |

**Best-performing class:** Forest, with 99.4% recall.

**Area for improvement:** Mountain and glacier had lower recall than the other classes, indicating that these categories deserve further investigation.

## 💾 Saved Model

The trained model was saved as:

`resnet18_intel.pt`

In the Kaggle environment, the file was saved to:

`/kaggle/working/resnet18_intel.pt`

To use the model outside Kaggle, download the saved file and load it using a compatible PyTorch environment and the same model architecture.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Notebooks
- ResNet18
- Convolutional Neural Networks
- Transfer Learning

## 📁 Repository Structure

```text
applied-ai-project2-image-classification/
│
├── week5-neural-networks-transfer-learning.ipynb
├── README.md
└── resnet18_intel.pt  (optional: add the trained model)
```

The trained model file is optional. If you upload it, check GitHub's file-size limits first; alternatively, provide instructions for downloading it from Kaggle.

## 🔍 Key Learnings

- Convolutional layers learn spatial features from images.
- Weight sharing reduces the number of parameters compared with fully connected operations over image regions.
- Transfer learning makes it possible to reuse features learned by a pretrained network.
- Feature extraction is computationally lighter because most pretrained parameters remain frozen.
- Fine-tuning can improve performance by adapting selected pretrained layers.
- Accuracy alone does not describe performance on every class; precision, recall, and F1-score provide additional insight.

## 🚀 Future Improvements

- Investigate misclassified images using the confusion matrix.
- Improve performance on visually similar categories, particularly mountains and glaciers.
- Experiment with data augmentation and learning-rate scheduling.
- Compare ResNet18 with other pretrained architectures.
- Test the model on new images outside the dataset.
- Explore a future project on electronics component classification.

## ✅ Conclusion

This project demonstrated image classification using a pretrained ResNet18 model and PyTorch. The final model achieved **93.5% test accuracy**, while fine-tuning improved the best validation accuracy to **93.8%**.

The experiments provided practical experience with CNN fundamentals, feature extraction, fine-tuning, and class-wise model evaluation.

---

**Project:** Applied AI Project 2 — Image Classification  
**Author:** Haseeb Bilal  
**Course:** Introduction to Applied AI
