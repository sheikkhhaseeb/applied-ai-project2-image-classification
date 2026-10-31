# Applied AI Project 2: Image Classification Using Transfer Learning and Computer Vision

**Author:** Haseeb Bilal  
**Course:** Introduction to Applied AI  
**Project:** Project 2 — Neural Networks, Convolutions, Transfer Learning, and Computer Vision  
**Framework:** PyTorch  
**Model:** ResNet-18  
**Dataset:** Intel Image Classification Dataset  
**Platform:** Kaggle Notebooks

---

## 📌 Project Overview

This project explores deep learning and computer vision techniques for classifying natural and urban scene images into six categories using the Intel Image Classification dataset.

The project is organized into weekly sections, covering neural networks, convolutional neural networks (CNNs), transfer learning, dataset auditing, model evaluation, and model explainability.

**Week 5** focuses on CNN fundamentals, feature extraction, and fine-tuning a pretrained ResNet-18 model.

**Week 6** extends the work to applied computer vision, including dataset quality checks, duplicate detection, data leakage prevention, class imbalance analysis, model evaluation, and Grad-CAM visualization.

The project demonstrates the importance of both model performance and reliable evaluation when developing deep learning systems.

---

# Week 5: Neural Networks, Convolutions, and Transfer Learning

## 🎯 Objectives

- Understand the fundamentals of neural networks and convolutional neural networks.
- Explore convolution operations, strides, pooling, and parameter sharing.
- Prepare and split an image classification dataset.
- Apply transfer learning using a pretrained ResNet-18 model.
- Compare feature extraction with fine-tuning.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Save the trained model for future use.

## 📂 Dataset

The project uses the **Intel Image Classification Dataset**, which contains six scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

### Dataset Split

| Dataset Split | Number of Images |
|---|---:|
| Training | 11,929 |
| Validation | 2,105 |
| Testing | 3,000 |
| **Total** | **17,034** |

Each image belongs to one of the six scene categories.

## 🧠 Methodology

### 1. Dataset Exploration and Preprocessing

The dataset was explored to understand its structure, image dimensions, data types, and class distribution. Images were prepared for input into the pretrained ResNet-18 model.

### 2. Convolutional Neural Networks

Convolution operations were studied to understand how filters extract visual features from images. The experiments covered convolutional output dimensions, strides, pooling, parameter counts, and weight sharing.

### 3. Transfer Learning: Feature Extraction

A pretrained ResNet-18 model was used as a feature extractor. The pretrained layers were frozen, and the final classification layer was adapted to predict the six target classes.

- **Trainable parameters:** 3,078
- **Total parameters:** 11,179,590
- **Best validation accuracy:** 90.1%

### 4. Fine-Tuning

The final convolutional block (`layer4`) was unfrozen so that the model could adapt its learned features to the Intel Image Classification dataset.

- **Trainable parameters after unfreezing `layer4`:** 8,396,806
- **Best validation accuracy:** 93.8%
- **Improvement over feature extraction:** 3.7 percentage points

### 5. Model Evaluation

The fine-tuned model was evaluated on the held-out test dataset using accuracy, precision, recall, and F1-score.

## 📊 Week 5 Results

| Evaluation Metric | Result |
|---|---:|
| Feature extraction — best validation accuracy | 90.1% |
| Fine-tuning — best validation accuracy | 93.8% |
| Final test accuracy | 93.5% |
| Macro-average F1-score | 93.6% |

Fine-tuning improved validation accuracy from 90.1% to 93.8%, demonstrating the potential benefit of adapting pretrained convolutional features to a target image classification task.

### Class-Wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Buildings | 92.8% | 93.8% | 93.3% |
| Forest | 99.4% | 99.4% | 99.4% |
| Glacier | 89.4% | 88.8% | 89.1% |
| Mountain | 90.8% | 88.6% | 89.7% |
| Sea | 94.5% | 98.4% | 96.4% |
| Street | 94.7% | 93.0% | 93.9% |

**Best-performing class:** Forest, with a reported recall of 99.4%.

**Area for improvement:** Mountain and glacier had lower recall than the other classes, suggesting that visually similar landscape features may require further investigation.

## 💾 Saved Model

The trained model was saved as:

`resnet18_intel.pt`

Expected Kaggle output location:

`/kaggle/working/resnet18_intel.pt`

The model file can be downloaded from Kaggle and loaded in a compatible PyTorch environment using the same model architecture and preprocessing configuration.

---

# Week 6: Applied Computer Vision — Dataset Audit, Metrics, and Model Explainability

## 🎯 Objectives

- Audit the dataset and inspect class distribution.
- Examine image dimensions, colour modes, and representative samples.
- Investigate exact and near-duplicate images.
- Reduce the risk of data leakage during training and validation.
- Examine class imbalance and calculate imbalance ratios mathematically.
- Train and fine-tune a ResNet-18 model for six-class image classification.
- Evaluate performance using core classification mathematical formulas (Accuracy, Precision, Recall, Macro-F1).
- Apply Grad-CAM to investigate the visual regions influencing model predictions.

## 📂 Dataset

Week 6 uses the Intel Image Classification dataset containing the following six classes:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

### Dataset Audit & Imbalance Calculations

The initial dataset audit produced the following sample distribution across the training classes:

| Dataset Detail | Count / Value |
|---|---:|
| Training images | 14,034 |
| Test images | 3,000 |
| Number of classes | 6 |
| Buildings — training images | 2,191 |
| Forest — training images | 2,271 |
| Glacier — training images | 2,404 |
| Mountain — training images | 2,512 |
| Sea — training images | 2,274 |
| Street — training images | 2,382 |

#### 📐 Mathematical Formulation of Class Imbalance Ratio
The imbalance ratio quantifies the skewness between the most frequent class and the least frequent class in the dataset:

$$\text{Imbalance Ratio} = \frac{\max(N_c)}{\min(N_c)}$$

Where $N_c$ represents the number of training samples in class $c$. Substituting the dataset counts:

$$\text{Imbalance Ratio} = \frac{2,512}{2,191} \approx 1.147$$

This low ratio ($1.147$) indicates that the six training classes are remarkably well-balanced, minimizing major classification biases.

---

## 📐 Mathematical Evaluation Formulas

To measure model performance rigorously during Week 6, standard evaluation metrics are computed using Confusion Matrix components: True Positives ($TP$), True Negatives ($TN$), False Positives ($FP$), and False Negatives ($FN$).

### 1. Accuracy
Measures the overall fraction of correctly predicted samples out of total samples:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 2. Precision
Measures the proportion of positive identifications that were actually correct:

$$\text{Precision} = \frac{TP}{TP + FP}$$

### 3. Recall (Sensitivity)
Measures the proportion of actual positives that were identified correctly:

$$\text{Recall} = \frac{TP}{TP + FN}$$

### 4. F1-Score
The harmonic mean of Precision and Recall, providing a balanced performance metric:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 5. Macro-Averaged F1-Score
Calculates the unweighted average of the F1-scores across all $C$ classes, ensuring minority classes carry equal weight:

$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1-Score}_c$$

---

## 🧠 Methodology & Explainability

### 1. Dataset Indexing and Quality Checks
The dataset was indexed to map image paths and labels. Exact duplicates were filtered via hashing algorithms, while near-duplicates were assessed using structural similarity metrics to prevent data leakage between training and validation splits.

### 2. ResNet-18 Training & Fine-Tuning
A pretrained ResNet-18 was adapted for the 6-class target. The classification head was initialized using a linear transformation:

$$\text{Linear}(512, 6) \implies \text{Trainable Parameters} = (512 \times 6) + 6 = 3,078$$

### 3. Grad-CAM (Gradient-weighted Class Activation Mapping)
To interpret model decisions and check for shortcut learning, Grad-CAM computes gradients of the score for class $c$ ($y^c$) with respect to the feature map activations $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

The localization map is then generated as a weighted combination of forward activation maps followed by a ReLU activation:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

This highlights exact image regions driving predictions (e.g., distinguishing mountain peaks from sky backgrounds).

---

## 📊 Week 6 Results & Summary

| Evaluation Item | Result |
|---|---|
| Training images | 14,034 |
| Test images | 3,000 |
| Number of classes | 6 |
| Class imbalance ratio | 1.147 |
| Final test accuracy | 93.5% |
| Macro-F1 score | 93.6% |
| Key takeaway | Grad-CAM confirmed models focused on structural semantic elements (forest canopies, architectural edges) rather than background noise. |

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL (Pillow)
- Kaggle Notebooks
- Convolutional Neural Networks
- Transfer Learning
- ResNet-18
- Grad-CAM

# 📁 Repository Structure

```text
applied-ai-project2-image-classification/
│
├── week5-neural-networks-transfer-learning.ipynb
├── week6-computer-vision.ipynb
├── README.md
└── resnet18_intel.pt  (optional)
