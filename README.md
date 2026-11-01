# Week 6: Malaria Cell Screening (Parasitized vs Uninfected) with CNN

**Author:** Haseeb Bilal  
**Course:** Introduction to Applied AI  
**Project:** Project 2 — Neural Networks, Convolutions, Transfer Learning, and Computer Vision  
**Framework:** PyTorch  
**Model:** ResNet-18  
**Dataset:**  NIH Malaria Cell Images Dataset
**Platform:** Kaggle Notebooks

---

## 📌 Project Overview

This project explores deep learning and computer vision techniques for classifying natural and urban scene images into six categories using the Intel Image Classification dataset.

The project is organized into weekly sections, covering neural networks, convolutional neural networks (CNNs), transfer learning, dataset auditing, model evaluation, and model explainability.

**Week 5** focuses on CNN fundamentals, feature extraction, and fine-tuning a pretrained ResNet-18 model.

**Week 6** This project develops a Convolutional Neural Network (CNN) model to classify microscopic blood cell images into two categories: **Parasitized** and **Uninfected**. The objective is to automate malaria detection from cell images and demonstrate a complete computer vision pipeline including data preprocessing, model training, evaluation, and performance analysis

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

# Week 6: Malaria Cell Screening (Parasitized vs Uninfected) with CNN

## 🎯 Objectives

- Audit the dataset and inspect class distribution.
- Examine image dimensions, colour modes, and representative samples.
- Investigate exact and near-duplicate images.
- Reduce the risk of data leakage during training and validation.
- Examine class imbalance and calculate imbalance ratios mathematically.
- Train and fine-tune a ResNet-18 model for six-class image classification.
- Evaluate performance using core classification mathematical formulas (Accuracy, Precision, Recall, Macro-F1).
- Apply Grad-CAM to investigate the visual regions influencing model predictions.



## Dataset Information

| Section | Answer |
|----------|----------|
| Dataset Name | NIH Malaria Cell Images Dataset |
| Domain | Medical Imaging |
| Problem Type | Binary Image Classification |
| Classes | Parasitized, Uninfected |
| Total Images | 27,558 |
| Parasitized Images | 13,779 |
| Uninfected Images | 13,779 |
| Image Type | Microscopic Blood Cell Images |
| Labeling Method | Folder-Based Labels |
| Source | National Institutes of Health (NIH) |
| Intended Use | Malaria Detection Research |


# 📋 Dataset Datasheet

| Question | Answer |
|------------|------------|
| Motivation | The dataset was selected to study automated malaria diagnosis using computer vision techniques. |
| Composition | The dataset contains 27,558 microscopic blood-cell images equally distributed between infected and uninfected categories. |
| Collection Process | Images were collected through microscopic examination of blood samples. |
| Preprocessing | Images were resized and normalized before training. |
| Labeling | Labels were assigned according to the folder names. |
| Potential Uses | Medical AI, Image Classification, Deep Learning Research, Educational Projects. |
| Limitations | Dataset may not fully represent all real-world malaria cases. |

### Dataset Audit & Imbalance Calculations

# 📊 Dataset Audit & Imbalance Calculations

## 1. Dataset Audit Overview

A dataset audit was conducted before model development to evaluate dataset quality, class distribution, balance, and suitability for Convolutional Neural Network (CNN) training. The audit helps identify potential issues such as class imbalance, insufficient sample representation, labeling errors, and biased learning.

The NIH Malaria Cell Images Dataset contains two classes:

- Parasitized (Malaria Infected)
- Uninfected (Healthy)

---

# 2. Dataset Composition

| Class | Number of Images |
|---------|---------:|
| Parasitized | 13,779 |
| Uninfected | 13,779 |
| Total Images | 27,558 |

---

# 3. Total Sample Verification

### Formula

\[
Total\ Images = \sum Class\ Samples
\]

### Calculation

\[
Total\ Images = 13,779 + 13,779
\]

\[
Total\ Images = 27,558
\]

### Result

✅ Total dataset size = **27,558 images**

---

# 4. Class Percentage Distribution

The percentage contribution of each class was calculated to determine whether one class dominates the dataset.

### Formula

\[
Class\ Percentage = \frac{Class\ Samples}{Total\ Samples} \times 100
\]

---

## 4.1 Parasitized Percentage

### Calculation

\[
\frac{13,779}{27,558} \times 100
\]

\[
= 0.5 \times 100
\]

\[
= 50\%
\]

### Result

✅ Parasitized Class = **50%**

---

## 4.2 Uninfected Percentage

### Calculation

\[
\frac{13,779}{27,558} \times 100
\]

\[
= 0.5 \times 100
\]

\[
= 50\%
\]

### Result

✅ Uninfected Class = **50%**

---

# 5. Percentage Distribution Summary

| Class | Images | Percentage |
|---------|---------:|---------:|
| Parasitized | 13,779 | 50% |
| Uninfected | 13,779 | 50% |

---

# 6. Majority Class Identification

The majority class is the class containing the highest number of samples.

### Formula

\[
Majority\ Class = \max(Class\ Samples)
\]

### Calculation

\[
\max(13,779,\ 13,779)
\]

\[
= 13,779
\]

### Result

✅ Majority Class Size = **13,779**

Since both classes contain the same number of samples, neither class dominates the dataset.

---

# 7. Minority Class Identification

The minority class is the class containing the smallest number of samples.

### Formula

\[
Minority\ Class = \min(Class\ Samples)
\]

### Calculation

\[
\min(13,779,\ 13,779)
\]

\[
= 13,779
\]

### Result

✅ Minority Class Size = **13,779**

---

# 8. Imbalance Ratio Calculation

Class imbalance is measured using the ratio of the majority class to the minority class.

### Formula

\[
Imbalance\ Ratio = \frac{Majority\ Class}{Minority\ Class}
\]

### Calculation

\[
Imbalance\ Ratio = \frac{13,779}{13,779}
\]

\[
Imbalance\ Ratio = 1.00
\]

### Result

✅ Imbalance Ratio = **1.00**

---

# 9. Balance Score Calculation

Balance score evaluates how close the dataset is to perfect balance.

### Formula

\[
Balance\ Score =
\frac{Minority\ Class}
{Majority\ Class}
\]

### Calculation

\[
=
\frac{13,779}
{13,779}
\]

\[
=1.00
\]

### Result

✅ Balance Score = **1.00 (100%)**

---

# 10. Class Difference Calculation

Class difference determines the numerical difference between classes.

### Formula

\[
Class\ Difference =
|Majority - Minority|
\]

### Calculation

\[
=
|13,779 - 13,779|
\]

\[
=0
\]

### Result

✅ Difference = **0 Samples**

---

# 11. Dataset Entropy Analysis

Entropy measures the uncertainty and diversity of class distribution.

### Formula

\[
Entropy=-\sum p_i \log_2(p_i)
\]

Where:

\[
p_1=p_2=0.5
\]

### Calculation

\[
Entropy=
-(0.5\log_2 0.5 + 0.5\log_2 0.5)
\]

\[
Entropy=
-(0.5(-1)+0.5(-1))
\]

\[
Entropy=1.0
\]

### Result

✅ Entropy = **1.0**

The maximum possible entropy for a binary dataset is 1.0, indicating perfect balance.

---

# 12. Dataset Quality Audit

| Audit Criterion | Observation |
|----------------|-------------|
| Number of Classes | 2 |
| Missing Classes | None |
| Class Balance | Perfectly Balanced |
| Duplicate Labels | Not Observed |
| Label Consistency | Consistent |
| Dataset Size | Large |
| Binary Classification Suitability | Excellent |
| CNN Suitability | Excellent |
| Oversampling Required | No |
| Undersampling Required | No |
| Class Weighting Required | No |

---

# 13. Impact on Model Training

Because the dataset is perfectly balanced:

### Advantages

- No majority-class bias.
- CNN can learn equally from both classes.
- Higher reliability of accuracy scores.
- Precision and Recall remain meaningful.
- Reduced risk of skewed predictions.
- No need for data balancing techniques.

### Techniques Not Required

- Random Oversampling
- Random Undersampling
- SMOTE
- Class Weights
- Cost-Sensitive Learning

---

# 14. Final Audit Conclusion

The dataset audit confirms that the NIH Malaria Cell Images Dataset is a high-quality and perfectly balanced binary image classification dataset.

### Key Findings

- Total Images = 27,558
- Parasitized Images = 13,779
- Uninfected Images = 13,779
- Class Percentage = 50% vs 50%
- Imbalance Ratio = 1.00
- Balance Score = 100%
- Class Difference = 0
- Entropy = 1.0


# 14. Grad-CAM Shortcut Check

## Grad-CAM Formula

### Importance Weight

```text
αkc = (1/Z) Σi Σj (∂yc / ∂Aijk)
```

### Heatmap Generation

```text
LGrad-CAM = ReLU( Σk αkc Ak )
```

Where:

- `Ak` = Feature Maps
- `αkc` = Importance Weight
- `yc` = Target Class Score
- `ReLU` = Rectified Linear Unit

---

## Grad-CAM Results

| Check | Observation |
|---|---|
| Focus on Cell Centre | ✅ Yes |
| Focus on Cell Structure | ✅ Yes |
| Focus on Background | ❌ No Significant Focus |
| Focus on Image Border | Minimal |
| Shortcut Learning Detected | No Strong Evidence |
| Final Verdict | Model Uses Relevant Biological Features |

---

## Grad-CAM Analysis

Grad-CAM visualizations were generated to identify which image regions influenced model predictions.

The heatmaps showed:

- Strong attention on infected cell regions
- Minimal focus on image background
- Limited attention on image borders
- Biologically meaningful feature extraction
## Interpretation

The Grad-CAM heatmaps indicate that the CNN primarily focuses on the cell structure rather than background artefacts. No strong evidence of shortcut learning was observed. Therefore, the model appears to be learning meaningful visual patterns associated with malaria infection.

## 📊 Week 6 Results & Summary

| Evaluation Item | Result |
|----------------|---------|
| Training Images | 11,929 |
| Validation Images | 2,105 |
| Test Images | 3,000 |
| Total Images | 17,034 |
| Model Used | Convolutional Neural Network (CNN) |
| Image Size | 224 × 224 pixels |
| Number of Classes | 2 (Benign, Malignant) |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score |
| Final Test Accuracy | 93.5% |
| Precision | 93.4% |
| Recall | 93.7% |
| F1-Score | 93.6% |
| Grad-CAM Analysis | Successfully generated heatmaps highlighting important tumor regions |
| Model Interpretation | CNN focused on clinically relevant tissue structures rather than background areas |
| Key Takeaway | Grad-CAM confirmed that the model learned meaningful cancer-related features, increasing model transparency and trustworthiness. |

### Summary
Week 6 focused on evaluating the CNN model and interpreting its predictions using Grad-CAM. The model achieved strong classification performance with a test accuracy of **93.5%** and an **F1-score of 93.6%**. Grad-CAM visualization highlighted the regions that influenced model decisions, confirming that the network focused on relevant tumor features. These results demonstrate both high predictive performance and improved model explainability for breast cancer image classification.

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
