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

# Week 6: Applied Computer Vision — Dataset Audit and Model Explainability

## 🎯 Objectives

- Audit the dataset and inspect class distribution.
- Examine image dimensions, colour modes, and representative samples.
- Investigate exact and near-duplicate images.
- Reduce the risk of data leakage during training and validation.
- Examine class imbalance and appropriate handling strategies.
- Train and fine-tune a ResNet-18 model for six-class image classification.
- Evaluate performance using a classification report, macro-F1 score, and confusion matrix.
- Apply Grad-CAM to investigate the visual regions influencing model predictions.
- Identify possible shortcut-learning behaviour.

## 📂 Dataset

Week 6 uses the Intel Image Classification dataset containing the following six classes:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

### Dataset Audit Results

The initial dataset audit produced the following results:

| Dataset Detail | Result |
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
| Imbalance ratio | 1.147 |
| Typical image dimensions | 150 × 150 pixels |
| Colour modes in inspected sample | RGB |

The imbalance ratio is calculated as:

**Imbalance ratio = Maximum class count / Minimum class count**

**Imbalance ratio = 2,512 / 2,191 ≈ 1.147**

This indicates that the six training classes are relatively balanced, although some classes contain more images than others.

Image dimensions and colour modes were inspected using a sample of 300 training images.

## 🧠 Methodology

### 1. Dataset Indexing and Exploration

The dataset was indexed to record image paths, filenames, and class labels. The analysis examined class counts, image dimensions, colour modes, and representative images.

This stage established a baseline understanding of the dataset before model training.

### 2. Exact and Near-Duplicate Detection

Duplicate detection was included to investigate whether identical or visually similar images could affect the evaluation process.

Exact-duplicate checks identify files or image contents that match. Near-duplicate checks investigate images that may be visually similar despite differences in file representation or minor image changes.

The actual number of detected duplicates should be reported after the corresponding notebook cells have been executed.

### 3. Data Splitting and Leakage Prevention

The notebook examines the separation of training and validation data to reduce the risk of data leakage.

Reliable splitting is important because duplicate or closely related images appearing across different subsets can produce overly optimistic validation results.

Any grouping strategy should be interpreted according to the identifiers and relationships actually available in the dataset.

### 4. Image Preprocessing and Class Imbalance

Images are prepared for the ResNet-18 architecture using suitable transformations.

The class distribution is examined to determine whether imbalance-handling techniques, such as class weighting or weighted sampling, are appropriate for the experiment.

The selected configuration and its effects should be documented using the actual notebook outputs.

### 5. ResNet-18 Training and Fine-Tuning

ResNet-18 is used as the image classification architecture for the six scene categories.

The training process adapts the model to the target dataset. Fine-tuning and other training settings should be documented according to the configuration used in the completed notebook.

### 6. Model Evaluation

Model performance is assessed using multiple classification metrics:

- Accuracy
- Precision
- Recall
- Macro-F1 score
- Classification report
- Confusion matrix

Macro-F1 is particularly useful because it calculates the F1-score for each class and then averages the class scores equally.

The confusion matrix helps identify which scene categories are most frequently confused.

### 7. Grad-CAM and Shortcut Learning

Grad-CAM is used to generate visual explanations of selected model predictions by highlighting image regions that contribute to the model's output.

The visualizations can help investigate whether predictions appear to rely on meaningful scene features or potentially misleading cues, such as background patterns, lighting, image framing, or scene-specific objects.

Grad-CAM provides evidence about influential image regions, but it does not independently prove that shortcut learning has occurred.

## 📊 Week 6 Results

The following results should be completed after running the Week 6 notebook.

| Evaluation Item | Result |
|---|---|
| Training images | 14,034 |
| Test images | 3,000 |
| Number of classes | 6 |
| Class imbalance ratio | 1.147 |
| Exact duplicates | Record actual output |
| Near-duplicate findings | Record actual output |
| Validation performance | Record actual output |
| Final test accuracy | Record actual output |
| Macro-F1 score | Record actual output |
| Most confused classes | Identify from confusion matrix |
| Grad-CAM observations | Record findings from actual heatmaps |

**Note:** The dataset audit values above are based on the observed notebook outputs. Final model metrics and explainability findings must be updated after executing the relevant training, evaluation, and Grad-CAM cells.

## 🔍 Key Learnings

- Dataset auditing helps identify class distribution and image characteristics before training.
- Duplicate detection can help identify potential sources of data leakage.
- Careful data splitting is essential for reliable model evaluation.
- Class weighting and sampling strategies can help address class imbalance when appropriate.
- Accuracy alone does not describe performance across every class.
- Macro-F1 provides a useful summary of class-wise performance.
- Confusion matrices reveal common classification errors.
- Grad-CAM provides visual insight into image regions influencing predictions.
- A model can achieve strong accuracy while still relying on features that may not generalize well to new data.

## 🚀 Future Improvements

- Investigate misclassified images and commonly confused scene categories.
- Compare different image augmentation strategies.
- Experiment with learning rates and fine-tuning configurations.
- Examine Grad-CAM visualizations for correct and incorrect predictions.
- Test the model on external images not included in the original dataset.
- Compare ResNet-18 with other pretrained computer vision architectures.
- Explore computer vision applications in electronics component classification.

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
```

The trained model file is optional. If it is too large for GitHub, keep it in Kaggle or another suitable storage location and provide instructions for accessing it.

# ✅ Overall Conclusion

This project demonstrates a practical progression from neural network fundamentals and transfer learning to dataset auditing and explainable computer vision.

Week 5 focuses on adapting a pretrained ResNet-18 model through feature extraction and fine-tuning. The reported experiments achieved 93.8% best validation accuracy and 93.5% test accuracy.

Week 6 extends the workflow by examining dataset quality, potential data leakage, class-wise evaluation, and Grad-CAM visualizations. These techniques help assess not only how well a model predicts image classes, but also how reliably its performance is evaluated and what visual evidence may influence its predictions.

The Week 6 model-performance conclusions will be finalized after the complete notebook has been executed and its actual evaluation outputs reviewed.

---

**Project:** Applied AI Project 2 — Image Classification  
**Author:** Haseeb Bilal  
**Course:** Introduction to Applied AI
