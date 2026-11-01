# Applied AI Project 2: Computer Vision, Text, and Multimodal AI

**Author:** Haseeb Bilal
**Course:** Introduction to Applied AI
**Frameworks:** PyTorch, scikit-learn, spaCy, EasyOCR
**Platform:** Kaggle Notebooks

---

# Week 5: Neural Networks, Convolutions, and Transfer Learning

## Goal

The main goal of Week 5 was to understand and implement Neural Networks, Convolutional Neural Networks (CNNs), and Transfer Learning for image classification.

The project focused on using a pre-trained ResNet-18 model to classify images into six categories: Buildings, Forest, Glacier, Mountain, Sea, and Street.
The key objectives were:
- Understand how CNNs extract important features from images.
- Learn how pre-trained models work using Transfer Learning.
- Implement Feature Extraction using frozen ResNet-18 layers.
- Apply Fine-Tuning by unfreezing the final `layer4` block.
- Compare both approaches using validation accuracy.
- Evaluate the final model using Accuracy, Precision, Recall, and F1-Score.
- Save the trained model as `resnet18_intel.pt` for future predictions.

**Final Objective:** Build an accurate image classification model that can recognize six different environmental categories using Deep Learning techniques.

## Dataset

The **Intel Image Classification Dataset** contains 17,034 images belonging to six different classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

### Dataset Distribution

| Split | Images |
|---|---:|
| Training | 11,929 |
| Validation | 2,105 |
| Testing | 3,000 |
| **Total** | **17,034** |

- **Training:** Used to train the model.
- **Validation:** Used to monitor performance and improve the model.
- **Testing:** Used for final evaluation on unseen images.

## Method

This project uses **ResNet-18**, a pre-trained Convolutional Neural Network (CNN), with Transfer Learning.

### 1. Feature Extraction

In this approach, all pre-trained ResNet-18 layers were frozen, and a new classification head was added for six classes.

- Trainable parameters: 3,078
- Total parameters: 11,179,590
- Best validation accuracy: 90.1%

Only the new classification head was trained, while the original network weights remained unchanged.

### 2. Fine-Tuning

In this approach, the final ResNet-18 block (`layer4`) was unfrozen so that the model could learn dataset-specific features.

- Trainable parameters: 8,396,806
- Total parameters: 11,179,590
- Best validation accuracy: 93.8%

Fine-tuning improved validation accuracy by **3.7 percentage points** compared to feature extraction.

### Comparison

| Approach | Trainable Parameters | Validation Accuracy |
|---|---:|---:|
| Feature Extraction | 3,078 | 90.1% |
| Fine-Tuning | 8,396,806 | 93.8% |

## Evaluation Metrics and Formulas

The final model was evaluated on 3,000 unseen test images using the following metrics.

**1. Accuracy**

Measures the percentage of correctly classified images.

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}\times100
\]

**2. Precision**

Measures how many predicted positive samples are actually positive.

\[
Precision = \frac{TP}{TP+FP}\times100
\]

**3. Recall**

Measures how many actual positive samples are correctly identified.

\[
Recall = \frac{TP}{TP+FN}\times100
\]

**4. F1-Score**

Calculates the harmonic mean of precision and recall.

\[
F1 = 2\times\frac{Precision\times Recall}{Precision+Recall}
\]

**5. Macro-Average F1**

Calculates the average F1-score across all six classes.

\[
Macro\ F1 = \frac{1}{6}\sum_{i=1}^{6}F1_i
\]

*Note: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative.*

## Results

### Overall Test Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **93.5%** |
| Macro-Average F1 | **93.6%** |

The model correctly classified approximately 93.5% of the test images.

### Class-wise Results

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Buildings | 92.8% | 93.8% | 93.3% |
| Forest | 99.4% | 99.4% | 99.4% |
| Glacier | 89.4% | 88.8% | 89.1% |
| Mountain | 90.8% | 88.6% | 89.7% |
| Sea | 94.5% | 98.4% | 96.4% |
| Street | 94.7% | 93.0% | 93.9% |

### Results Analysis

- **Best Class:** Forest achieved the highest F1-score of 99.4%.
- **Sea:** Achieved the highest recall of 98.4%.
- **Lowest Recall:** Mountain (88.6%) and Glacier (88.8%).
- Mountain and Glacier were more challenging because their images contain visually similar landscape features.

### Conclusion

The fine-tuned ResNet-18 model achieved **93.5% test accuracy** and **93.6% macro-average F1-score**.

Fine-tuning improved validation accuracy from 90.1% to 93.8%, demonstrating the benefit of adapting pre-trained CNN layers to a new dataset.

**Saved Model:** `resnet18_intel.pt`

---

# Week 6: Applied Computer Vision, Malaria Parasite Detection (ResNet-18)

## Week 6: Project Proposal – Malaria Parasite Detection

### Problem Statement
Malaria is a serious disease caused by parasites that can be detected through blood-smear images. Manual examination of these images is time-consuming and requires expert attention.

The goal of this project is to develop a Deep Learning model using **ResNet-18** to automatically classify blood-smear cell images into two categories:

- **Parasitized:** Cells containing malaria parasites.
- **Uninfected:** Healthy cells without visible parasites.

### Objectives
- Apply CNNs for medical image classification.
- Use Transfer Learning with pre-trained ResNet-18.
- Implement Feature Extraction and Fine-Tuning.
- Detect infected cells with high sensitivity (recall).
- Evaluate performance using Accuracy, Precision, Recall, F1-score, and AUC.
- Use Grad-CAM to understand model predictions.

### Intended Use
The model is designed as a **triage assistance tool for microscopists**, not as a replacement for medical professionals or a standalone diagnostic system.

**Primary Metric:** Sensitivity (Recall) for the Parasitized class, because missing an infected cell is the most costly mistake.

## Dataset and Audit
NIH / NLM thin-blood-smear images (Kaggle `iarunava/cell-images-for-detecting-malaria`).

| Item | Result |
|---|---|
| Parasitized / Uninfected | 13,779 / 13,779 (total 27,558) |
| Imbalance ratio | 1.0 (balanced by construction) |
| Image size (300-image sample) | median 130 × 133 px (min 61 × 82, max 229 × 187), all RGB |
| Groups (patients/slides) | 200 groups, about 137.8 cells each |
| Groups with Parasitized cells | 150 (50 patients contribute only healthy cells) |

The Kaggle copy contains a nested second folder (`cell_images/cell_images`). Only the top-level class folders were indexed, so no image is counted twice.

## Leakage Audit
- **Grouped split** with `StratifiedGroupKFold` (no patient in two splits): train / val / test = 19,118 / 5,105 / 3,335 images. Shared groups: 0 / 0 / 0.
- Test prevalence of Parasitized is 40.2% (patients cannot be split, so class balance is approximate).
- **Exact duplicates (MD5):** 0. **Near-duplicates** (32×32 fingerprint, threshold 0.05): 0 of 3,335 test and 0 of 5,105 validation images. Closest genuine pair d = 0.065 (different cells, nothing deleted).

## Transforms and Class Weights
- **Train augmentation:** RandomResizedCrop 224 (scale 0.8 to 1.0), horizontal and vertical flips, rotation ±180°, mild ColorJitter. Evaluation: resize 224 and ImageNet normalisation. RGB kept because stain colour is evidence.
- **Class weights** w_c = N / (K · n_c): Parasitized 0.9972, Uninfected 1.0028 (practically ordinary cross-entropy).

## Model and Training
ResNet-18 with local ImageNet weights, new 2-class head (512 × 2 + 2 = 1,026 trainable parameters). Adam, model selected by validation macro-F1.

| Phase | Trainable | Epochs | Best val macro-F1 |
|---|---|---:|---:|
| 1: feature extraction | head only (1,026) | 2 | 0.855 |
| 2: fine-tuning | `layer4` + head (about 8.39 M) | 3 | 0.938 |

Fine-tuning improved validation macro-F1 by 0.083 (+9.7% relative).

## Test Results (test set used once, threshold 0.5)
Confusion matrix (positive = Parasitized): TP 1,283, FN 57, FP 78, TN 1,917.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Parasitized | 0.943 | 0.957 | 0.950 | 1,340 |
| Uninfected | 0.971 | 0.961 | 0.966 | 1,995 |

| Metric | Result |
|---|---:|
| Accuracy | 0.960 |
| Macro-F1 | 0.958 |
| Weighted-F1 | 0.960 |

## Threshold for Target Recall (≥ 0.95)
Threshold chosen on the **validation** set, then applied once to the test set.

| Quantity | Value |
|---|---:|
| Threshold | 0.283 |
| Test AUC | 0.991 |
| Test sensitivity | 0.972 (target reached) |
| Test specificity | 0.936 |
| Test PPV (40.2% prevalence) | 0.910 |
| PPV at 5% prevalence | 0.444 |

Lowering the threshold from 0.5 to 0.283 gives about 19 fewer missed infected cells for about 50 more false alarms. PPV depends on prevalence, so it must be quoted for the real use case.

## Grad-CAM Shortcut Check
Grad-CAM on `layer4`: α_k = (1/Z) Σ_i Σ_j ∂y^c/∂A_ij^k, heatmap = ReLU(Σ_k α_k A^k).

- **Correct Uninfected cells:** heat on the cell centre (legitimate evidence).
- **Missed Parasitized cells (3 examples):** heat near the black border or on the cell body, and in one case the visible purple parasite spot was cold (not legitimate evidence).
- **Verdict:** no strong shortcut confirmed, but a shortcut is suspected and the check is incomplete (no correctly classified Parasitized examples were shown).
- **Next experiments:** Grad-CAM on correct Parasitized cells, mask the background and border and re-score, remove stain colour and re-score, external validation on another hospital or microscope.

## Saved Model
`week6_model.pt` (state dict, classes, backbone `resnet18`, image size 224, grayscale False).

---

# Week 7: Text and Multimodal AI

## Text Classifier
- **Dataset:** SMS Spam Collection, 5,572 texts (13.4% spam)
- **Macro-F1:** 0.974 (word 1-2 grams) vs 0.982 (char 2-5 grams)
- **Spam precision / recall:** 0.979 / 0.933 (word) vs 0.986 / 0.953 (char)

## Top Features
- **Positive (spam):** call, txt, free, text, reply
- **Negative (ham):** my, me, ok, that, it
- **Shortcuts found:** `uk`, `gt`, `lt` (dataset artefacts, not spam meaning)

## NER (Healthcare Domain)
- **Held-out set (5 sentences, 17 gold entities):** F1 0.437 -> 0.848 with entity ruler
- **Provided sentences (6 sentences, 16 gold entities):** F1 0.467 -> 0.812

## Regex Fields
- **Fields extracted:** CNIC, mobile, amount, date, email, invoice, MRN, temperature, dose, SpO2
- **False positives tested:** 7 negative tests (e.g. `Bus 39C`, `Patient ID 123456`, `Humidity 92%`)
- **Result:** 17/17 checks passed

## OCR
- **Synthetic invoice CER:** 0.000 raw -> 0.000 after correction rules
- **Hand-made OCR errors:** 0.062 / 0.059 / 0.143 -> 0.000

## Multimodal Pipeline
Week 6 image model (ResNet-18, Parasitized vs Uninfected) + text evidence -> JSON record.

```json
{
  "image": "C100P61ThinF_IMG_20150918_144104_cell_162.png",
  "image_probabilities": {"Parasitized": 0.999, "Uninfected": 0.001},
  "note_masked": "Patient Bilal Hussain, aged 28, fever with chills and sweating for 3 days. Seen at Shifa International Islamabad on 12/09/2026. Mobile [PHONE], CNIC [CNIC].",
  "text_evidence": {
    "entities": [["Bilal Hussain","PERSON"],["Shifa International","ORG"],["Islamabad","GPE"],["12/09/2026","DATE"]],
    "fields": {"date": ["12/09/2026"]},
    "urgent_terms": ["fever", "chills", "sweating"]
  },
  "decision": "REFER",
  "reasons": ["image model: P(Parasitized) = 1.00", "note mentions: fever, chills, sweating"],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

## Privacy
CNIC, phone, and email are masked before extraction and logging. A search of the JSON for the phone and CNIC digits found 0 leaks.

---

## Technologies Used
Python, PyTorch, Torchvision, scikit-learn, spaCy, EasyOCR, NumPy, Pandas, Matplotlib, Pillow, Kaggle Notebooks

## Disclaimer
Educational project. The malaria model and case-record system are decision support only; a qualified human makes any medical decision.
