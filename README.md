# Applied AI Project 2: Computer Vision, Text, and Multimodal AI

**Author:** Haseeb Bilal
**Course:** Introduction to Applied AI
**Frameworks:** PyTorch, scikit-learn, spaCy, EasyOCR
**Platform:** Kaggle Notebooks

---

# Week 5: Neural Networks, Convolutions, and Transfer Learning

## Dataset
Intel Image Classification (Buildings, Forest, Glacier, Mountain, Sea, Street).

| Split | Images |
|---|---:|
| Training | 11,929 |
| Validation | 2,105 |
| Testing | 3,000 |
| **Total** | **17,034** |

## Method
- **Feature extraction:** frozen ResNet-18, new 6-class head (3,078 trainable of 11,179,590 parameters). Best validation accuracy 90.1%.
- **Fine-tuning:** `layer4` unfrozen (8,396,806 trainable parameters). Best validation accuracy 93.8% (+3.7 points).
- **Evaluation:** held-out test set.

## Results

| Metric | Result |
|---|---:|
| Test accuracy | 93.5% |
| Macro-average F1 | 93.6% |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Buildings | 92.8% | 93.8% | 93.3% |
| Forest | 99.4% | 99.4% | 99.4% |
| Glacier | 89.4% | 88.8% | 89.1% |
| Mountain | 90.8% | 88.6% | 89.7% |
| Sea | 94.5% | 98.4% | 96.4% |
| Street | 94.7% | 93.0% | 93.9% |

Best class: Forest. Lowest recall: Mountain and Glacier (visually similar landscapes).
Saved model: `resnet18_intel.pt`

---

# Week 6: Applied Computer Vision, Malaria Parasite Detection (ResNet-18)

**Task:** classify a single thin-blood-smear cell crop as **Parasitized** or **Uninfected**.
**Intended use:** triage aid for a microscopist, not a diagnosis.
**Primary metric (declared before training):** sensitivity (recall) on Parasitized, because missing an infected cell is the costliest mistake.

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
