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

# Week 6: Malaria Cell Screening with a CNN

## Dataset
NIH Malaria Cell Images: 27,558 images (13,779 Parasitized, 13,779 Uninfected), folder-based labels.

## Dataset Audit

| Measure | Result |
|---|---|
| Class split | 50% / 50% |
| Imbalance ratio | 1.00 |
| Balance score | 1.00 |
| Class difference | 0 |
| Entropy | 1.0 (maximum for binary) |

The dataset is perfectly balanced, so oversampling, undersampling, SMOTE, and class weights are not needed.

## Model
ResNet-18, 224 × 224 RGB input, ImageNet normalisation, classes: Parasitized and Uninfected. Saved as `week6_model.pt`.

## Grad-CAM Shortcut Check
Heatmaps focused on cell structure, with minimal attention on background and borders. No strong evidence of shortcut learning (qualitative evidence only).

## Test Results

| Metric | Result |
|---|---|
| Training / Validation / Test images | _[fill in]_ |
| Test accuracy | _[fill in]_ |
| Precision | _[fill in]_ |
| Recall | _[fill in]_ |
| F1-score | _[fill in]_ |

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
Educational project. The case-record system is decision support only; a qualified human makes any medical decision.
