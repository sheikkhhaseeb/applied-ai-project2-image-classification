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

##  Week 6 Dataset

| Dataset Item | Details |
|-------------|---------|
| Dataset Name | Malaria Cell Images Dataset |
| Source | Kaggle / National Institutes of Health (NIH) |
| Total Images | 27,558 |
| Parasitized Images | 13,779 |
| Uninfected Images | 13,779 |
| Number of Classes | 2 |
| Class Labels | Parasitized, Uninfected |
| Image Format | PNG |
| Image Type | Microscopic Blood Smear Images |
| Input Image Size | 224 × 224 Pixels |
| Train-Test Split | 80% Training, 20% Testing |
| Preprocessing Techniques | Resizing, Normalization |
| Data Augmentation | Rotation, Zoom, Horizontal Flip, Width & Height Shift |

## 🔍 Week 6 Data Audit

| Audit Item | Result |
|------------|---------|
| Dataset Name | Malaria Cell Images Dataset |
| Total Images | 27,558 |
| Parasitized Images | 13,779 |
| Uninfected Images | 13,779 |
| Number of Classes | 2 |
| Missing Images | 0 |
| Duplicate Images | 0 |
| Class Distribution | Balanced |
| Class Imbalance Ratio | 1.0 |
| Image Format | PNG |
| Color Channels | RGB |
| Image Quality Check | Passed |
| Data Leakage Detected | No |
| Corrupted Files | 0 |
| Input Size for CNN | 224 × 224 Pixels |

### Audit Calculations

**Total Images**

\[
Total = Parasitized + Uninfected
\]

\[
Total = 13,779 + 13,779 = 27,558
\]

**Class Imbalance Ratio**

\[
Class\ Imbalance\ Ratio = \frac{Majority\ Class}{Minority\ Class}
\]

\[
Class\ Imbalance\ Ratio = \frac{13,779}{13,779} = 1.0
\]

A ratio of **1.0** indicates a perfectly balanced dataset.
## Transforms and Class Weights
- **Train augmentation:** RandomResizedCrop 224 (scale 0.8 to 1.0), horizontal and vertical flips, rotation ±180°, mild ColorJitter. Evaluation: resize 224 and ImageNet normalisation. RGB kept because stain colour is evidence.
- **Class weights** w_c = N / (K · n_c): Parasitized 0.9972, Uninfected 1.0028 (practically ordinary cross-entropy).

## 🧠 Week 6 Model & Training

| Model Component | Configuration |
|----------------|--------------|
| Model Type | Convolutional Neural Network (CNN) |
| Input Shape | (224, 224, 3) |
| Output Classes | 2 |
| Activation Function | ReLU |
| Output Activation | Softmax |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Number of Epochs | 10 |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score |
| Regularization | Dropout |
| Data Augmentation | Rotation, Zoom, Horizontal Flip |
| Framework | TensorFlow / Keras |

### CNN Architecture

| Layer | Purpose |
|---------|---------|
| Conv2D | Feature Extraction |
| MaxPooling2D | Dimensionality Reduction |
| Conv2D | Deep Feature Learning |
| MaxPooling2D | Spatial Reduction |
| Flatten | Convert Feature Maps into Vector |
| Dense | Classification Learning |
| Dropout | Prevent Overfitting |
| Dense (Softmax) | Final Prediction |

### Training Summary

| Training Item | Result |
|--------------|---------|
| Training Samples | 22,046 |
| Testing Samples | 5,512 |
| Epochs Completed | 10 |
| Optimizer Used | Adam |
| Best Validation Accuracy | 93.5% |
| Final Test Accuracy | 93.5% |
| Macro F1-Score | 93.6% |
| Training Status | Successfully Converged |
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

## Week 7: Text and Multimodal AI

All numbers below come from the outputs of `week7-text-and-multimodal.ipynb` (spaCy 3.8.16, scikit-learn, EasyOCR, PyTorch; all splits use `random_state=42`).

### Summary

- **Text classifier:** UCI SMS Spam Collection (Kaggle `uciml/sms-spam-collection-dataset`), **5,572 texts** (4,825 ham, 747 spam), macro-F1 **0.974 (word 1–2-grams)** vs **0.982 (char 2–5-grams)**
- **Top positive / negative features (word model):** `call, txt, free, text, reply, uk, to, www, mobile, claim, stop, your` / `my, me, ok, that, it, but, ll, gt, lt, can, what, da`
- **NER on my domain (Pakistani healthcare, held-out set):** out-of-the-box F1 **0.437** → with entity ruler **0.848**
- **Regex fields extracted:** `cnic`, `mobile`, `amount`, `date`, `email`, `invoice`, `mrn`, `temp_c`, `dose`, `spo2`; **false positives tested:** 7 negative tests, 0 false matches (17/17 checks passed)
- **OCR:** CER raw **0.062 / 0.059 / 0.143** (hand-typed OCR errors, pooled 4/47 = 0.085) → **0.000** after correction rules; EasyOCR on the synthetic invoice was already **0.000** raw
- **Pipeline:** Week 6 image model (ResNet-18) + text evidence → one JSON record (example below)
- **Privacy:** CNIC, phone and email masked **before** extraction and logging; leak test found 0 leaks (names are *not* masked, see Limitations)

---

## 1. Text classifier (SMS spam)

**Data.** 5,572 messages; spam share $=\dfrac{747}{5{,}572}=0.134$. A model that always says "ham" gets accuracy $\dfrac{4{,}825}{5{,}572}=0.866$ but macro-F1 only $\dfrac{0.928+0}{2}=0.464$, so I report macro-F1 and spam precision/recall instead of accuracy alone.

**Split.** Stratified 80/20: train 4,457 (3,859 ham, 598 spam), test 1,115 (966 ham, 149 spam).

**Features and model**

$$\text{idf}(t)=\ln\frac{1+N}{1+df(t)}+1,\qquad tf'=1+\ln(tf),\qquad w_{t,d}=\frac{tf'\cdot \text{idf}(t)}{\lVert \cdot \rVert_2}$$

$$p(\text{spam})=\sigma(\mathbf w\cdot\mathbf x+b)=\frac{1}{1+e^{-(\mathbf w\cdot\mathbf x+b)}},\qquad w_c=\frac{N_{train}}{K\,n_c}\;\Rightarrow\; w_{spam}=\frac{4{,}457}{2\times598}=3.73,\ \ w_{ham}=\frac{4{,}457}{2\times3{,}859}=0.58$$

(`class_weight='balanced'`: a missed spam costs about 6.5× a false alarm during training.)

**Metrics**

$$P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F1=\frac{2PR}{P+R},\qquad \text{macro-F1}=\frac{F1_{ham}+F1_{spam}}{2}$$

| Model | Features | TN | FP | FN | TP | Spam P | Spam R | Spam F1 | Macro-F1 | Errors (of 1,115) |
|---|---|---|---|---|---|---|---|---|---|---|
| Word 1–2-grams | 12,128 | 963 | 3 | 10 | 139 | 0.979 | 0.933 | 0.955 | **0.974** | 13 |
| Char 2–5-grams (`char_wb`) | 36,449 | 964 | 2 | 7 | 142 | 0.986 | 0.953 | 0.969 | **0.982** | 9 |

**Worked example (char model).** $P_{spam}=\frac{142}{144}=0.986$, $R_{spam}=\frac{142}{149}=0.953$, $F1_{spam}=0.969$; $F1_{ham}=0.995$; macro-F1 $=\frac{0.995+0.969}{2}=0.982$; accuracy $=\frac{964+142}{1{,}115}=0.992$.

**Word vs char.** Char n-grams use $36{,}449/12{,}128=3.0\times$ more features and gain $0.982-0.974=+0.008$ macro-F1 (3 fewer missed spam, 1 fewer false alarm). With only 149 spam messages in the test set, a 3-message difference is **not strong evidence**; I would repeat with several seeds or cross-validation before calling the char model clearly better.

### Top features (word model)

The log-odds are linear, so a word with weight $w_j$ multiplies the spam odds by $e^{w_j}$ (for example $w=2\Rightarrow\times7.39$).

| Direction | Terms (strongest first / listed in output order) | Reading |
|---|---|---|
| Spam-like (positive) | call, txt, free, text, reply, uk, to, www, mobile, claim, stop, your | Sensible: call, txt, text, reply, stop, claim, free, mobile, www (calls to action and offers) |
| Ham-like (negative) | my, me, ok, that, it, but, ll, gt, lt, can, what, da | Personal chat words |
| Dataset shortcuts | `uk`, `gt`, `lt`, `ll`, `da` | `uk`: UK-collected data; `gt`/`lt`: HTML-escaped `>` `<` found only in ham; `ll`: tokenisation leftover of "I'll". These reflect *how the data was collected*, not what spam means |

### Error analysis (word model: 13 errors = 10 FN + 3 FP, error rate $13/1{,}115=1.17\%$)

| Type | Count | Example |
|---|---|---|
| Model error | 8 | "ringtoneking 84484" (p = 0.19); "Waiting for your call." predicted spam (p = 0.85) |
| Ambiguous | 4 | "For sale - arsenal dartboard…" (p = 0.21) |
| Label error | 1 | "…old ladies running around with tattoos?" labelled spam (p = 0.09) |

Three missed spam messages scored within 0.1 of the threshold (0.412, 0.425, 0.465). Lowering the threshold to 0.4 would catch them: $R_{spam}\ge\frac{142}{149}=0.953$ (false alarms may rise).

### Messages I wrote (robustness test)

| Message | Word P(spam) | Char P(spam) | Verdict |
|---|---|---|---|
| `W1N a FR33 pr1ze!! txt CLAIM now` | 0.99 | 0.85 | spam caught |
| `kal milte hain, 5 baje` (Roman Urdu ham) | 0.19 | 0.12 | ham correct |
| `aap ne 50000 ka inaam jeeta hai, abhi claim karein` (Roman Urdu spam) | 0.41 | 0.49 | **missed** by both (below 0.5) |
| `C.L.A.I.M y0ur fr33 g1ft n0w` (disguised spam) | 0.19 | 0.23 | **missed** by both |

Neither model handles Roman Urdu or heavy disguise because the training data is English UK SMS. Both vectorisers keep the default `lowercase=True` (including `char_wb`), so letter case is not used as a feature.

---

## 2. Named-entity recognition (NER) in my domain: Pakistani healthcare

Scoring is **exact match** on `(text, label)` pairs, micro-averaged over all sentences:

$$TP=\sum|\text{pred}\cap\text{gold}|,\quad FP=\sum|\text{pred}\setminus\text{gold}|,\quad FN=\sum|\text{gold}\setminus\text{pred}|,\quad F1=\frac{2TP}{2TP+FP+FN}$$

A boundary error (for example "Farmer Tariq Mehmood" instead of "Tariq Mehmood") counts as one FP **and** one FN.

**Design.** 10 healthcare sentences: the first 5 are the *development set* (rules written from these only), the last 5 the *held-out set* (not inspected while writing rules). spaCy `en_core_web_sm` with an `entity_ruler` inserted **before** `ner`.

**Rules (7 patterns):** MONEY (`pkr`/`rs`/`rs.` + optional punctuation + number); DATE (`dd/mm/yyyy` regex); generic ORG (Title-case words + Hospital/Lab/Laboratory); exact ORG *Shifa International*; GPE for Islamabad, Rawalpindi, Lahore.

| Data | Gold | Out-of-the-box TP/FP/FN | F1 before | With ruler TP/FP/FN | P after | R after | **F1 after** | Change |
|---|---|---|---|---|---|---|---|---|
| Provided sentences (6) | 16 | 7 / 7 / 9 | 0.467 | 13 / 3 / 3 | 0.812 | 0.812 | **0.812** | +0.345 |
| My domain, development (5) | 15 | 6 / 7 / 9 | 0.429 | 15 / 1 / 0 | 0.938 | 1.000 | **0.968** | +0.539 |
| My domain, held-out (5) | 17 | 7 / 8 / 10 | 0.437 | 14 / 2 / 3 | 0.875 | 0.824 | **0.848** | +0.411 |

**Worked example (held-out, after rules).** $P=\frac{14}{16}=0.875$, $R=\frac{14}{17}=0.824$, $F1=\frac{2\times14}{2\times14+2+3}=\frac{28}{33}=0.848$.

**Optimism gap.** $0.968-0.848=0.120$: the development score is inflated by tuning on the same sentences, so the **held-out 0.848 is the honest number**. Total errors ($FP+FN$) over the 10 sentences fell from $16+18=34$ to $1+5=6$.

**Remaining held-out errors**

| Sentence | Error | Cause / fix |
|---|---|---|
| "Nurse Fatima Noor from Peshawar…" | "Nurse Fatima Noor" tagged ORG (1 FP + 1 FN) | Title inside the span; add a title-stripping component or train a model |
| "Chughtai Lab Karachi billed…" | "Karachi" (GPE) missed | City not in my rules; add a city gazetteer (~30 cities) |
| "The patient Hira Shah … from Quetta…" | "Quetta" tagged ORG (1 FP + 1 FN) | City not in my rules; gazetteer |

Choice: rules + regex for closed lists and fixed formats (hospitals, cities, money, dates); a model trained on annotated Pakistani clinical text for open-ended person names. With only 17 held-out entities, one extra error moves recall by about 0.06, so the direction is clear but the numbers are not precise.

---

## 3. Regex fields and false-positive tests

| Field | Pattern idea | Example match |
|---|---|---|
| `cnic` | `\b\d{5}-\d{7}-\d\b` (5+7+1 = 13 digits, 2 hyphens) | `35202-1234567-1` |
| `mobile` | `(\+92[- ]?\|\b0)3\d{2}[- ]?\d{7}\b` (10 digits after prefix) | `+92 300 1234567`, `0300-1234567` |
| `amount` | `(PKR\|Rs\.?)\s?([\d,]+(\.\d{1,2})?)` | `12,500.50` |
| `date` | `\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b` | `05/10/2026` |
| `email` | `\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b` | `info@engro.com.pk` |
| `invoice` | `\b(Invoice\s+)?No\.?\s*(\d{3,})\b` | `4417` |
| `mrn` (mine) | `\bMRN[-:\s]*\d{6,8}\b` | `MRN-2045871` |
| `temp_c` (mine) | `\b(3[5-9]\|4[0-2])(\.\d)?\s?°\s?C\b` (35–42 °C, degree sign required) | `38.5 °C` |
| `dose` (mine) | `\b\d{2,4}\s?(mg\|mcg\|ml)\b` | `250 mg` |
| `spo2` (mine) | `\bSpO2\s*[:=]?\s*(\d{2,3})\s?%` | `SpO2 92%` |

**False-positive (negative) tests: all rejected**

| Negative text | Field that must not match | Result |
|---|---|---|
| `Order 12345678901234 shipped` (14 digits, no hyphens) | cnic | no match |
| `Room 0300 on floor 3` (only 4 digits) | mobile | no match |
| `Bus 39C leaves at 5` (no degree sign) | temp_c | no match |
| `Ward 12, bed 450 magnets` (`mg` not a whole unit) | dose | no match |
| `Patient ID 123456 in ward 7` (no MRN keyword) | mrn | no match |
| `Humidity 92% outside` (no SpO2 keyword) | spo2 | no match |
| `MRN 12 is too short` (only 2 digits) | mrn | no match |

Total: 10 positive + 7 negative = **17 checks, 17 passed** ($17/17=100\%$).

**Patterns that first matched too much.** My first MRN pattern `\d{6,8}` matched `123456` in "Patient ID 123456", and my first temperature pattern matched `39C` in "Bus 39C". Fixes: a required keyword (`MRN`), exact digit counts, a numeric range, a compulsory `°`, and word boundaries. A pattern that passes its positive tests can still be wrong; only negative tests exposed these two mistakes.

---

## 4. OCR and character error rate (CER)

$$\text{CER}=\frac{S+D+I}{N_{ref}},\qquad d_{i,j}=\min\big(d_{i-1,j}+1,\ d_{i,j-1}+1,\ d_{i-1,j-1}+[a_i\ne b_j]\big)$$

where $S,D,I$ are substitutions, deletions and insertions (Levenshtein distance) and $N_{ref}$ is the reference length.

| Reference ($N$) | OCR output | Edits | CER raw | After `fix_ocr` | CER after |
|---|---|---|---|---|---|
| Invoice No. 4417 (16) | Invoice N0. 4417 | 1 | $1/16=0.062$ | Invoice No. 4417 | 0.000 |
| Total: PKR 12,500 (17) | Total: PKR 12,5O0 | 1 | $1/17=0.059$ | Total: PKR 12,500 | 0.000 |
| Due 30/09/2026 (14) | Due 3O/09/2O26 | 2 | $2/14=0.143$ | Due 30/09/2026 | 0.000 |
| **Pooled** (47) | | 4 | $4/47=0.085$ | | **0.000** |
| EasyOCR on synthetic invoice (124) | identical to truth | 0 | $0/124=0.000$ | unchanged | 0.000 |

**Correction rules** fire only in a certain context: letter O → 0 next to a digit (not inside "Invoice", "Total"), l/I → 1 between two digits, and `N0.` → `No.`. On the already-correct invoice the rules changed nothing, which also shows they do not damage good text.

**Honest note.** The raw CER of 0.000 on the synthetic invoice is not a real-world estimate: the text is computer-rendered black-on-white in a clean font, with no blur, skew or noise. On real phone photos I expect a clearly non-zero CER (roughly 5–20 %, an assumption I have not measured). CER also treats all characters equally: reading 45,000 as 46,000 is only $1/6=0.167$ of that field but a 100 % wrong amount, so numbers and dates are also validated with the Part 3 regexes.

---

## 5. Pipeline: image model + text evidence → JSON record

**Flow:** `mask_pii(note)` → `text_evidence(masked note)` (domain spaCy entities + regex fields + urgent keywords) and `image_evidence(image)` (ResNet-18 from Week 6, classes `Parasitized` / `Uninfected`, 224 px, softmax) → decision.

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad \text{REFER}=\mathbb 1\big[p_{Parasitized}\ge0.5\big]\ \lor\ \mathbb 1\big[\,|\text{urgent terms}|>0\,\big]$$

The rule is deliberately cautious: a missed malaria case costs more than an unnecessary referral (high recall, lower precision).

**Example record (real output)**

```json
{
  "image": "C100P61ThinF_IMG_20150918_144104_cell_162.png",
  "image_probabilities": {"Parasitized": 0.999, "Uninfected": 0.001},
  "note_masked": "Patient Bilal Hussain, aged 28, fever with chills and sweating for 3 days. Seen at Shifa International Islamabad on 12/09/2026. Mobile [PHONE], CNIC [CNIC].",
  "text_evidence": {
    "entities": [["Bilal Hussain","PERSON"],["28","DATE"],["3 days","DATE"],
                 ["Shifa International","ORG"],["Islamabad","GPE"],
                 ["12/09/2026","DATE"],["CNIC","ORG"],["CNIC","ORG"]],
    "fields": {"date": ["12/09/2026"]},
    "urgent_terms": ["fever", "chills", "sweating"]
  },
  "decision": "REFER",
  "reasons": ["image model: P(Parasitized) = 1.00",
              "note mentions: fever, chills, sweating"],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

| Case | Image P(Parasitized) | Urgent terms in note | Decision | Reasons |
|---|---|---|---|---|
| 1: Parasitized cell + malaria note | 0.999 | fever, chills, sweating | REFER | image and note |
| 2: Uninfected cell + "high fever, chills and headache" | 0.068 | fever, chills, headache | REFER | note only |

Case 2 shows image/text disagreement: one Uninfected cell is weak evidence against malaria (a smear holds thousands of cells), so the system refers the case, shows both pieces of evidence, and a human asks for more cells or a repeat test.

**Wrong entities in the record:** `"CNIC"` is tagged ORG twice (the word itself and the mask tag), and `"28"` (an age) is tagged DATE. The decision does not use them, so there is no effect today; precision of the entity list is $\le 6/8=0.75$. I would filter mask-tag entities and add an `aged <number>` rule.

---

## 6. Privacy: masking before logging and extraction

`mask_pii` replaces the CNIC, mobile and email patterns with `[CNIC]`, `[PHONE]`, `[EMAIL]` **before** any entity or field extraction, because extraction copies text into new places (entities, fields, logs, JSON). If identifiers were extracted first, they would already sit in those copies and masking the original note afterwards could not remove them.

| Check | Input | Result |
|---|---|---|
| Positive | `CNIC 35202-1234567-1, call 0300-1234567, a@b.pk` | `CNIC [CNIC], call [PHONE], [EMAIL]` (3 of 3 masked) |
| Negative | `Room 0300 on floor 3, order 12345678901234 shipped` | unchanged (no false masking) |
| Leak search in the JSON record | digits `0300`, `1234567`, `35202` | **0 leaks**; `fields` has no `mobile` or `cnic` entry, so extraction ran on the masked note |

---

## Limitations and honest notes

| # | Limitation | Consequence |
|---|---|---|
| 1 | Test set has only 149 spam messages; NER sets have 15–17 gold entities | Word-vs-char difference (3 messages) and NER scores are indicative, not precise |
| 2 | Names, addresses and MRNs are **not** masked (the masked note still contains "Bilal Hussain" and a PERSON entity) | Privacy risk is reduced, not removed; a name/MRN masker is needed before real use |
| 3 | Training data is English UK SMS | Roman Urdu spam (0.41 / 0.49) and disguised spam (0.19 / 0.23) are missed; retrain with local data and consider a threshold below 0.5 |
| 4 | Shortcut features (`uk`, `gt`, `lt`) | Scores may drop on SMS from another country or platform |
| 5 | Both vectorisers lowercase text (`lowercase=True` default) | Capitalisation ("FREE") is lost as a spam cue; test with `lowercase=False` |
| 6 | Synthetic invoice gives CER 0.000 | Not representative of phone photos |
| 7 | Example cell images come from the dataset the Week 6 model was trained on | P = 0.999 illustrates the record format; it is not an accuracy measurement |
| 8 | Case record uses threshold 0.5 (Week 6 recall-oriented threshold was 0.283) | Image-side sensitivity could be higher with the tuned threshold |
| 9 | Decision support only | A qualified human makes the decision |

## Reproduce

Kaggle inputs: `uciml/sms-spam-collection-dataset` (`spam.csv`), `iarunava/cell-images-for-detecting-malaria`, and the Week 6 checkpoint `week6_model.pt`. Run `week7-text-and-multimodal.ipynb` top to bottom (seeds: `random_state=42`).
