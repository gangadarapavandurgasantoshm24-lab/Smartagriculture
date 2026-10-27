# SmartAgriculture — Complete Project Audit & Technical Blueprint

**Project Path:** `D:\SmartAgriculture`  
**Application:** SmartAgri AI — Clinical Plant Pathology Screening & Grad-CAM Explainability  
**Audit Date:** October 27, 2026  
**Auditor:** Antigravity AI Pair Programming System  
**Frameworks:** Python 3.12.3, Streamlit 1.56.0, TensorFlow 2.21.0, Keras 3.14.0, OpenCV 5.0.0, Plotly 7.1.0  
**Active Checkpoint:** `models/best_model.keras` (SHA-256: `4C8CA02F5F77B321DCE84877F02C2E9A05DBDF5217CDCC9538E800156C1735EE`)  
**Status:** Read-Only Complete Repository Audit for PPT, Live Demo, and Viva Examination  

---

## 1. Executive Summary

SmartAgri AI is a web-based agricultural computer vision system designed to provide preliminary leaf disease screening and explainable artificial intelligence (XAI) diagnostics for farmers, agronomists, and extension workers. The application takes an uploaded digital photograph of a crop leaf, performs standardized preprocessing, evaluates the image through a 3-block convolutional neural network (CNN), computes the top candidate conditions across a closed set of 38 classes (14 crop species), generates a Gradient-weighted Class Activation Map (Grad-CAM) targeting convolutional layer `conv2d_3`, provides source-grounded agronomic and chemical guidance, and generates an on-demand, print-ready diagnostic assessment report.

### Key Audit Findings
1. **Model Scope & Checkpoint Integrity:** The active model (`models/best_model.keras`) is a Sequential CNN with **14,843,878 trainable parameters** and a 38-way softmax output. Its SHA-256 checksum is verified and untouched. On the PlantVillage validation benchmark (17,572 images), the model achieves **93.22% overall accuracy**, a macro F1-score of **0.9319**, and a weighted F1-score of **0.9320**. On a balanced 50-sample-per-class test split (1,900 images), the application model achieves **92.61% accuracy**.
2. **Dataset Nature:** The model was trained on the Kaggle "New Plant Diseases Dataset (Augmented)" derived from the benchmark PlantVillage corpus (70,295 training images with pre-computed offline augmentations; 17,572 validation images). All benchmark images feature severed single leaves photographed against uniform, flat laboratory backgrounds. The repository contains **no regional or Andhra Pradesh-specific field training images**.
3. **Field Generalization & Known Failure Modes:** Real-world field images featuring natural outdoor backgrounds, soil, multiple overlapping leaves, or human fingers can cause cross-crop confusion. The repository actively documents four verified field failure cases in `data/user_failure_cases/` where an apple leaf with round brown lesions was classified as `Peach___Bacterial_spot` (98.14%) and a healthy maize leaf photographed in-hand was classified as `Pepper,_bell___healthy` (99.99%). The application introduces explicit **Cross-Crop Ambiguity Detection** and an optional **Target Crop Disambiguation** control to mitigate these errors during inference without modifying model weights.
4. **Clinical Disease Guidance & CIBRC Chemical Standards:** The guidance catalog (`data/disease_guidance.json`) covers all 38 classes. Exactly **12 classes represent healthy foliage**, for which pesticide recommendations are strictly suppressed with a prominent disclaimer (*"Healthy leaf detected — no disease treatment or pesticide is required based on this prediction"*). Of the 26 diseased classes, **10 classes** have verified commercial trade names registered in India with the Central Insecticides Board & Registration Committee (CIBRC/PPQS) comprising **22 distinct commercial formulations** (e.g., *Score 25 EC*, *Dithane M-45*, *Amistar Top*, *Curzate M8*, *Ridomil Gold 68 WP*, *Kavach*, *Oberon*, *Confidor 17.8 SL*, *Actara 25 WG*, *Rogor 30 EC*). For the remaining **16 diseased classes**, commercial trade names are deliberately withheld and display a mandatory KVK expert confirmation notice.
5. **Session-Scoped Storage & Reporting:** Scan history is strictly session-scoped in-memory (`src/session_history.py`) using Streamlit session state. No user scan data or images are written to disk or SQLite during normal operations. Each scan features an isolated, printable A4 HTML diagnostic report with high-contrast typography, embedded Base64 original and Grad-CAM images, top-3 candidates, symptoms, IPM cultural care, and verified chemical formulations.
6. **Codebase Health & Testing:** The repository contains **41 automated tests** across 6 test modules (`test_refinement.py`, `test_inference.py`, `test_guidance.py`, `test_gradcam.py`, `test_disease_data.py`, `test_database.py`), achieving a **100% pass rate**. No hardcoded credentials or API keys exist.

---

## 2. Repository Structure

```
D:\SmartAgriculture
├── app.py                         # Streamlit application entry point & 5-page sidebar routing
├── requirements.txt               # Documented project dependencies
├── README.md                      # Project documentation and setup instructions
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration (forest green palette)
├── models/
│   ├── best_model.keras           # Active application CNN checkpoint (14.8M params, SHA-256: 4C8CA02F...)
│   └── class_names.json           # Exact 38-class alphabetical label list
├── cnn/
│   ├── model.py                   # CNN architecture definition (build_cnn_model)
│   ├── train.py                   # Training script with EarlyStopping & ModelCheckpoint
│   ├── data_loader.py             # tf.data loading pipeline with Rescaling(1/255)
│   ├── evaluate.py                # Validation evaluation script
│   ├── error_analysis.py          # Error analysis and confusion matrix calculation
│   ├── predict_image.py           # CLI prediction utility
│   ├── training_history.csv       # 10-epoch training metrics log
│   ├── classification_report.txt  # Text evaluation report on 17,572 validation images
│   ├── confusion_matrix.csv       # Raw 38x38 confusion matrix
│   ├── best_model.keras           # Alternative experimental checkpoint (SHA-256: B3B3A162...)
│   └── experiments/               # Controlled online augmentation experiment scripts
├── preprocessing/
│   └── preprocess_image.py        # Standalone image preprocessing utility
├── src/
│   ├── config.py                  # Global paths, constants, design tokens, and theme colors
│   ├── inference.py               # Preprocessing, model caching, predict_leaf, ambiguity detection
│   ├── gradcam.py                 # Grad-CAM heatmap generation at conv2d_3 & alpha overlay
│   ├── guidance.py                # 38-class clinical disease precautions & 7-tab renderer
│   ├── pesticides.py              # CIBRC verified commercial product catalog & card renderer
│   ├── session_history.py         # Session-scoped in-memory history storage (zero disk writes)
│   ├── report_generator.py        # Print-ready A4 HTML diagnostic report generator
│   ├── database.py                # Legacy SQLite database module (preserved for reference)
│   ├── disease_data.py            # Disease metadata, symptoms, and crop helpers
│   ├── evaluation_data.py         # Evaluation metrics loader & report parser
│   └── styles.py                  # Streamlit CSS styling, hero banners, and UI cards
├── pages_impl/
│   ├── dashboard.py               # Page 1: Overview KPIs, diagnostic workflow, recent feed
│   ├── detection.py               # Page 2: Leaf image upload, CNN inference, top-3, Grad-CAM, guidance
│   ├── explainability.py          # Page 3: Grad-CAM attention at conv2d_3, side-by-side comparison
│   ├── history.py                 # Page 4: In-memory session history & per-scan Print Report
│   ├── settings.py                # Page 5: 9-section master technical blueprint & encyclopedia
│   ├── performance.py             # Preserved standalone performance module (consolidated)
│   └── library.py                 # Preserved standalone library module (consolidated)
├── data/
│   ├── disease_guidance.json      # 38-class clinical guidance database (symptoms, IPM, chemicals)
│   ├── verified_pesticides.json   # 22 CIBRC registered commercial products across 10 classes
│   ├── class_analysis.csv         # Class-wise support and accuracy analysis
│   ├── scan_history.db            # Legacy SQLite database file (preserved)
│   ├── user_failure_cases/        # 4 real-world images reproducing cross-crop errors
│   ├── raw/
│   │   ├── test/test/             # 33 benchmark test leaf images from PlantVillage
│   │   └── New Plant Diseases...  # Training and validation dataset directory tree
│   ├── processed/                 # Processed dataset storage directory
│   └── scan_uploads/              # Legacy upload image storage directory (preserved)
├── reports/
│   ├── classification_report.csv  # Precision, recall, F1, support for all 38 classes
│   ├── confusion_matrix.csv       # Confusion matrix on 17,572 validation images
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   ├── normalized_confusion_matrix.png # Normalized confusion matrix heatmap
│   ├── training_curves.png        # Loss and accuracy curves across 50 epochs
│   ├── misclassified_gallery.png  # Visual gallery of misclassified validation samples
│   ├── misclassified_samples.csv  # Metadata of misclassified validation images
│   ├── error_analysis.md          # In-depth analysis of weak classes and directed errors
│   ├── eval_per_class_app_model.csv # Per-class evaluation on 1,900 balanced samples
│   ├── eval_per_class_cnn_model.csv # Per-class evaluation on CNN checkpoint
│   ├── experiment_comparison.csv  # Baseline vs augmentation experiment table
│   └── augmentation_experiment.md # Augmentation experiment documentation
├── scripts/
│   ├── reproduce_failures.py      # Script reproducing reported field failure images
│   ├── test_four_specific_images.py # Inference verification on failure case images
│   ├── analyze_dataset.py         # Dataset class distribution and balance analyzer
│   ├── apply_guidance_audit.py    # Chemical guidance audit and verification tool
│   ├── run_cross_crop_evaluation.py # Cross-crop ambiguity evaluation tool
│   └── test_safe_inference.py     # Inference safety and edge case tester
└── tests/
    ├── test_refinement.py         # 12 tests: 5 pages, 38 classes, in-memory history, reports
    ├── test_inference.py          # 10 tests: preprocessing, prediction, top-k, ambiguity
    ├── test_guidance.py           # 8 tests: schema validation, healthy logic, chemical rules
    ├── test_gradcam.py            # 3 tests: layer conv2d_3, gradient tape, heatmap bounds
    ├── test_disease_data.py       # 3 tests: crop list, disease metadata, class analysis
    └── test_database.py           # 5 tests: legacy database schema and CRUD operations
```

### Module Responsibilities & Key Files

| Module / File | Responsibility & Architectural Role | Status |
| :--- | :--- | :--- |
| **`app.py`** | Application entry point. Configures page layout, injects CSS, defines exactly five sidebar navigation items, and routes execution to `pages_impl/`. | Active (Verified) |
| **`src/config.py`** | Central configuration file. Defines file paths, model parameters (`INPUT_SHAPE=(256, 256, 3)`, `TARGET_CONV_LAYER='conv2d_3'`), and color hex codes. | Active (Verified) |
| **`src/inference.py`** | Core inference engine. Implements cached model loading (`st.cache_resource`), deterministic label parsing, `preprocess_image` (EXIF correction, resizing, normalization), `predict_leaf`, cross-crop ambiguity detection, and target crop disambiguation. | Active (Verified) |
| **`src/gradcam.py`** | Gradient-weighted Class Activation Mapping engine. Builds multi-output Keras 3 model, computes tape gradients of target class w.r.t. `conv2d_3`, calculates global average pooled weights, and performs alpha-blended heatmap overlays. | Active (Verified) |
| **`src/guidance.py`** | Disease precautions and treatment guidance module. Loads `data/disease_guidance.json`, validates schema against model classes, and renders the 7-tab guidance interface. | Active (Verified) |
| **`src/pesticides.py`** | Commercial pesticide reference module. Loads `data/verified_pesticides.json`, enforces strict suppression on healthy foliage, renders verified CIBRC product cards, and falls back to KVK notices for unverified classes. | Active (Verified) |
| **`src/session_history.py`** | In-memory session history manager. Stores scans in `st.session_state["session_scans"]` with Base64 image caching. Does not write to disk or SQLite. Provides `clear_session_history`. | Active (Verified) |
| **`src/report_generator.py`** | Diagnostic report generator. Constructs standalone, print-ready A4 HTML reports with embedded Base64 images, top-3 candidates, symptoms, IPM care, verified products, and `@media print` CSS. | Active (Verified) |
| **`pages_impl/settings.py`** | Unified master reference page. Consolidates 9 complete sections covering application overview, layer specifications, searchable 38-class library, dataset info, validation metrics, confusion matrix, Grad-CAM theory, and safety disclaimers. | Active (Verified) |
| **`pages_impl/performance.py`** | Legacy standalone performance page. Preserved for code integrity; content fully consolidated into `pages_impl/settings.py`. Not in navigation. | Preserved (Inactive) |
| **`pages_impl/library.py`** | Legacy standalone disease library page. Preserved for code integrity; content fully consolidated into `pages_impl/settings.py`. Not in navigation. | Preserved (Inactive) |
| **`src/database.py`** | Legacy SQLite persistence engine. Preserved for code integrity; user scan history now uses session-scoped in-memory storage. | Preserved (Inactive) |

---

## 3. Dataset Audit

### 3.1 Dataset Identity & Provenance
- **Dataset Name:** Kaggle "New Plant Diseases Dataset (Augmented)" derived from the benchmark **PlantVillage** dataset (Hughes and Salathé, 2015).
- **Physical Location:** `data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/`
- **Dataset Structure:** Two pre-partitioned directories: `train/` and `valid/`, each containing exactly 38 subdirectories named in the format `Crop___Disease_or_Status`.
- **Supported Classes:** Exactly **38 classes**.
- **Supported Crops:** Exactly **14 crop species** (Apple, Blueberry, Cherry, Corn/Maize, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato).

### 3.2 Image Counts & Distribution
- **Training Images:** **70,295 images**
- **Validation Images:** **17,572 images**
- **Total Images:** **87,867 images**
- **Split Ratio:** ~80% training (70,295) and ~20% validation (17,572).
- **Independent Test Set:** `data/raw/test/test/` contains **33 sample test images** representing various classes.
- **Image Dimensions & Format:** Original images are 256×256 pixels in RGB JPEG format.

### 3.3 Class-Wise Distribution (Validation Split: 17,572 images)

| # | Exact Class Name | Crop | Condition | Support (Val) |
| :-: | :--- | :--- | :--- | :-: |
| 0 | `Apple___Apple_scab` | Apple | Apple scab | 504 |
| 1 | `Apple___Black_rot` | Apple | Black rot | 497 |
| 2 | `Apple___Cedar_apple_rust` | Apple | Cedar apple rust | 440 |
| 3 | `Apple___healthy` | Apple | Healthy | 502 |
| 4 | `Blueberry___healthy` | Blueberry | Healthy | 454 |
| 5 | `Cherry_(including_sour)___Powdery_mildew` | Cherry | Powdery mildew | 421 |
| 6 | `Cherry_(including_sour)___healthy` | Cherry | Healthy | 456 |
| 7 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Corn (maize) | Gray leaf spot | 410 |
| 8 | `Corn_(maize)___Common_rust_` | Corn (maize) | Common rust | 477 |
| 9 | `Corn_(maize)___Northern_Leaf_Blight` | Corn (maize) | Northern Leaf Blight | 477 |
| 10 | `Corn_(maize)___healthy` | Corn (maize) | Healthy | 465 |
| 11 | `Grape___Black_rot` | Grape | Black rot | 472 |
| 12 | `Grape___Esca_(Black_Measles)` | Grape | Esca (Black Measles) | 480 |
| 13 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape | Leaf blight | 430 |
| 14 | `Grape___healthy` | Grape | Healthy | 423 |
| 15 | `Orange___Haunglongbing_(Citrus_greening)` | Orange | Citrus greening | 503 |
| 16 | `Peach___Bacterial_spot` | Peach | Bacterial spot | 459 |
| 17 | `Peach___healthy` | Peach | Healthy | 432 |
| 18 | `Pepper,_bell___Bacterial_spot` | Pepper, bell | Bacterial spot | 478 |
| 19 | `Pepper,_bell___healthy` | Pepper, bell | Healthy | 497 |
| 20 | `Potato___Early_blight` | Potato | Early blight | 485 |
| 21 | `Potato___Late_blight` | Potato | Late blight | 485 |
| 22 | `Potato___healthy` | Potato | Healthy | 456 |
| 23 | `Raspberry___healthy` | Raspberry | Healthy | 445 |
| 24 | `Soybean___healthy` | Soybean | Healthy | 505 |
| 25 | `Squash___Powdery_mildew` | Squash | Powdery mildew | 434 |
| 26 | `Strawberry___Leaf_scorch` | Strawberry | Leaf scorch | 444 |
| 27 | `Strawberry___healthy` | Strawberry | Healthy | 456 |
| 28 | `Tomato___Bacterial_spot` | Tomato | Bacterial spot | 425 |
| 29 | `Tomato___Early_blight` | Tomato | Early blight | 480 |
| 30 | `Tomato___Late_blight` | Tomato | Late blight | 463 |
| 31 | `Tomato___Leaf_Mold` | Tomato | Leaf Mold | 470 |
| 32 | `Tomato___Septoria_leaf_spot` | Tomato | Septoria leaf spot | 436 |
| 33 | `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Two-spotted spider mite | 435 |
| 34 | `Tomato___Target_Spot` | Tomato | Target Spot | 457 |
| 35 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Tomato Yellow Leaf Curl Virus | 490 |
| 36 | `Tomato___Tomato_mosaic_virus` | Tomato | Tomato mosaic virus | 448 |
| 37 | `Tomato___healthy` | Tomato | Healthy | 481 |

### 3.4 Preprocessing and Augmentation Analysis
- **Offline Augmentation:** The training directory contains pre-computed augmented image files saved to disk (identifiable by filenames such as `_90deg`, `_270deg`, `_new30degFlipLR`). This increased the training volume from the original ~54,305 raw images to 70,295 images.
- **Online Pipeline:** In `cnn/data_loader.py`, images are loaded using `tf.keras.utils.image_dataset_from_directory` with `image_size=(256, 256)` and normalized via `tf.keras.layers.Rescaling(1.0 / 255.0)`. No dynamic online rotations or distortions were applied in the baseline training loader.
- **Controlled Augmentation Experiment:** A smoke-tested experiment is documented in `cnn/experiments/augmentation_experiment.py` and `reports/augmentation_experiment.md` introducing mild flips, rotations (±0.04), and zoom (0.08), but the baseline checkpoint was preserved.

### 3.5 Critical Dataset Limitations & Domain Shift
1. **Laboratory Environment Bias:** PlantVillage leaves were severed from the plant and photographed on uniform gray, black, or white poster boards under controlled illumination.
2. **Absence of Real Field Context:** The dataset lacks natural soil, weeds, multiple overlapping canopy leaves, variable sunlight, shadows, insect damage, or human hands holding leaves.
3. **Geographic Limitation:** The dataset does not include major Indian/Andhra Pradesh crops such as paddy/rice, cotton, sugarcane, mango, chili, or groundnut.

---

## 4. Model Architecture & Training Audit

### 4.1 Checkpoint Specifications

| Checkpoint Property | Application Checkpoint (`models/best_model.keras`) | Alternative Checkpoint (`cnn/best_model.keras`) |
| :--- | :--- | :--- |
| **Physical Path** | `D:\SmartAgriculture\models\best_model.keras` | `D:\SmartAgriculture\cnn\best_model.keras` |
| **File Size** | 178,174,133 bytes | 178,174,114 bytes |
| **SHA-256 Checksum** | `4C8CA02F5F77B321DCE84877F02C2E9A05DBDF5217CDCC9538E800156C1735EE` | `B3B3A1621E3C37FCF7B901A24C3F2548B66CAC82B6A1C4D04102DC1ABC1788E0` |
| **Origin** | Verified application model configured in `src/config.py` | Training experiment checkpoint generated by `cnn/train.py` |
| **Validation Accuracy**| 92.61% (balanced 1,900 split) / 93.22% (full validation) | 93.22% (17,572 full validation) |

### 4.2 Layer-by-Layer Architectural Blueprint

The model is defined in `cnn/model.py` as a Sequential network:

```
Input: (None, 256, 256, 3) [RGB Image Tensor, float32, normalized to [0.0, 1.0]]
 │
 ├── [Block 1] Conv2D(32 filters, 3x3 kernel, relu, valid padding) ──> Output: (None, 254, 254, 32) [896 params]
 ├──           MaxPooling2D(pool_size=2x2, strides=2)               ──> Output: (None, 127, 127, 32) [0 params]
 │
 ├── [Block 2] Conv2D(64 filters, 3x3 kernel, relu, valid padding) ──> Output: (None, 125, 125, 64) [18,496 params]
 ├──           MaxPooling2D(pool_size=2x2, strides=2)               ──> Output: (None, 62, 62, 64)   [0 params]
 │
 ├── [Block 3] Conv2D(128 filters, 3x3 kernel, relu, valid padding) ──> Output: (None, 60, 60, 128)  [73,856 params] <── TARGET GRAD-CAM LAYER
 ├──           MaxPooling2D(pool_size=2x2, strides=2)               ──> Output: (None, 30, 30, 128)  [0 params]
 │
 ├── Flatten()                                                      ──> Output: (None, 115200)       [0 params]
 ├── Dense(128 units, activation='relu')                            ──> Output: (None, 128)          [14,745,728 params]
 ├── Dropout(rate=0.5)                                              ──> Output: (None, 128)          [0 params]
 └── Dense(38 units, activation='softmax')                          ──> Output: (None, 38)           [4,902 params]
```

### 4.3 Parameter Breakdown
- **Trainable Parameters:** **14,843,878** (56.62 MB)
- **Non-Trainable Parameters:** **0** (0.00 B)
- **Total Model Parameters:** **14,843,878**
- **Optimizer Parameters:** 29,687,758 (113.25 MB)
- **Input Tensor Dimensions:** `(None, 256, 256, 3)`
- **Output Tensor Dimensions:** `(None, 38)`

### 4.4 Training Hyperparameters & Execution History
As documented in `cnn/train.py` and `cnn/training_history.csv`:
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam` (default learning rate: 0.001)
- **Batch Size:** 32
- **Configured Maximum Epochs:** 20
- **Callbacks Implemented:**
  1. `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`
  2. `ModelCheckpoint(filepath='cnn/best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True)`
- **Actual Training Run:** The training run ran for **10 epochs** before early stopping triggered:
  - Epoch 1: Loss 1.7862, Acc 47.65%, Val Loss 0.6623, Val Acc 80.50%
  - Epoch 4: Loss 0.5503, Acc 82.25%, Val Loss 0.2650, Val Acc 91.53%
  - **Epoch 7 (Optimal):** Loss 0.3278, Acc 89.25%, **Val Loss 0.2211**, **Val Acc 93.22%**
  - Epoch 8: Loss 0.2816, Acc 90.64%, Val Loss 0.2377, Val Acc 93.00%
  - Epoch 9: Loss 0.2639, Acc 91.29%, Val Loss 0.2438, Val Acc 92.94%
  - Epoch 10: Loss 0.2287, Acc 92.45%, Val Loss 0.2477, Val Acc 93.11%
- Because validation loss did not improve beyond epoch 7 (0.2211) for 3 consecutive epochs (epochs 8, 9, 10), early stopping halted training at epoch 10 and restored weights from Epoch 7.

---

## 5. Inference Pipeline & Application Architecture

### 5.1 End-to-End Data Flow

```
[ User Uploads Leaf Image (.jpg, .png) ]
                    │
                    ▼
[ Image Decoding & EXIF Transpose ] (PIL ImageOps.exif_transpose)
                    │
                    ▼
[ Optional Center Crop ] (Shortest-edge square crop for non-square photos)
                    │
                    ▼
[ Spatial Resizing ] (OpenCV INTER_AREA -> 256x256 RGB)
                    │
                    ▼
[ Float32 Normalization ] (Pixel values / 255.0 strictly into [0.0, 1.0])
                    │
                    ▼
[ Tensor Batch Expansion ] (Shape: (1, 256, 256, 3))
                    │
                    ▼
[ CNN Forward Pass ] (models/best_model.keras via st.cache_resource)
                    │
                    ▼
[ 38-Way Softmax Distribution ] (Sum of probabilities = 1.0)
                    │
                    ▼
[ Raw Candidate Extraction ] (Top predicted class & Top-3 ranked list)
                    │
     ┌──────────────┴──────────────┐
     ▼                             ▼
[ Cross-Crop Ambiguity Check ]   [ Target Crop Disambiguation ]
(Flags multi-species splits)     (Calculates within-crop relative %)
     │                             │
     └──────────────┬──────────────┘
                    │
                    ▼
[ Grad-CAM Backpropagation ] (Targeting layer conv2d_3 via GradientTape)
                    │
                    ▼
[ Heatmap Generation & Alpha Blending ] (OpenCV JET colormap, alpha=0.45)
                    │
                    ▼
[ Clinical Disease Guidance Retrieval ] (data/disease_guidance.json)
                    │
                    ▼
[ Verified CIBRC Pesticide Retrieval ] (data/verified_pesticides.json)
(Suppressed for healthy; Trade names displayed for 10 classes; KVK notice for 16)
                    │
                    ▼
[ In-Memory Session History Storage ] (src/session_history.py, st.session_state)
                    │
                    ▼
[ Print-Ready A4 Diagnostic Report ] (src/report_generator.py, window.print())
```

### 5.2 Application Calling Graph

```
app.py (Navigation & Master Shell)
 │
 ├── pages_impl/dashboard.py
 │    ├── src/evaluation_data.py (KPI metrics)
 │    ├── src/session_history.py (get_session_scan_stats, get_session_scans)
 │    └── src/disease_data.py
 │
 ├── pages_impl/detection.py
 │    ├── src/inference.py (predict_leaf, preprocess_image, parse_class_label)
 │    ├── src/gradcam.py (generate_gradcam_heatmap, apply_gradcam_overlay)
 │    ├── src/guidance.py (get_guidance, render_guidance_tabs)
 │    ├── src/pesticides.py (render_commercial_product_cards)
 │    └── src/session_history.py (add_session_scan)
 │
 ├── pages_impl/explainability.py
 │    ├── src/gradcam.py (generate_gradcam_heatmap, apply_gradcam_overlay)
 │    └── src/inference.py (predict_leaf)
 │
 ├── pages_impl/history.py
 │    ├── src/session_history.py (get_session_scans, clear_session_history)
 │    └── src/report_generator.py (generate_printable_html_report)
 │
 └── pages_impl/settings.py
      ├── src/inference.py (load_model, load_class_names)
      ├── src/evaluation_data.py (load_classification_report_csv, load_training_history)
      ├── src/guidance.py (get_all_guidance)
      └── src/pesticides.py (get_pesticide_audit_summary)
```

---

## 6. Grad-CAM Explainability Audit

### 6.1 Theoretical Foundations
Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) generates visual explanations for decisions from convolutional neural networks without requiring architectural changes or retraining.

In this project, Grad-CAM answers: **"Which spatial regions of this plant leaf caused the CNN to predict this specific disease class?"**

### 6.2 Target Layer Rationale: `conv2d_3`
- The model contains three convolutional layers: `conv2d_1` (32 filters), `conv2d_2` (64 filters), and `conv2d_3` (128 filters).
- `conv2d_3` is the final convolutional layer directly before spatial pooling and flattening.
- **Why `conv2d_3` is targeted:** Lower convolutional layers (`conv2d_1`) capture generic edges, lines, and textures. Higher convolutional layers (`conv2d_3`) capture complex, class-specific pathological features such as chlorotic halos, concentric target rings, fungal pustules, and necrotic lesions while retaining spatial coordinate information (feature map dimensions: 60×60).

### 6.3 Mathematical Formulation & Implementation
1. **Differentiable Graph Reconstruction:** In Keras 3, the loaded Sequential model is wrapped in a multi-output functional graph (`src/gradcam.py`, `get_grad_model`) outputting both the target activation tensor $A$ from `conv2d_3` and the final class score $y$.
2. **Gradient Computation:** Using `tf.GradientTape()`, the gradient of the target class logit $y^c$ is computed with respect to each feature map activation $A^k$ of `conv2d_3`:
   $$\frac{\partial y^c}{\partial A^k}$$
3. **Global Average Pooling of Gradients:** Neuron importance weights $\alpha_k^c$ are calculated by averaging gradients across width ($u$) and height ($v$):
   $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
   In `src/gradcam.py`, line 97: `pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))` (shape: 128).
4. **Weighted Linear Combination & ReLU:** The 128 feature maps are weighted by their importance, summed, and passed through a Rectified Linear Unit (ReLU) to isolate features positively correlated with the class:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   In `src/gradcam.py`, line 105: `heatmap = tf.maximum(heatmap, 0.0)`.
5. **Upsampling & Alpha Blending:** The coarse 60×60 heatmap is normalized to $[0.0, 1.0]$, upscaled using bicubic interpolation to 256×256, mapped to RGB using the OpenCV JET colormap, and blended onto the original image with $\alpha = 0.45$:
   $$I_{\text{overlay}} = (1 - \alpha) \cdot I_{\text{original}} + \alpha \cdot H_{\text{colormap}}$$

### 6.4 Clinical Caveat
Grad-CAM highlights regions that influenced the model's numerical activation. **It does not constitute independent biological proof that a pathogen is present.** If the model focuses on background soil, a finger, or a leaf tip rather than a lesion, Grad-CAM alerts the agronomist to an ungrounded prediction.

---

## 7. Disease Guidance & Pesticide Audit

### 7.1 Statistical Summary of Guidance Records (`data/disease_guidance.json`)
- **Total Classes Covered:** Exactly **38 classes** (100% catalog coverage).
- **Healthy Foliage Classes:** **12 classes** (31.6% of classes).
- **Diseased Foliage Classes:** **26 classes** (68.4% of classes).
- **Classes with Plain-Language Symptoms:** **38 classes** (100%).
- **Classes with Preventive Cultural Measures:** **38 classes** (100%).
- **Classes with Immediate Non-Chemical Sanitation:** **38 classes** (100%).

### 7.2 Chemical Treatment Status Distribution

| Status Category | Count | Classes Included | Application Action |
| :--- | :---: | :--- | :--- |
| **Inappropriate (Healthy Foliage)** | **12** | All 12 `*___healthy` classes | Chemical advice strictly blocked. Displays: *"Healthy leaf detected — no disease treatment or pesticide is required based on this prediction."* |
| **Appropriate (Verified CIBRC Fungal/Bacterial)** | **7** | Apple scab, Corn common rust, Corn northern leaf blight, Potato early blight, Potato late blight, Tomato early blight, Tomato late blight | Displays verified CIBRC commercial trade names, formulations, dosages, and safety intervals. |
| **Appropriate (Acaricide / Miticide Only)** | **1** | `Tomato___Spider_mites Two-spotted_spider_mite` | Displays verified registered miticides (*Oberon*, *Omite 57 EC*). Explicitly notes fungicides are ineffective. |
| **Vector Management Only (Pathogen Incurable)** | **2** | `Orange___Haunglongbing_(Citrus_greening)`, `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Displays registered insecticides targeting the insect vectors (*Diaphorina citri* psyllid; *Bemisia tabaci* whitefly). Explicitly notes systemic pathogens cannot be cured by spraying. |
| **Inappropriate (Viral Infection)** | **1** | `Tomato___Tomato_mosaic_virus` | Chemical spraying blocked. Instructs rogueing and sanitation. |
| **Inappropriate (Trunk/Vascular Esca)** | **1** | `Grape___Esca_(Black_Measles)` | Chemical foliar spraying blocked. Internal wood rot cannot be cured by topical sprays. |
| **Requires Expert Confirmation** | **13** | Apple black rot, Apple cedar rust, Cherry powdery mildew, Grape black rot, Grape leaf blight, Peach bacterial spot, Pepper bacterial spot, Squash powdery mildew, Strawberry leaf scorch, Tomato bacterial spot, Tomato leaf mold, Tomato septoria, Orange citrus greening (direct bactericide) | Commercial brand names withheld. Displays mandatory KVK expert confirmation advisory. |
| **Unavailable (Insufficient Evidence)** | **1** | `Tomato___Target_Spot` | Commercial brand names withheld. Displays KVK confirmation advisory. |

### 7.3 Verified CIBRC Commercial Pesticide Products Catalog (`data/verified_pesticides.json`)

The catalog contains **22 verified commercial product registrations** across **10 distinct crop condition classes** verified against the Central Insecticides Board & Registration Committee (CIBRC), Directorate of Plant Protection, Quarantine & Storage (PPQS), Ministry of Agriculture & Farmers Welfare, Government of India:

1. **`Apple___Apple_scab`**
   - *Score 25 EC* (Difenoconazole 25% EC; Syngenta India; 0.3 ml/L; PHI: 30 days)
   - *Dithane M-45* (Mancozeb 75% WP; Indofil Industries; 2.0–2.5 g/L; PHI: 15–20 days)
2. **`Corn_(maize)___Common_rust_`**
   - *Dithane M-45* (Mancozeb 75% WP; Indofil Industries; 2.0 g/L; PHI: 15 days)
   - *Amistar* (Azoxystrobin 23% SC; Syngenta India; 1.0 ml/L; PHI: 10 days)
3. **`Corn_(maize)___Northern_Leaf_Blight`**
   - *Amistar Top* (Azoxystrobin 18.2% + Difenoconazole 11.4% SC; Syngenta India; 1.0 ml/L; PHI: 14 days)
   - *Dithane M-45* (Mancozeb 75% WP; Indofil Industries; 2.0 g/L; PHI: 15 days)
4. **`Orange___Haunglongbing_(Citrus_greening)`** *(Vector: Asian Citrus Psyllid)*
   - *Confidor 17.8 SL* (Imidacloprid 17.8% SL; Bayer CropScience; 0.5 ml/L; PHI: 15 days)
   - *Rogor 30 EC* (Dimethoate 30% EC; FMC India; 1.5–2.0 ml/L; PHI: 15 days)
5. **`Potato___Early_blight`**
   - *Dithane M-45* (Mancozeb 75% WP; Indofil Industries; 2.0 g/L; PHI: 14 days)
   - *Score 25 EC* (Difenoconazole 25% EC; Syngenta India; 0.5 ml/L; PHI: 14 days)
   - *Kavach* (Chlorothalonil 75% WP; Syngenta India; 2.0 g/L; PHI: 14 days)
6. **`Potato___Late_blight`**
   - *Ridomil Gold 68 WP* (Metalaxyl-M 4% + Mancozeb 64% WP; Syngenta India; 2.5 g/L; PHI: 14 days)
   - *Dithane M-45* (Mancozeb 75% WP; Indofil Industries; 2.0–2.5 g/L; PHI: 14 days)
7. **`Tomato___Early_blight`**
   - *Dithane M-45* (Mancozeb 75% WP; Indofil Industries; 2.0 g/L; PHI: 7 days)
   - *Amistar* (Azoxystrobin 23% SC; Syngenta India; 1.0 ml/L; PHI: 5 days)
   - *Kavach* (Chlorothalonil 75% WP; Syngenta India; 2.0 g/L; PHI: 7 days)
8. **`Tomato___Late_blight`**
   - *Curzate M8* (Cymoxanil 8% + Mancozeb 64% WP; DuPont / Corteva Agriscience; 2.5 g/L; PHI: 7 days)
   - *Dithane M-45* (Mancozeb 75% WP; Indofil Industries; 2.0 g/L; PHI: 7 days)
9. **`Tomato___Spider_mites Two-spotted_spider_mite`**
   - *Oberon* (Spiromesifen 22.9% SC; Bayer CropScience; 0.8–1.0 ml/L; PHI: 3 days)
   - *Omite 57 EC* (Propargite 57% EC; Dhanuka Agritech; 2.0 ml/L; PHI: 7 days)
10. **`Tomato___Tomato_Yellow_Leaf_Curl_Virus`** *(Vector: Bemisia tabaci)*
    - *Confidor 17.8 SL* (Imidacloprid 17.8% SL; Bayer CropScience; 0.3–0.5 ml/L; PHI: 5 days)
    - *Actara 25 WG* (Thiamethoxam 25% WG; Syngenta India; 0.4 g/L; PHI: 5 days)

### 7.4 Fallback Advisory for Unverified Classes (16 Classes)
For all 16 diseased classes without verified CIBRC commercial trade names, the UI and generated reports suppress commercial brands and render:
> *"Verified Indian trade name unavailable. Active ingredient: [name and concentration]. Consult the current registered product label or your local KVK before purchasing or applying."*

---

## 8. Feature-by-Feature Functional Audit

| # | Application Feature | Status | Primary Code Files | Verification Method | Actual Test Evidence / Findings | Known Issues / Caveats |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Dashboard** | Verified | `pages_impl/dashboard.py`, `src/styles.py` | Browser UI & Code Review | Renders hero banner, 4 KPI cards (Accuracy, Scope, Layer, Session Scans), workflow buttons, and recent scans feed. | Session scans feed resets when browser session expires. |
| **2** | **Disease Detection Workflow** | Verified | `pages_impl/detection.py`, `src/inference.py` | Automated tests & live inference | Accepts user uploads, benchmark presets (33 images), and failure cases (4 images). Resizes, normalizes, and runs inference. | Outdoor photos with background clutter can cause cross-crop confusion. |
| **3** | **Top-3 Prediction Probabilities** | Verified | `src/inference.py`, `pages_impl/detection.py` | `test_inference.py` (Test 1, 2) | Renders horizontal Plotly bar chart with exact percentages. Validates sum of probabilities. | Softmax outputs can exhibit overconfidence on out-of-distribution inputs. |
| **4** | **Healthy-Leaf Guidance** | Verified | `src/guidance.py`, `src/pesticides.py` | `test_refinement.py` (Test 4, 11) | Displays: *"Healthy leaf detected — no disease treatment or pesticide is required based on this prediction."* Pesticides blocked. | If a healthy leaf is misclassified as diseased, guidance will reflect the disease. |
| **5** | **Disease Prevention & IPM Management** | Verified | `src/guidance.py`, `data/disease_guidance.json` | `test_guidance.py` (Test 1–8) | 7-tab interface: Overview, Symptoms, Preventive Measures, Non-Chemical Management, Chemical Control, Sanitation, Sources. | Requires user to evaluate actual field feasibility. |
| **6** | **Commercial Chemical Guidance** | Verified | `src/pesticides.py`, `data/verified_pesticides.json` | `test_refinement.py` (Test 7) | Verified cards render trade names, active ingredients, formulations, dosages, PHI, REI, and CIBRC citations. | 16 classes lack confirmed commercial products; fall back to KVK notice. |
| **7** | **Grad-CAM Explainability** | Verified | `src/gradcam.py`, `pages_impl/explainability.py` | `test_gradcam.py` (Test 1–3) | GradientTape targeting `conv2d_3`. Renders original and heatmap overlay side-by-side with 5 colormaps. | Explains feature attention, not biological ground truth. |
| **8** | **Scan History** | Verified | `src/session_history.py`, `pages_impl/history.py` | `test_refinement.py` (Test 5, 6) | In-memory session logging in `st.session_state`. Filters by crop/status. Clear History wipes memory. Zero disk writes. | History does not persist across hard browser restarts or new tabs. |
| **9** | **Print Report for Each Scan** | Verified | `src/report_generator.py`, `pages_impl/history.py` | `test_refinement.py` (Test 8, 9, 10) | Individual **🖨️ Print Report** button on each card. Generates standalone A4 HTML report with `@media print` and `window.print()`. | User must select "Save as PDF" in browser print dialog. |
| **10** | **Settings/About Technical Blueprint** | Verified | `pages_impl/settings.py` | `test_refinement.py` (Test 1) | Consolidated 9-section master blueprint. Renders layer table, searchable 38-class library, evaluation metrics, and charts. | Large page; uses expanders for readability. |
| **11** | **Evaluation Charts & Confusion Matrix** | Verified | `reports/confusion_matrix.png`, `reports/training_curves.png` | `pages_impl/settings.py`, `src/evaluation_data.py` | Displays interactive Plotly loss/accuracy curves, raw CM, normalized CM, and misclassified gallery. | Missing image paths display graceful warning fallback. |
| **12** | **38 Class Mappings** | Verified | `models/class_names.json`, `src/inference.py` | `test_refinement.py` (Test 3) | Verified 38 classes across 14 crops. Alphabetical ordering matches data loader. | Adding classes requires full model retraining. |
| **13** | **Navigation & Theme** | Verified | `app.py`, `.streamlit/config.toml`, `src/styles.py` | `test_refinement.py` (Test 1) | Exactly 5 sidebar pages. Forest green (`#173F35`), secondary green (`#2F6B4F`), sage (`#E8F2E8`), off-white (`#F7F8F4`). | High-contrast CSS overrides light-theme text issues. |
| **14** | **Error Handling & Empty States** | Verified | `pages_impl/detection.py`, `pages_impl/explainability.py` | Code audit & test runs | Unsupported formats show validation errors; empty history displays helper prompts; missing images in reports show placeholders. | Extremely corrupted image files raise descriptive Streamlit exceptions. |

---

## 9. Model Performance & Evaluation Metrics

### 9.1 Overall Benchmark Performance (PlantVillage Validation Split: 17,572 images)
- **Validation Accuracy:** **93.22%** (16,380 correct out of 17,572 images)
- **Macro Average Precision:** **0.9333** (93.33%)
- **Macro Average Recall:** **0.9318** (93.18%)
- **Macro Average F1-Score:** **0.9319** (93.19%)
- **Weighted Average Precision:** **0.9332** (93.32%)
- **Weighted Average Recall:** **0.9322** (93.22%)
- **Weighted Average F1-Score:** **0.9320** (93.20%)
- **Sampled Application Evaluation:** **92.61% accuracy** on 1,900 balanced validation samples (50 per class).

### 9.2 Top Performing Classes by Recall

| Class Name | Precision | Recall | F1-Score | Validation Support |
| :--- | :---: | :---: | :---: | :---: |
| `Corn_(maize)___Common_rust_` | 99.37% | **99.58%** | 99.48% | 477 |
| `Strawberry___healthy` | 98.26% | **99.12%** | 98.69% | 456 |
| `Orange___Haunglongbing_(Citrus_greening)` | 96.32% | **99.01%** | 97.65% | 503 |
| `Blueberry___healthy` | 95.52% | **98.68%** | 97.07% | 454 |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 99.07% | **98.60%** | 98.83% | 430 |
| `Corn_(maize)___healthy` | 98.71% | **98.49%** | 98.60% | 465 |

### 9.3 Weakest Performing Classes by Recall

| Class Name | Precision | Recall | F1-Score | Validation Support | Primary Misclassification Target |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Peach___Bacterial_spot` | 97.80% | **77.34%** | 86.37% | 459 | Confused with `Tomato___Late_blight` (22) and `Tomato___Bacterial_spot` (21) |
| `Tomato___Septoria_leaf_spot` | 81.97% | **78.21%** | 80.05% | 436 | Confused with `Tomato___Early_blight` and `Tomato___Target_Spot` |
| `Tomato___Early_blight` | 81.56% | **78.33%** | 79.91% | 480 | Confused with `Tomato___Late_blight` (37) |
| `Tomato___Target_Spot` | 83.63% | **81.62%** | 82.61% | 457 | Confused with `Tomato___Spider_mites` (28) |
| `Corn_(maize)___Cercospora_leaf_spot` | 91.15% | **82.93%** | 86.85% | 410 | Confused with `Corn_(maize)___Northern_Leaf_Blight` (67) |
| `Tomato___Late_blight` | 76.31% | **84.88%** | 80.37% | 463 | Confused with `Potato___Late_blight` (23) and `Tomato___Early_blight` (22) |

### 9.4 Most Frequent Directed Confusion Pairs

| Rank | True Class | Model Prediction | Misclassified Count | Plausible Visual Cause |
| :-: | :--- | :--- | :-: | :--- |
| 1 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn_(maize)___Northern_Leaf_Blight` | **67** | Both produce elongated necrotic lesions parallel to leaf veins. |
| 2 | `Tomato___Early_blight` | `Tomato___Late_blight` | **37** | Overlapping dark brown foliar necrosis on identical tomato foliage. |
| 3 | `Grape___Black_rot` | `Grape___Esca_(Black_Measles)` | **31** | Both produce dark reddish-brown foliar spots on grape leaves. |
| 4 | `Tomato___Spider_mites Two-spotted_spider_mite` | `Tomato___Target_Spot` | **29** | Fine chlorotic stippling resembles early pinpoint necrotic target spots. |
| 5 | `Tomato___Target_Spot` | `Tomato___Spider_mites Two-spotted_spider_mite` | **28** | Reciprocal visual confusion of foliar spotting. |
| 6 | `Tomato___Late_blight` | `Potato___Late_blight` | **23** | Same pathogen (*Phytophthora infestans*) causing identical water-soaked lesions across nightshade relatives. |
| 7 | `Peach___Bacterial_spot` | `Tomato___Late_blight` | **22** | Cross-species lesion similarity. |
| 8 | `Tomato___Late_blight` | `Tomato___Early_blight` | **22** | Reciprocal foliar necrosis confusion. |
| 9 | `Peach___Bacterial_spot` | `Tomato___Bacterial_spot` | **21** | Shared bacterial lesion morphology (*Xanthomonas* species). |
| 10 | `Potato___Late_blight` | `Tomato___Late_blight` | **21** | Reciprocal *Phytophthora* cross-crop confusion. |

### 9.5 Verified Field Failure Cases (`data/user_failure_cases/`)
Executed via `scripts/reproduce_failures.py`:
1. **`images_apple.jpg`** (Apple leaf with circular brown lesions photographed outdoors):
   - Model prediction: `Peach___Bacterial_spot` (98.14%)
   - Second candidate: `Corn_(maize)___Cercospora_leaf_spot` (1.83%)
   - Cause: Background noise and outdoor lighting triggered non-apple filters.
2. **`images_apple2.jpg`** (Apple leaf with lesions):
   - Model prediction: `Peach___Bacterial_spot` (72.40%)
   - Second candidate: `Corn_(maize)___Northern_Leaf_Blight` (25.14%)
   - Cause: Cross-crop division of probability mass across Peach and Corn.
3. **`images_corn health.jpg`** (Healthy maize leaf with white background border):
   - Model prediction: `Pepper,_bell___healthy` (99.99%)
   - Second candidate: `Corn_(maize)___healthy` (0.01%)
   - Cause: Letterbox framing distortion and aspect ratio stretching. Mitigated when **Center Leaf Focus (Square Crop)** is enabled.
4. **`images_corn.jpg`** (Diseased maize leaf):
   - Model prediction: `Corn_(maize)___Northern_Leaf_Blight` (99.93%)
   - Accurate prediction.

---

## 10. Testing & Quality Assurance

### 10.1 Automated Test Suite Execution Results
The complete automated test suite was executed in the workspace:

```
Command: pytest tests/
Platform: Windows (Python 3.12.3, pytest-9.1.1, pluggy-1.7.0)
Total Tests Discovered: 41
Total Passed: 41 (100%)
Total Failed: 0
Total Skipped: 0
Execution Time: 18.11 seconds
Status: ALL TESTS PASSED
```

### 10.2 Module-Wise Test Coverage

| Test File | Test Count | Features & Requirements Covered |
| :--- | :---: | :--- |
| **`tests/test_refinement.py`** | 12 | Navigation has exactly 5 pages; Model Performance & Library removed; 38 classes intact; in-memory session history isolation; zero SQLite writes; Clear History; CIBRC commercial products for 10 classes; unverified fallback for 16 classes; Print Report generation for 4 distinct clinical scenarios. |
| **`tests/test_inference.py`** | 10 | Model loading; class names loading; label parsing; image preprocessing (RGB, float32, [0,1], 256x256); `predict_leaf`; top-3 probabilities; cross-crop ambiguity detection; target crop disambiguation. |
| **`tests/test_guidance.py`** | 8 | 38 guidance records exist; schema validation; healthy leaf pesticide prohibition; bacterial disease non-fungicide rule; viral disease non-cure rule; source URL presence. |
| **`tests/test_gradcam.py`** | 3 | Target layer `conv2d_3` exists; multi-output model graph construction; non-zero gradient flow; normalized heatmap output range [0.0, 1.0]. |
| **`tests/test_disease_data.py`** | 3 | 14 supported crops helper; disease info retrieval; class analysis CSV schema integrity. |
| **`tests/test_database.py`** | 5 | Legacy SQLite schema initialization; scan record insertion; record retrieval; scan deletion; database clearing. |

### 10.3 Missing Test Cases & Recommendations
- Automated end-to-end browser tests using Playwright/Selenium to simulate client-side PDF export via `window.print()`.
- Load testing under concurrent multi-user Streamlit sessions.
- Unit tests validating EXIF rotation across diverse smartphone camera orientations.

---

## 11. Technology Stack & Dependencies

### 11.1 Verified Technologies & Environment Versions

| Technology / Library | Version | Purpose in SmartAgri AI |
| :--- | :--- | :--- |
| **Python** | 3.12.3 | Core programming language. |
| **Streamlit** | 1.56.0 | Web application framework for interactive UI, state management, and real-time visualization. |
| **TensorFlow** | 2.21.0 | Deep learning framework executing CNN model computation and automatic differentiation. |
| **Keras** | 3.14.0 | High-level neural network API defining sequential topology and multi-output Grad-CAM functional graph. |
| **OpenCV (`cv2`)** | 5.0.0 | Computer vision library for image resizing (`INTER_AREA`, `INTER_CUBIC`), color conversions, and colormap overlays. |
| **Pillow (`PIL`)** | 10.3.0 | Image I/O library handling EXIF orientation transposition and format decoding. |
| **Pandas** | 2.2.2 | Structured data manipulation for evaluation CSVs and confusion matrices. |
| **NumPy** | 2.5.4 | Numerical computing for tensor manipulation, probability sorting, and image array slicing. |
| **Plotly** | 7.1.0 | Interactive charting library rendering horizontal probability bars and training history curves. |
| **Matplotlib** | 3.8.4 | Static plotting engine used in offline report generation and confusion matrix rendering. |
| **Scikit-Learn** | 1.4.2 | Machine learning evaluation library computing precision, recall, F1-scores, and confusion matrices. |
| **Pytest** | 9.1.1 | Automated testing framework executing regression test suites. |

---

## 12. Security, Privacy & Reliability

1. **In-Memory Session Isolation:** User scan images and diagnosis records are maintained strictly within `st.session_state` in RAM. No user photographs are stored to persistent disk or committed to databases.
2. **Sanitized Input Handling:** `preprocess_image` safely decodes images through PIL and validates tensor dimensions, preventing buffer overflow or shell execution vulnerabilities from uploaded files.
3. **No Hardcoded Secrets:** Static analysis confirmed zero API keys, database passwords, or private access tokens exist in the repository.
4. **HTML Injection Defense:** Standalone reports are constructed using sanitized templates. User notes are rendered as plain text within table cells.
5. **Client-Side Print Isolation:** Reports are printed using sandboxed browser execution (`window.print()`), eliminating server-side headless browser vulnerabilities.

---

## 13. Known Limitations & Future Improvements

### 13.1 Current Limitations
1. **Closed Taxonomic Scope:** The system can classify only the 38 pre-trained classes across 14 crops. Submitting unsupported regional crops (e.g., rice, cotton, mango, chili) will force a false prediction.
2. **Laboratory vs. Field Domain Shift:** Trained exclusively on PlantVillage lab images. Complex natural field backgrounds, soil, multiple overlapping leaves, or fingers can degrade confidence and trigger cross-crop confusion.
3. **Softmax Calibration:** Standard softmax outputs can produce artificially high confidence on out-of-distribution leaf inputs.
4. **Session-Scoped History:** Scan history does not persist across browser restarts or new tabs.
5. **CPU Execution Latency:** Inference runs on native Windows CPU (average latency ~35–45ms per image).

### 13.2 Realistic Future Improvements
1. **Out-of-Distribution (OOD) Detection:** Implement an energy-based or Mahalanobis distance rejection filter to automatically reject non-leaf images or unsupported crop species.
2. **Temperature Scaling & Confidence Calibration:** Calibrate softmax probabilities to align model confidence with empirical accuracy.
3. **Field Dataset Fine-Tuning:** Collect and fine-tune on in-situ Indian field imagery captured under natural weather and lighting conditions.
4. **Mobile Deployment via TensorFlow Lite (TFLite):** Quantize the 14.8M parameter model to INT8 (reducing size from 170MB to ~15MB) for offline on-device mobile execution.
5. **Dual-Branch Architecture:** Implement a hierarchical classifier (Crop Detection Head -> Disease Condition Head) to structurally eliminate cross-crop misclassification.

---

## 14. PPT-Ready Presentation Outline (12 Slides)

### Slide 1: Title & Project Identity
- **Title:** SmartAgri AI: Explainable Plant Leaf Disease Diagnostics
- **Subtitle:** Deep Learning & Grad-CAM Visual Attention for Precision Plant Pathology
- **Presenter Name / Candidate Details:** [Insert Your Name, Roll Number, Department]
- **Key Visual:** System Logo, Forest Green Theme Banner, Application Hero Badge.
- **Verified Facts:** 38 classes, 14 crops, 93.22% validation accuracy, Keras 3 & TensorFlow.
- **Claims to Avoid:** Do not claim the model diagnoses all Indian crops or replaces human agronomists.

### Slide 2: Problem Statement & Motivation
- **Title:** The Agricultural Challenge: Early Disease Diagnosis
- **Bullet Points:**
  - Crop diseases cause 20–40% global yield loss annually.
  - Smallholder farmers lack immediate access to certified plant pathologists.
  - Misdiagnosis leads to inappropriate pesticide use, input waste, and chemical resistance.
  - Existing "black box" computer vision apps provide predictions without visual reasoning.
- **Key Visual:** Misclassification examples or comparison of healthy vs diseased field leaves.
- **Verified Facts:** Diagnostic delay directly impacts farm economics; visual explanations build farmer trust.

### Slide 3: Objectives & Scope
- **Title:** Project Objectives & Architectural Scope
- **Bullet Points:**
  - Build an end-to-end deep learning web screening tool.
  - Implement real-time Gradient-weighted Class Activation Mapping (Grad-CAM) at layer `conv2d_3`.
  - Provide source-grounded agronomic care with CIBRC-registered commercial chemical trade names.
  - Ensure ethical AI standards: strictly suppress chemical advice on healthy foliage.
- **Key Visual:** Overview diagram of the 5-page application layout.
- **Verified Facts:** Exactly 38 classes, 14 crops, 12 healthy classes, 10 CIBRC verified classes.

### Slide 4: Dataset & Preprocessing Pipeline
- **Title:** Dataset Characteristics & Pipeline Standardization
- **Bullet Points:**
  - Benchmark PlantVillage Augmented Dataset: 87,867 total images (70,295 train; 17,572 validation).
  - Preprocessing: EXIF orientation correction, optional center crop, resize to 256×256 RGB.
  - Normalization: Strict `float32 / 255.0` scaling into range $[0.0, 1.0]$.
  - Critical Limitation: Laboratory images on flat poster boards; domain shift on field photos.
- **Key Visual:** Sample benchmark images from `data/raw/test/test/` showing 256×256 resolution.
- **Verified Facts:** 80/20 train/valid split; 38 deterministic alphabetical classes.

### Slide 5: Model Architecture Blueprint
- **Title:** Convolutional Neural Network Topology
- **Bullet Points:**
  - Sequential 3-Block Architecture designed for high-resolution leaf feature extraction.
  - Conv Block 1: 32 filters (3×3) -> MaxPool (2×2) -> Edges and low-level textures.
  - Conv Block 2: 64 filters (3×3) -> MaxPool (2×2) -> Leaf spot patterns and venation.
  - Conv Block 3: 128 filters (3×3) -> MaxPool (2×2) -> Complex pathological lesion features (`conv2d_3`).
  - Classifier Head: Flatten (115,200) -> Dense (128, ReLU) -> Dropout (0.5) -> Softmax (38).
- **Key Visual:** Layer parameter diagram or architecture table from Section 4.2.
- **Verified Facts:** Exactly 14,843,878 trainable parameters; Adam optimizer; sparse categorical crossentropy.

### Slide 6: Model Training & Convergence History
- **Title:** Empirical Training Dynamics & Checkpoint Selection
- **Bullet Points:**
  - Trained with batch size 32 on native environment.
  - Callbacks: `EarlyStopping(patience=3)` and `ModelCheckpoint(save_best_only=True)`.
  - Optimal validation accuracy reached at **Epoch 7: 93.22%** (Validation Loss: **0.2211**).
  - Training completed at Epoch 10 as EarlyStopping prevented overfitting on augmented data.
- **Key Visual:** Training and validation loss/accuracy curves (`reports/training_curves.png`).
- **Verified Facts:** 10 recorded epochs; best validation accuracy: 93.22%; best validation loss: 0.2211.

### Slide 7: AI Explainability: Grad-CAM at `conv2d_3`
- **Title:** Explainable AI via Grad-CAM Visual Attention
- **Bullet Points:**
  - Overcomes the black-box dilemma: provides visual transparency for agronomic decisions.
  - Computes gradients of predicted class score with respect to feature maps at `conv2d_3`.
  - Global average pooled gradients define spatial importance weights $\alpha_k^c$.
  - Generates 60×60 activation map, upsamples to 256×256, and blends JET colormap ($\alpha=0.45$).
  - Identifies whether the CNN evaluated disease lesions or irrelevant background artifacts.
- **Key Visual:** Side-by-side comparison of original leaf photo and Grad-CAM overlay from `pages_impl/explainability.py`.
- **Verified Facts:** Target layer `conv2d_3` contains 128 channels with 60×60 spatial resolution.

### Slide 8: Evaluation Metrics & Confusion Matrix Analysis
- **Title:** Quantitative Evaluation & Error Analysis
- **Bullet Points:**
  - Macro F1-Score: **0.9319**; Weighted F1-Score: **0.9320** on 17,572 validation samples.
  - Highest Recall: Corn Common Rust (99.58%), Strawberry healthy (99.12%), Citrus greening (99.01%).
  - Lowest Recall: Peach Bacterial Spot (77.34%), Tomato Septoria (78.21%), Tomato Early Blight (78.33%).
  - Frequent Confusion: Corn Gray Leaf Spot vs Northern Leaf Blight (67 errors due to parallel streak similarity).
- **Key Visual:** Normalized Confusion Matrix heatmap (`reports/normalized_confusion_matrix.png`).
- **Verified Facts:** 16,380 correct predictions out of 17,572; 1,192 errors cataloged in `misclassified_samples.csv`.

### Slide 9: Real-World Generalization & Ambiguity Detection
- **Title:** Addressing Field Domain Shift & Failure Modes
- **Bullet Points:**
  - Documented Failure Mode: Outdoor lighting and hands trigger cross-crop confusion (Apple confused as Peach).
  - Implemented Safeguard 1: **Cross-Crop Ambiguity Warning** alerts users when probability splits across crops.
  - Implemented Safeguard 2: **Target Crop Disambiguation** allows farmer to confirm crop species (e.g. Apple) to filter background noise.
  - Implemented Safeguard 3: **Center Square Focus** eliminates non-square letterbox and framing distortion.
- **Key Visual:** UI screenshot of Ambiguity Warning and Crop Disambiguation selector.
- **Verified Facts:** Verified on 4 physical failure case images in `data/user_failure_cases/`.

### Slide 10: Clinical Guidance & Verified CIBRC Chemicals
- **Title:** Source-Grounded Treatment & Statutory Pesticide Standards
- **Bullet Points:**
  - Ethical AI Rule: 12 healthy classes receive zero pesticide advice; preventive IPM only.
  - Verified Commercial Trade Names: 22 CIBRC registered products across 10 classes (e.g., *Score 25 EC*, *Dithane M-45*, *Amistar Top*, *Curzate M8*).
  - Formulation details: Active ingredient, concentration, dosage per liter, PHI, REI, and safety equipment.
  - Unverified Fallback: 16 classes display mandatory KVK confirmation notice; commercial brands withheld.
- **Key Visual:** Commercial product card UI screenshot from Disease Detection Tab 5.
- **Verified Facts:** Sourced from PPQS / CIBRC, Ministry of Agriculture & Farmers Welfare, India.

### Slide 11: Live Demonstration Workflow & Printable Reports
- **Title:** Demonstration Architecture & Print-Ready PDF Reports
- **Bullet Points:**
  - Page 1: Dashboard with live KPIs and quick start.
  - Page 2: Disease Detection with top-3 candidates, Grad-CAM, and 7-tab guidance.
  - Page 3: AI Explainability deep-dive with interactive colormaps.
  - Page 4: Scan History with in-memory session isolation.
  - Individual **🖨️ Print Report** generates standalone, print-ready A4 reports suitable for saving as PDF.
- **Key Visual:** Screenshot of the print-ready A4 diagnostic report layout.
- **Verified Facts:** Zero disk writes for user privacy; full report isolation per scan.

### Slide 12: Conclusion & Future Roadmap
- **Title:** Project Conclusion & Future Enhancements
- **Bullet Points:**
  - Successfully refined an explainable plant pathology screening system with 93.22% benchmark accuracy.
  - Demonstrated end-to-end safety: ambiguity alerts, healthy leaf safeguards, and CIBRC chemical verification.
  - Future Roadmap: Out-of-Distribution (OOD) rejection filters, INT8 TFLite mobile deployment, and expanded Indian regional crop datasets.
- **Key Visual:** Final application summary card and repository badge.
- **Verified Facts:** 41/41 automated tests passed; clean 5-page sidebar navigation.

---

## 15. Comprehensive Viva Preparation (25 Questions & Answers)

### Category 1: Problem Motivation & Dataset
1. **Q: Why did you choose plant leaf disease classification as your project?**  
   *A:* Plant diseases cause up to 40% loss in crop yields globally. Smallholder farmers often misdiagnose diseases or apply inappropriate chemicals, causing financial loss and environmental harm. An automated, explainable screening tool provides instant diagnostic assistance and IPM guidance directly in the field.
2. **Q: What dataset was used, and what are its primary limitations?**  
   *A:* The model was trained on the Kaggle "New Plant Diseases Dataset (Augmented)" derived from PlantVillage, consisting of 87,867 images (70,295 train, 17,572 validation) across 38 classes and 14 crops. Its primary limitation is that leaves were photographed severed on uniform laboratory backgrounds; real field images with soil, weeds, or hands introduce a domain shift that can cause misclassifications.
3. **Q: How many crops and classes does your model support? Can it classify paddy (rice) or cotton?**  
   *A:* Exactly 14 crop species and 38 conditions. It cannot classify paddy, cotton, mango, or chili because they were not part of the 38 classes in the training dataset. Predicting an unsupported crop would result in a forced, erroneous classification.
4. **Q: What is the difference between offline and online data augmentation?**  
   *A:* Offline augmentation creates and saves altered images (rotations, flips) to disk prior to training, expanding the dataset size. Online augmentation applies random transformations dynamically on batches in memory during training using CPU/GPU operations. This dataset utilized pre-computed offline augmentations.
5. **Q: Did you find any class imbalance in the validation dataset?**  
   *A:* The validation split is relatively well-balanced across all 38 classes, ranging from 410 images (`Corn Gray Leaf Spot`) to 505 images (`Soybean healthy`), with an average of approximately 462 images per class.

### Category 2: Model Architecture & Deep Learning Concepts
6. **Q: Explain the architecture of your CNN model.**  
   *A:* It is a Sequential CNN consisting of three convolutional blocks followed by a dense classification head. Block 1 has 32 filters (3×3), Block 2 has 64 filters (3×3), and Block 3 has 128 filters (3×3), each followed by 2×2 MaxPooling. The spatial tensor (30×30×128) is flattened to 115,200 elements, passed to a Dense layer of 128 units with ReLU, regularized with Dropout (0.5), and outputted through a 38-unit Dense layer with Softmax.
7. **Q: How many trainable parameters does your model have?**  
   *A:* Exactly 14,843,878 trainable parameters (56.62 MB). The vast majority (14.7M) reside in the first Dense layer connecting the 115,200 flattened features to 128 hidden neurons.
8. **Q: Why did you use `sparse_categorical_crossentropy` instead of `categorical_crossentropy`?**  
   *A:* `sparse_categorical_crossentropy` accepts integer class labels (0 to 37) directly, eliminating the memory overhead of one-hot encoding 87,867 vectors of dimension 38 during training.
9. **Q: What is the purpose of the Dropout layer (0.5)?**  
   *A:* During training, Dropout randomly deactivates 50% of the neurons in the dense hidden layer on each forward pass. This prevents co-adaptation of features and significantly reduces overfitting.
10. **Q: What optimizer was used and what was the training stopping criterion?**  
    *A:* The Adam optimizer with default learning rate (0.001). Training utilized `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`. Training halted at epoch 10 because validation loss reached its global minimum (0.2211) at epoch 7 and failed to improve for 3 subsequent epochs.

### Category 3: Explainable AI & Grad-CAM
11. **Q: What is Grad-CAM and why is it necessary in agriculture?**  
    *A:* Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainability technique that visualizes the spatial attention of a CNN by computing gradients of the predicted class score with respect to convolutional feature maps. It is essential in agriculture to verify that the network is focusing on actual disease lesions rather than background soil, sunlight glare, or human fingers.
12. **Q: Which layer was selected for Grad-CAM in your model and why?**  
    *A:* Layer `conv2d_3`, the third and final convolutional layer (128 filters, 60×60 resolution). Final convolutional layers provide the best trade-off between high-level semantic disease patterns and spatial detail, whereas earlier layers only capture primitive edges.
13. **Q: Walk me through the mathematical steps of your Grad-CAM implementation.**  
    *A:* First, backpropagate the target class score $y^c$ to the activation maps $A^k$ of `conv2d_3`. Second, compute neuron importance weights $\alpha_k^c$ via global average pooling of gradients: $\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$. Third, compute the weighted sum of feature maps and apply ReLU: $\text{ReLU}(\sum_k \alpha_k^c A^k)$. Finally, normalize and upscale the 60×60 heatmap to 256×256 and alpha-blend it onto the RGB image.
14. **Q: Does a strong Grad-CAM heatmap prove that the diagnosis is correct?**  
    *A:* No. Grad-CAM only shows what visual regions influenced the model's numerical output. If a model misclassifies an image, Grad-CAM accurately shows the regions that misled the model. It is an explanatory tool, not independent biological verification.

### Category 4: Evaluation Metrics & Error Analysis
15. **Q: What is the difference between Precision and Recall in plant pathology?**  
    *A:* Precision measures the proportion of predicted disease instances that were truly that disease (minimizing false alarms). Recall measures the proportion of actual diseased leaves correctly detected by the model (minimizing missed infections). In agriculture, high recall is vital to prevent undetected epidemics.
16. **Q: What is the macro F1-score vs weighted F1-score?**  
    *A:* Macro F1 calculates the unweighted mean of F1-scores across all 38 classes, treating every class equally regardless of sample count (0.9319). Weighted F1 weights each class's F1-score by its support in the validation set (0.9320).
17. **Q: Which disease class had the lowest recall and why?**  
    *A:* `Peach___Bacterial_spot` had the lowest recall (77.34%), frequently confused with `Tomato Late Blight` and `Tomato Bacterial Spot`. This is caused by visual similarities in small, dark water-soaked bacterial lesions across different plant species.
18. **Q: Why did the model confuse Corn Gray Leaf Spot with Northern Leaf Blight 67 times?**  
    *A:* Both diseases affect maize foliage and produce elongated rectangular or cigar-shaped necrotic lesions running parallel to leaf veins. Under lower image resolution, their textural features overlap significantly.

### Category 5: Clinical Guidance, Safety & Pesticide Verification
19. **Q: How does the application handle predictions of healthy leaves?**  
    *A:* All 12 healthy classes strictly prohibit chemical treatments and display the prominent notice: *"Healthy leaf detected — no disease treatment or pesticide is required based on this prediction."* The UI suppresses chemical cards and recommends only routine monitoring, balanced nutrition, and field hygiene.
20. **Q: Can viral or phytoplasma diseases be cured with chemical fungicides?**  
    *A:* No. Systemic plant viruses (e.g., Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus) and phloem-limited bacteria (Citrus greening) cannot be cured by foliar chemical sprays once inside the vascular tissue. Guidance for TYLCV and Citrus Greening focuses strictly on vector control (whiteflies and psyllids) and rogueing infected plants.
21. **Q: How did you verify the commercial pesticide recommendations?**  
    *A:* Recommendations were audited against current CIBRC (Central Insecticides Board & Registration Committee) registrations published by the Directorate of Plant Protection, Quarantine & Storage (PPQS), Ministry of Agriculture, Government of India. Exactly 22 commercial products across 10 classes were confirmed for exact crop, disease, formulation, dosage, and pre-harvest interval (PHI).
22. **Q: What happens if a chemical recommendation cannot be verified?**  
    *A:* Commercial trade names are withheld, and the system renders: *"Verified Indian trade name unavailable. Active ingredient: [name and concentration]. Consult the current registered product label or your local KVK before purchasing or applying."*

### Category 6: Application Architecture & Implementation Details
23. **Q: How is Scan History stored, and why did you eliminate persistent database writes?**  
    *A:* Scan history is stored in-memory using Streamlit's `st.session_state["session_scans"]`. Persistent SQLite writes were removed to ensure complete user privacy and eliminate disk clutter. Scans and Base64 images exist only for the duration of the active browser session.
24. **Q: How does the "Print Report" feature work?**  
    *A:* In `src/report_generator.py`, an individual HTML document is constructed for the selected scan featuring a white A4 background, high-contrast dark typography, embedded Base64 images, top-3 candidates, symptoms, IPM care, and `@media print` CSS. An interactive button triggers native browser printing (`window.print()`), enabling farmers to print or save as PDF.
25. **Q: What is the "Target Crop Disambiguation" feature and why is it needed?**  
    *A:* When farmers upload field photos containing background clutter or fingers, the model can split probability across multiple crops. The user can optionally select their target crop (e.g., "Apple"). The inference engine filters classes to that species and computes relative within-crop probabilities, eliminating foreign background noise while preserving raw probabilities in the metadata.

---

## 16. Unverified Claims vs. Verified Repository Facts

| Claim / Topic | Status | Evidence in Repository | Required Viva Presentation Boundary |
| :--- | :---: | :--- | :--- |
| **Model covers all Indian / AP crops** | **UNVERIFIED (False)** | `models/class_names.json` covers exactly 14 species and 38 classes. | State clearly: *The model is restricted to the 38 classes of the PlantVillage benchmark and does not support rice, cotton, or chili.* |
| **Model validation accuracy is 93.22%** | **VERIFIED** | `reports/classification_report.csv`, `reports/error_analysis.md`, `cnn/training_history.csv` | Accurately cite 93.22% on 17,572 validation images; 92.61% on balanced 1,900 split. |
| **Model is reliable for in-situ field diagnosis** | **UNVERIFIED (Caveated)** | `data/user_failure_cases/` reproduces cross-crop errors on field photos. | State clearly: *The model exhibits domain shift when exposed to outdoor backgrounds; ambiguity warnings and crop confirmation are required.* |
| **Target layer for Grad-CAM is `conv2d_3`** | **VERIFIED** | `src/config.py:34`, `src/gradcam.py:22`, `models/best_model.keras` | Layer `conv2d_3` is the final conv layer with 128 filters and 60×60 feature maps. |
| **Pesticide recommendations are registered in India** | **VERIFIED (Partial)** | `data/verified_pesticides.json` (10 classes verified with CIBRC; 16 classes withheld). | Distinguish that exactly 10 classes have verified Indian trade names; 16 classes require KVK expert confirmation. |
| **Scan history is saved to permanent SQLite database** | **UNVERIFIED (Historical)** | `src/session_history.py` uses `st.session_state`; `src/database.py` is inactive. | State clearly: *Scan history is session-scoped in-memory for user privacy; no disk writes occur.* |
| **Application contains exactly 5 sidebar pages** | **VERIFIED** | `app.py:53` defines Dashboard, Detection, Explainability, History, Settings/About. | Performance and Library were consolidated into Settings/About and removed from navigation. |
| **All automated tests pass** | **VERIFIED** | `pytest tests/` executes 41 tests with 100% pass rate in 18.11s. | Confirmed via test runner execution. |

---
*End of Complete Project Audit & Technical Blueprint.*
