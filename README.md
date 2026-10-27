# Smart Agriculture: AI Plant Disease Diagnostics & Grad-CAM Explainability

An end-to-end deep learning and agricultural intelligence platform for plant leaf pathology detection, featuring interactive 38-class classification, empirical model performance analytics, and real-time **Grad-CAM (Gradient-Weighted Class Activation Mapping)** visual explainability targeting layer `conv2d_3`.

---

## 🌿 Overview & Key Capabilities

- **38 Botanical Classes across 14 Crops:** Full coverage of the PlantVillage benchmark dataset (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato).
- **Application Model:** Verified CNN checkpoint (`models/best_model.keras`) with 92.61% validation accuracy and 100% 1-to-1 label mapping to `models/class_names.json`.
- **Live Grad-CAM Explainability:** Live backpropagation through `conv2d_3` (128 filters) using `tf.GradientTape`, generating normalized activation heatmaps and blended overlays highlighting pathological lesions on leaf tissue.
- **Persistent Diagnostic History:** Integrated SQLite database (`data/scan_history.db`) saving timestamps, confidences, agronomist field notes, and visual thumbnails with CSV export.
- **Empirical Evaluation Hub:** Visualizations of actual validation benchmarks on 17,572 samples (93.22% baseline accuracy, macro F1 0.9319, confusion matrices, and training history curves).
- **Disease & Agronomic Library:** Searchable encyclopedia of symptoms, causal pathogens, and Integrated Pest Management (IPM) guidelines.
- **Custom Design System:** Forest green (`#173F35`), secondary green (`#2F6B4F`), sage (`#E8F2E8`), warm off-white (`#F7F8F4`), modern typography, and refined responsive cards.

---

## 🏛️ Neural Network Architecture

The model uses a sequential convolutional neural network (`smart_agriculture_cnn`):

| Block | Layer Name | Configuration | Output Shape | Receptive Focus |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `input_image` | RGB Leaf Image | `(None, 256, 256, 3)` | Original resolution |
| **Block 1** | `conv2d_1` / `maxpool_1` | 32 filters, 3×3, ReLU / MaxPool(2×2) | `(None, 127, 127, 32)` | Edges, textures |
| **Block 2** | `conv2d_2` / `maxpool_2` | 64 filters, 3×3, ReLU / MaxPool(2×2) | `(None, 62, 62, 64)` | Spot contours, motifs |
| **Block 3** | **`conv2d_3`** / `maxpool_3` | **128 filters, 3×3, ReLU** / MaxPool(2×2) | `(None, 30, 30, 128)` | **Semantic lesion patterns (Grad-CAM target)** |
| **Classifier** | `flatten` | Spatial vectorization | `(None, 115200)` | 1D feature representation |
| **Head** | `dense_features` | Dense (128 units, ReLU) | `(None, 128)` | Feature integration |
| **Regularizer**| `dropout_regularization`| Dropout (rate = 0.5) | `(None, 128)` | Overfitting prevention |
| **Output** | `output_predictions` | Dense (38 units, Softmax) | `(None, 38)` | 38 Class probabilities |

**Total Parameters:** 14,846,758 (all trainable).

---

## 📁 Repository Directory Layout

```
SmartAgriculture/
├── app.py                      # Main Streamlit application entrypoint
├── models/
│   ├── best_model.keras        # Active application model (92.61% validation accuracy)
│   └── class_names.json        # 38 deterministic class names
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, theme tokens, and constants
│   ├── styles.py               # Injected CSS design system and components
│   ├── inference.py            # Cached model loading, preprocessing, inference
│   ├── gradcam.py              # Grad-CAM engine targeting conv2d_3
│   ├── database.py             # SQLite scan history and image persistence
│   ├── disease_data.py         # 38-class botanical and IPM knowledge base
│   └── evaluation_data.py      # Real reports & confusion matrix loader
├── pages_impl/
│   ├── __init__.py
│   ├── dashboard.py            # Overview telemetry, quick scans, stats
│   ├── detection.py            # Single-image upload, diagnosis, top-3 candidates
│   ├── explainability.py       # Tri-view Grad-CAM deep dive with alpha slider
│   ├── performance.py          # Real metrics, training history, confusion heatmaps
│   ├── library.py              # Filterable disease & IPM encyclopedia
│   ├── history.py              # SQLite scan history table, thumbnails, CSV export
│   └── settings.py             # Model topology breakdown & technical blueprint
├── cnn/                        # Baseline training scripts, data loader, evaluation
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data_loader.py
│   ├── predict_image.py
│   ├── training_history.csv
│   └── classification_report.txt
├── reports/                    # Pre-generated evaluation figures & CSVs
│   ├── confusion_matrix.png
│   ├── normalized_confusion_matrix.png
│   ├── misclassified_gallery.png
│   └── classification_report.csv
├── data/
│   ├── class_analysis.csv
│   ├── scan_history.db         # SQLite persistent scan database
│   ├── scan_uploads/           # Saved scan thumbnails and overlays
│   └── raw/                    # Raw PlantVillage dataset splits
├── tests/
│   ├── test_inference.py       # Tests for model, preprocessing, inference
│   ├── test_gradcam.py         # Tests for gradient calculation at conv2d_3
│   ├── test_database.py        # Tests for SQLite database CRUD
│   └── test_disease_data.py    # Tests for 38-class knowledge base coverage
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start: Running on Windows

### 1. Prerequisites
Ensure Python 3.10+ (tested on Python 3.12 64-bit on Windows) is installed.

### 2. Install Dependencies
Open **PowerShell** or **Command Prompt** in the `SmartAgriculture` directory:

```powershell
pip install -r requirements.txt
```

### 3. Run Automated Tests
Execute the comprehensive test suite to verify model loading, preprocessing, Grad-CAM gradients, and SQLite database:

```powershell
pytest tests -v
```

### 4. Launch the Streamlit Web Application
Launch the application locally:

```powershell
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🔍 How to Use the Application

1. **Dashboard:** Review key telemetry (accuracy, classes, active layer, scans count) and choose quick-start sample leaves.
2. **Disease Detection:** Upload your own leaf image (`.jpg`, `.png`, `.jpeg`) or select from the built-in dataset samples dropdown. View the top predicted condition, confidence meter, top-3 candidates, and save notes to the persistent scan history.
3. **AI Explainability:** Deep dive into how the model made its decision. Adjust the heatmap opacity slider, try different palettes (Jet, Viridis, Inferno, Turbo), and observe whether network activations localize to necrotic spots or healthy leaf tissue.
4. **Model Performance:** Explore interactive training history loss/accuracy curves, per-class recall extremes, confusion matrices, and frequent validation error pairs.
5. **Disease Library:** Filter across 14 crops and 38 conditions with instant search. Review biological symptoms and integrated pest management (IPM) practices.
6. **Scan History:** Review past field inspections with side-by-side original and Grad-CAM thumbnails, search by crop, or export the audit log to CSV.
7. **Settings/About:** Inspect the 10-layer topology, trainable parameters count, runtime system diagnostics, and advisory guidelines.

---

## 🌿 Source-Grounded Disease Precautions & Treatment Guidance

The application includes an official, source-grounded Disease Precautions & Treatment Guidance engine (`src/guidance.py` and `data/disease_guidance.json`) specifically designed for the **exact 38 PlantVillage classes** recognized by the model.

### 🛡️ Source Verification Policy & Regulatory Standards
- **Zero Hallucination Guarantee:** Treatments, active ingredients, formulations, dosages, and pre-harvest intervals are strictly derived from official registrations. AI is never allowed to fabricate pesticide advice.
- **Primary Authorities:**
  - **CIBRC / PPQS:** Central Insecticide Board & Registration Committee / Directorate of Plant Protection, Quarantine & Storage, Ministry of Agriculture & Farmers Welfare, Government of India (Major Uses of Pesticides registered under the Insecticides Act, 1968).
  - **ICAR Institutes:** ICAR-NCIPM (National Research Centre for Integrated Pest Management), ICAR-CPRI (Central Potato Research Institute), ICAR-IIVR (Indian Institute of Vegetable Research).
  - **State Agricultural Universities (SAUs):** Tamil Nadu Agricultural University (TNAU Agritech Portal), Punjab Agricultural University (PAU), Acharya N.G. Ranga Agricultural University (ANGRAU).

### 📋 7-Tab Agronomic Interface
For every diagnosis (in **Disease Detection**) and every catalog entry (in **Disease Library**), users can inspect:
1. **📖 Disease Overview:** Biological overview, causal agent scientific taxonomy, etiology, and the distinction between *disease containment* vs. an unattainable *guaranteed cure*.
2. **🔍 Symptoms:** Macroscopic farmer observations, foliage lesions, margin discoloration, sporulation, and chlorotic halos.
3. **🛡️ Prevention:** Long-term cultural measures, certified disease-free seed, crop rotation schedules, and drip irrigation hygiene to avoid leaf wetness.
4. **⚡ Immediate Management:** Non-chemical interventions, physical rogueing of diseased tissue, bio-control agents (e.g., *Trichoderma viride*, *Bacillus subtilis*), and canopy thinning for sunlight penetration.
5. **🧪 Verified Chemical Options:** Active ingredients, formulations, specific dosages (e.g. g/L, ml/L, kg/ha), application methods, and source links. Where evidence is insufficient or inappropriate, displays:  
   `"A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer."`
6. **⛑️ Application Safety:** Pre-Harvest Intervals (PHI), Re-Entry Intervals (REI), Personal Protective Equipment (PPE: gloves, respirator mask, goggles), pollinator/honeybee safety, and water body protection.
7. **📚 Sources & References:** Authoritative citations with official portal URLs, source organization type, and verification dates.

### ⚖️ Disease-Specific Rule Compliance & Regulatory Audit Breakdown
- **Healthy Crops (12 Classes):** All 12 healthy classes have chemical treatment strictly marked as **Inappropriate (Healthy Foliage)** with zero active ingredients or dosages.
- **Classes with Verified Chemical Recommendations (8 Classes):** Fungal and mite foliar conditions with active ingredients, formulations, specific dosages, and harvest intervals verified against PPQS / CIBRC Major Uses:
  - *Apple Scab* (Difenoconazole / Mancozeb)
  - *Corn Common Rust* (Azoxystrobin + Difenoconazole / Mancozeb)
  - *Corn Northern Leaf Blight* (Mancozeb / Azoxystrobin + Difenoconazole)
  - *Potato Early Blight* (Mancozeb / Chlorothalonil)
  - *Potato Late Blight* (Mancozeb / Cymoxanil + Mancozeb / Metalaxyl + Mancozeb)
  - *Tomato Early Blight* (Mancozeb / Chlorothalonil / Azoxystrobin)
  - *Tomato Late Blight* (Mancozeb / Cymoxanil + Mancozeb / Metalaxyl + Mancozeb)
  - *Tomato Spider Mites* (Spiromesifen 22.9% SC - Acaricide Only)
- **Vector-Management Only (2 Classes):** For incurable systemic pathogens, chemical options apply strictly to vector suppression; dosages for the incurable primary pathogen are not claimed:
  - *Citrus Greening / HLB* (Imidacloprid for Asian citrus psyllid vector *Diaphorina citri*)
  - *Tomato Yellow Leaf Curl Virus* (Imidacloprid / Thiamethoxam for Whitefly vector *Bemisia tabaci*)
- **Classes Requiring Expert Confirmation (13 Classes):** Unverified dosages are suppressed and direct users to local KVKs because specific crop-disease label registrations under national CIBRC tables cannot be established:
  - *Tomato & Pepper Bacterial Spot* (Antibiotic/Streptocycline use on vegetables is prohibited/restricted by regulatory orders to prevent Antimicrobial Resistance; standalone copper requires localized KVK advisory)
  - *Peach Bacterial Spot* (Fungicides ineffective; peach foliage is severely sensitive to copper phytotoxicity during vegetative growth)
  - *Cherry Powdery Mildew, Corn Gray Leaf Spot, Grape Black Rot, Grape Leaf Blight, Squash Powdery Mildew, Strawberry Leaf Scorch, Tomato Leaf Mold, Tomato Septoria, Tomato Target Spot, Apple Black Rot*
- **Inappropriate / Unavailable Non-Chemical Status (3 Classes):**
  - *Tomato Mosaic Virus* (Inappropriate - Mechanically transmitted viral infection; pesticides cannot cure plant viruses)
  - *Grape Esca* (Inappropriate - Deep vascular wood-decay trunk disease; foliar sprays ineffective)
  - *Apple Cedar Apple Rust* (Unavailable - Heteroecious rust requiring alternate host eradication; no CIBRC major use)
- **Low Model Confidence (<70%):** Automatically triggers a high-priority warning: *"Never spray chemicals solely based on a low-confidence computer vision prediction. Confirm symptoms physically in the field with a certified agricultural extension officer or local KVK before considering chemical intervention."*

### 🛠️ Updating or Auditing Guidance Records
1. Edit `data/disease_guidance.json`.
2. Ensure the class name strictly matches an existing key in `models/class_names.json`.
3. Provide all mandatory fields: `overview`, `plain_language_symptoms`, `visual_signs`, `preventive_measures`, `immediate_non_chemical_management`, `sanitation_and_irrigation`, `chemical_treatment_status`, `chemical_guidance`, and `sources`.
4. Verify all active ingredients and dosages against current CIBRC registered major uses. If a crop-specific label claim cannot be established, set `available: false`, suppress dosages, and provide the mandatory KVK disclaimer.
5. Run the validation test suite to confirm compliance:
   ```powershell
   pytest tests/test_guidance.py -v
   ```
   *(Note: Automated unit tests verify schema integrity, field presence, and data consistency. They do not constitute agronomic field validation).*

---

## 🛡️ Agricultural & Advisory Scope Notice
This software is intended as an agricultural decision-support and educational screening system. Predictions are constrained to the 38 classes of the trained model. High-consequence management actions (such as chemical applications or orchard tree removal) should always be verified by certified crop advisors or plant pathologists at your nearest Krishi Vigyan Kendra (KVK).

