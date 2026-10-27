# SmartAgriculture

SmartAgriculture is a Streamlit application for screening plant leaf images against 38 crop health classes. It combines a TensorFlow/Keras convolutional neural network with Grad-CAM visualizations, agronomic reference material, and a local scan history.

## Features

- **Disease screening:** Upload a leaf image and review the model's predicted class and confidence.
- **Visual explanations:** Generate Grad-CAM heatmaps using the model's `conv2d_3` layer.
- **Agronomic guidance:** Browse disease information, prevention practices, and sourced treatment guidance for the supported classes.
- **Scan history:** Store scan records and images locally in SQLite and export history as CSV.
- **Model and evaluation resources:** Review the model configuration and the included evaluation reports and figures.

The model covers 38 PlantVillage classes across 14 crops. It is a screening aid; field conditions and image quality can differ from the training data, so confirm important diagnoses with a qualified local agricultural advisor.

## Requirements

- Python 3.10 or later, using a version supported by TensorFlow for your operating system
- Git and Git LFS to retrieve the model checkpoints

## Setup

Clone the repository after installing Git LFS. Git LFS should download the model checkpoints during clone; if the files contain LFS pointer text, run `git lfs pull` from the repository directory.

```powershell
git lfs install
git clone https://github.com/gangadarapavandurgasantoshm24-lab/Smartagriculture.git
cd Smartagriculture
git lfs pull
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The application opens at `http://localhost:8501`. On macOS or Linux, activate the virtual environment with `source .venv/bin/activate` instead of the PowerShell activation command.

## Application navigation

The sidebar contains five sections:

1. **Dashboard** — project overview and quick-start content.
2. **Disease Detection** — image upload, model prediction, and disease guidance.
3. **AI Explainability** — Grad-CAM heatmap and image overlay.
4. **Scan History** — prior scans and CSV export.
5. **Settings/About** — model and application details.

## Repository layout

```text
app.py                  Streamlit entry point
pages_impl/             Dashboard, detection, explainability, history, and settings views
src/                    Inference, Grad-CAM, data, database, guidance, and styling modules
models/                 Application checkpoint and ordered class-name mapping
cnn/                    Model definition, training, evaluation, and prediction utilities
data/                   Guidance catalog, analysis data, and example images
reports/                Evaluation tables, plots, and experiment notes
scripts/                Data analysis and evaluation helper scripts
tests/                  Unit tests
```

The `.keras` checkpoints are stored with Git LFS. The complete PlantVillage training dataset is not included. Training utilities expect the dataset's `train` and `valid` directories under `data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/`.

## Tests

Run the automated test suite from the repository root:

```powershell
pytest tests -v
```

The tests cover application utilities such as inference, Grad-CAM, scan history, disease data, and guidance data validation. Passing software tests does not establish clinical, agronomic, or regulatory validity of a diagnosis or treatment recommendation.

## Guidance and safety

Treatment guidance is provided as an informational reference. Pesticide registrations, permitted crops, application rates, pre-harvest intervals, and local requirements can change. Check current official product labels and consult a qualified agricultural professional or local Krishi Vigyan Kendra before acting on treatment information. Do not apply chemicals based only on an automated prediction.
