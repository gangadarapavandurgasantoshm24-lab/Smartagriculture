from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

DATA_DIR = ROOT_DIR / "data"
CLASS_ANALYSIS_PATH = DATA_DIR / "class_analysis.csv"
DISEASE_GUIDANCE_PATH = DATA_DIR / "disease_guidance.json"
TEST_IMAGES_DIR = DATA_DIR / "raw" / "test" / "test"

REPORTS_DIR = ROOT_DIR / "reports"
CNN_DIR = ROOT_DIR / "cnn"
TRAINING_HISTORY_PATH = CNN_DIR / "training_history.csv"
CLASSIFICATION_REPORT_TXT_PATH = CNN_DIR / "classification_report.txt"
CLASSIFICATION_REPORT_CSV_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG_PATH = REPORTS_DIR / "confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PNG_PATH = REPORTS_DIR / "normalized_confusion_matrix.png"
MISCLASSIFIED_GALLERY_PNG_PATH = REPORTS_DIR / "misclassified_gallery.png"
MISCLASSIFIED_SAMPLES_CSV_PATH = REPORTS_DIR / "misclassified_samples.csv"
ERROR_ANALYSIS_MD_PATH = REPORTS_DIR / "error_analysis.md"

# Persistent Storage
SCAN_UPLOADS_DIR = DATA_DIR / "scan_uploads"
DB_PATH = DATA_DIR / "scan_history.db"

# Model Architecture Specifics
INPUT_SHAPE = (256, 256, 3)
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 38
TARGET_CONV_LAYER = "conv2d_3"
MODEL_VALIDATION_ACCURACY = 92.61  # Reported best validation accuracy for application model

# Design System Color Palette
COLOR_FOREST_GREEN = "#173F35"
COLOR_SECONDARY_GREEN = "#2F6B4F"
COLOR_ACCENT_GREEN = "#3D8B68"
COLOR_LIGHT_SAGE = "#E8F2E8"
COLOR_WARM_OFFWHITE = "#F7F8F4"
COLOR_WHITE = "#FFFFFF"
COLOR_BORDER = "#D3E2D3"
COLOR_TEXT_PRIMARY = "#26332D"
COLOR_TEXT_MUTED = "#556E63"
COLOR_DANGER = "#D9383A"
COLOR_DANGER_BG = "#FDF2F2"
COLOR_SUCCESS = "#2E8B57"
COLOR_SUCCESS_BG = "#EDF7F2"
COLOR_WARNING = "#D97706"
COLOR_WARNING_BG = "#FFFBEB"
COLOR_INFO = "#2563EB"
COLOR_INFO_BG = "#EFF6FF"
