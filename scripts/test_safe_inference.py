import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_PATH
from src.inference import load_class_names, parse_class_label, preprocess_image, predict_leaf

print("=" * 80)
print("TESTING SAFE INFERENCE & CROP CONFIRMATION LOGIC")
print("=" * 80)

def predict_with_crop_context(image_path, confirmed_crop=None, center_crop=False):
    # Preprocess
    from PIL import Image, ImageOps
    raw_pil = Image.open(image_path).convert("RGB")
    raw_pil = ImageOps.exif_transpose(raw_pil)
    
    if center_crop:
        w, h = raw_pil.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        raw_pil = raw_pil.crop((left, top, left + min_dim, top + min_dim))
        
    res = predict_leaf(raw_pil, top_k=38)
    
    # Check cross-crop ambiguity
    top3 = res["top_predictions"][:3]
    top_crops = [p["crop"] for p in top3 if p["confidence_percent"] > 5.0]
    cross_crop_ambiguous = len(set(top_crops)) > 1
    
    print(f"\nImage: {Path(image_path).name} (Center crop: {center_crop}, Confirmed Crop: {confirmed_crop})")
    print(f"  Raw Top-1: {res['predicted_class']} ({res['confidence_percent']:.2f}%)")
    print(f"  Top-3 Candidates:")
    for p in top3:
        print(f"    #{p['rank']}: {p['class_name']} ({p['confidence_percent']:.2f}%) [Crop: {p['crop']}]")
    print(f"  Cross-Crop Ambiguity Detected: {cross_crop_ambiguous}")
    
    if confirmed_crop and confirmed_crop != "Auto-Detect":
        # Filter classes belonging to confirmed crop
        crop_preds = [p for p in res["top_predictions"] if p["crop"].lower() == confirmed_crop.lower() or confirmed_crop.lower() in p["crop"].lower()]
        total_crop_prob = sum(p["confidence"] for p in crop_preds)
        print(f"\n  [Confirmed Crop Filter: {confirmed_crop}]")
        print(f"  Total raw probability in {confirmed_crop}: {total_crop_prob*100:.4f}%")
        if total_crop_prob > 0:
            for p in crop_preds:
                norm_conf = (p["confidence"] / total_crop_prob) * 100.0
                print(f"    -> {p['condition']:<30} Raw: {p['confidence_percent']:6.2f}% | Within-Crop Relative: {norm_conf:6.2f}%")
        else:
            print("    -> No non-zero probabilities for this crop.")

# 1. Test Apple2 without and with crop confirmation
predict_with_crop_context("data/user_failure_cases/images_apple2.jpg", confirmed_crop="Auto-Detect")
predict_with_crop_context("data/user_failure_cases/images_apple2.jpg", confirmed_crop="Apple")

# 2. Test Corn health without and with center crop
predict_with_crop_context("data/user_failure_cases/images_corn health.jpg", center_crop=False)
predict_with_crop_context("data/user_failure_cases/images_corn health.jpg", center_crop=True)
