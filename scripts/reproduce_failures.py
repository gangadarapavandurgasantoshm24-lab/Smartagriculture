import hashlib
import sys
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES_PATH, MODEL_PATH
from src.inference import load_class_names, predict_leaf, preprocess_image

def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()

print("=" * 80)
print("TASK 1: REPRODUCE BOTH REPORTED FAILURES")
print("=" * 80)

# Checkpoint verification
app_model_path = MODEL_PATH
cnn_model_path = Path("cnn/best_model.keras")

app_hash = get_file_hash(app_model_path)
cnn_hash = get_file_hash(cnn_model_path) if cnn_model_path.is_file() else "NOT FOUND"

print(f"\n1. App Configured Checkpoint (src/config.py):")
print(f"   Path:   {app_model_path}")
print(f"   Exists: {app_model_path.is_file()}")
print(f"   Size:   {app_model_path.stat().st_size:,} bytes")
print(f"   SHA256: {app_hash}")

print(f"\n2. Alternative Checkpoint (cnn/best_model.keras):")
print(f"   Path:   {cnn_model_path}")
print(f"   Exists: {cnn_model_path.is_file()}")
if cnn_model_path.is_file():
    print(f"   Size:   {cnn_model_path.stat().st_size:,} bytes")
    print(f"   SHA256: {cnn_hash}")
    print(f"   Identical to app model? {app_hash == cnn_hash}")

class_names = load_class_names()
print(f"\n3. Total class names loaded: {len(class_names)}")

# Test images in data/user_failure_cases/
failure_dir = Path("data/user_failure_cases")
test_images = sorted(list(failure_dir.glob("*.jpg")) + list(failure_dir.glob("*.png")))

print(f"\n4. Testing {len(test_images)} located failure case images:")

# Load both models for comparison
model_app = tf.keras.models.load_model(str(app_model_path))
model_cnn = tf.keras.models.load_model(str(cnn_model_path)) if cnn_model_path.is_file() else None

for img_file in test_images:
    print("\n" + "-" * 75)
    print(f"IMAGE: {img_file.name} ({img_file.stat().st_size:,} bytes)")
    
    # Preprocess using app pipeline
    batch_tensor, display_rgb = preprocess_image(img_file)
    print(f"  Preprocessed tensor shape: {batch_tensor.shape}, min: {batch_tensor.min():.3f}, max: {batch_tensor.max():.3f}")
    
    # 1. Predictions using Application Checkpoint (models/best_model.keras)
    probs_app = model_app.predict(batch_tensor, verbose=0)[0]
    top5_app = np.argsort(probs_app)[::-1][:5]
    print(f"\n  [App Checkpoint: models/best_model.keras] Top-5 Predictions:")
    for rank, idx in enumerate(top5_app, 1):
        print(f"    #{rank}: {class_names[idx]:45} {probs_app[idx]*100:6.2f}% (prob: {probs_app[idx]:.6f})")
        
    # 2. Predictions using CNN Checkpoint (cnn/best_model.keras)
    if model_cnn is not None:
        probs_cnn = model_cnn.predict(batch_tensor, verbose=0)[0]
        top5_cnn = np.argsort(probs_cnn)[::-1][:5]
        print(f"\n  [CNN Checkpoint: cnn/best_model.keras] Top-5 Predictions:")
        for rank, idx in enumerate(top5_cnn, 1):
            print(f"    #{rank}: {class_names[idx]:45} {probs_cnn[idx]*100:6.2f}% (prob: {probs_cnn[idx]:.6f})")

print("\n" + "=" * 80)
print("REPRODUCTION ANALYSIS COMPLETE")
print("=" * 80)
