import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_PATH
from src.inference import load_class_names

model = tf.keras.models.load_model(str(MODEL_PATH))
class_names = load_class_names()

def predict_pil(pil_img):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    tensor = np.expand_dims(arr, axis=0)
    probs = model.predict(tensor, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    return [(class_names[i], float(probs[i])) for i in top3]

images = {
    "images_apple2.jpg": Path("data/user_failure_cases/images_apple2.jpg"),
    "images_apple.jpg": Path("data/user_failure_cases/images_apple.jpg"),
    "images_corn health.jpg": Path("data/user_failure_cases/images_corn health.jpg"),
    "images_corn.jpg": Path("data/user_failure_cases/images_corn.jpg")
}

print("=" * 80)
print("TESTING PREPROCESSING VARIATIONS ON USER FAILURE CASES")
print("=" * 80)

for name, path in images.items():
    print(f"\nTarget Image: {name}")
    raw_pil = Image.open(path).convert("RGB")
    raw_pil = ImageOps.exif_transpose(raw_pil)
    w, h = raw_pil.size
    print(f"Original size: {w}x{h}")
    
    # 1. Direct Resize (Bilinear - Keras default)
    im_bilinear = raw_pil.resize((256, 256), Image.Resampling.BILINEAR)
    preds1 = predict_pil(im_bilinear)
    print("  1. Direct Bilinear Resize (Keras default):")
    for cls, prob in preds1:
        print(f"     {cls:<42} {prob*100:6.2f}%")
        
    # 2. Direct Resize (Area / Lanczos)
    im_lanczos = raw_pil.resize((256, 256), Image.Resampling.LANCZOS)
    preds2 = predict_pil(im_lanczos)
    print("  2. Direct Lanczos Resize:")
    for cls, prob in preds2:
        print(f"     {cls:<42} {prob*100:6.2f}%")
        
    # 3. Square Letterbox (Preserve aspect ratio + pad)
    scale = 256 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    im_scaled = raw_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    im_padded = Image.new("RGB", (256, 256), (128, 128, 128))
    im_padded.paste(im_scaled, ((256 - new_w) // 2, (256 - new_h) // 2))
    preds3 = predict_pil(im_padded)
    print("  3. Letterbox (Padded to 256x256 with neutral gray background):")
    for cls, prob in preds3:
        print(f"     {cls:<42} {prob*100:6.2f}%")

    # 4. Center Crop to Square then resize
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    im_crop = raw_pil.crop((left, top, left + min_dim, top + min_dim)).resize((256, 256), Image.Resampling.BILINEAR)
    preds4 = predict_pil(im_crop)
    print("  4. Center Crop to Square (focuses on leaf tissue):")
    for cls, prob in preds4:
        print(f"     {cls:<42} {prob*100:6.2f}%")
