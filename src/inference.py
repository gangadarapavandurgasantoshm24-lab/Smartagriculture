import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import keras
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

from src.config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH, NUM_CLASSES


@st.cache_resource(show_spinner=False)
def load_class_names() -> List[str]:
    """
    Load the 38 deterministic class names from models/class_names.json.
    Cached across all app sessions.
    """
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found at: {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes in {CLASS_NAMES_PATH}, but got {len(class_names)}"
        )

    return class_names


@st.cache_resource(show_spinner=False)
def load_model() -> keras.Model:
    """
    Load the trained application model (models/best_model.keras) once.
    Cached globally across sessions using st.cache_resource.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Application model not found at: {MODEL_PATH}")

    model = keras.models.load_model(str(MODEL_PATH))

    # Smoke call with dummy input to ensure Keras 3 layers are built
    dummy = tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
    _ = model(dummy)

    return model


def parse_class_label(class_name: str) -> Tuple[str, str, bool]:
    """
    Parse a PlantVillage formatted class string into (crop, condition, is_healthy).
    Example: 'Apple___Apple_scab' -> ('Apple', 'Apple scab', False)
             'Tomato___healthy'    -> ('Tomato', 'Healthy', True)
    """
    if "___" in class_name:
        crop_part, condition_part = class_name.split("___", 1)
    else:
        crop_part = "Unknown"
        condition_part = class_name

    crop = crop_part.replace("_", " ").strip()
    condition = condition_part.replace("_", " ").strip()
    is_healthy = "healthy" in condition.lower()

    return crop, condition, is_healthy



SUPPORTED_CROPS: List[str] = [
    "Apple",
    "Blueberry",
    "Cherry (including sour)",
    "Corn (maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper, bell",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]


def preprocess_image(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    center_crop: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess one input image strictly matching the training pipeline:
      1. Converts/reads image in RGB color format with EXIF orientation correction.
      2. If center_crop is True, takes a square center crop along the shortest edge.
      3. Resizes to (256, 256) spatial resolution.
      4. Converts to float32 NumPy array normalized to [0.0, 1.0] (divided by 255.0) exactly once.
      5. Expands dimensions to batch tensor: (1, 256, 256, 3).

    Returns:
      (batch_tensor, display_rgb_image_uint8)
    """
    rgb_arr: np.ndarray

    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            with Image.open(path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                rgb_arr = np.array(pil_img.convert("RGB"))

        elif isinstance(image_input, bytes):
            with Image.open(io.BytesIO(image_input)) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                rgb_arr = np.array(pil_img.convert("RGB"))

        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input)
            rgb_arr = np.array(pil_img.convert("RGB"))

        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                rgb_arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3:
                if arr.shape[2] == 4:
                    rgb_arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                elif arr.shape[2] == 3:
                    rgb_arr = arr
                else:
                    raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
            else:
                raise ValueError(f"Invalid image array dimension: {arr.ndim}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    except Exception as exc:
        raise ValueError(f"Image validation/decoding error: {exc}") from exc

    # Optional Center Crop for square leaf framing (removes letterbox / framing distortion)
    if center_crop:
        h, w = rgb_arr.shape[:2]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        rgb_arr = rgb_arr[top : top + min_dim, left : left + min_dim]

    # Ensure spatial dimension (256, 256)
    if rgb_arr.shape[:2] != IMAGE_SIZE:
        resized = cv2.resize(rgb_arr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    else:
        resized = rgb_arr

    # Normalize to [0.0, 1.0] float32 strictly once
    if issubclass(resized.dtype.type, np.floating) and resized.max() <= 1.0:
        # Already normalized float [0.0, 1.0]
        float_tensor = np.clip(resized.astype(np.float32), 0.0, 1.0)
        display_rgb = np.clip(float_tensor * 255.0, 0, 255).astype(np.uint8)
    else:
        # Standard uint8 or unnormalized values in range [0, 255]
        display_rgb = np.clip(resized, 0, 255).astype(np.uint8)
        float_tensor = display_rgb.astype(np.float32) / 255.0
        float_tensor = np.clip(float_tensor, 0.0, 1.0)

    # Expand batch dimension (1, 256, 256, 3)
    batch_tensor = np.expand_dims(float_tensor, axis=0)

    return batch_tensor, display_rgb


def predict_leaf(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    top_k: int = 3,
    confirmed_crop: Union[str, None] = None,
    center_crop: bool = False,
) -> Dict[str, Any]:
    """
    Run full end-to-end inference on a single plant leaf image with safety features:
      - Raw 38-way softmax distribution.
      - Detection of cross-crop species ambiguity (when model splits probability across crops).
      - Target crop disambiguation: when the user confirms their crop species,
        calculates within-crop relative condition probabilities to filter out background noise.
      - Leaf framing option: center square crop to eliminate border/aspect distortion.

    Returns dictionary containing full prediction metadata.
    """
    # 1. Load resources
    model = load_model()
    class_names = load_class_names()

    # 2. Preprocess
    batch_tensor, display_rgb = preprocess_image(image_input, center_crop=center_crop)

    # 3. Model prediction
    raw_probs = model.predict(batch_tensor, verbose=0)[0]
    probs = np.asarray(raw_probs, dtype=np.float32)

    # Safety check on probability distribution
    if len(probs) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} output probabilities, but model returned {len(probs)}"
        )

    # 4. Extract raw top candidate & rankings across all 38 classes
    raw_top_idx = int(np.argmax(probs))
    raw_top_class = class_names[raw_top_idx]
    raw_top_conf = float(probs[raw_top_idx])
    raw_crop, raw_condition, raw_healthy = parse_class_label(raw_top_class)

    # Raw Top K ranked list
    sorted_indices = np.argsort(probs)[::-1][:top_k]
    raw_top_predictions = []
    for rank, idx in enumerate(sorted_indices, start=1):
        c_name = class_names[idx]
        c_crop, c_cond, c_healthy = parse_class_label(c_name)
        raw_top_predictions.append(
            {
                "rank": rank,
                "class_index": int(idx),
                "class_name": c_name,
                "crop": c_crop,
                "condition": c_cond,
                "is_healthy": c_healthy,
                "confidence": float(probs[idx]),
                "confidence_percent": float(probs[idx] * 100.0),
            }
        )

    # 5. Cross-Crop Ambiguity Detection
    # Determine if top predictions span multiple distinct crop species
    conflicting_crops: List[str] = []
    for p in raw_top_predictions:
        if p["confidence_percent"] >= 2.0 and p["crop"] not in conflicting_crops:
            conflicting_crops.append(p["crop"])

    cross_crop_ambiguous = len(conflicting_crops) > 1 and (
        raw_top_predictions[0]["confidence"] < 0.85
        or (
            len(raw_top_predictions) > 1
            and raw_top_predictions[1]["confidence_percent"] >= 10.0
            and raw_top_predictions[1]["crop"] != raw_top_predictions[0]["crop"]
        )
    )

    # 6. Target Crop Disambiguation (if user confirms their crop species)
    active_crop = raw_crop
    active_condition = raw_condition
    active_healthy = raw_healthy
    active_class = raw_top_class
    active_idx = raw_top_idx
    active_confidence = raw_top_conf
    active_confidence_percent = raw_top_conf * 100.0
    active_top_predictions = raw_top_predictions

    within_crop_ranking = None
    crop_prob_sum = None

    valid_confirmed_crop = (
        confirmed_crop
        if (confirmed_crop and confirmed_crop in SUPPORTED_CROPS)
        else None
    )

    if valid_confirmed_crop:
        crop_indices = [
            i for i, c in enumerate(class_names) if parse_class_label(c)[0] == valid_confirmed_crop
        ]
        if crop_indices:
            crop_prob_sum = float(np.sum(probs[crop_indices]))
            if crop_prob_sum > 0:
                within_crop_raw = probs[crop_indices]
                within_crop_rel = within_crop_raw / crop_prob_sum
                sorted_crop_order = np.argsort(within_crop_rel)[::-1]

                within_crop_ranking = []
                for rank, local_idx in enumerate(sorted_crop_order, start=1):
                    global_idx = crop_indices[local_idx]
                    c_name = class_names[global_idx]
                    _, c_cond, c_healthy = parse_class_label(c_name)
                    within_crop_ranking.append(
                        {
                            "rank": rank,
                            "class_index": global_idx,
                            "class_name": c_name,
                            "crop": valid_confirmed_crop,
                            "condition": c_cond,
                            "is_healthy": c_healthy,
                            "confidence": float(within_crop_rel[local_idx]),
                            "confidence_percent": float(within_crop_rel[local_idx] * 100.0),
                            "raw_confidence": float(probs[global_idx]),
                            "raw_confidence_percent": float(probs[global_idx] * 100.0),
                        }
                    )

                top_within = within_crop_ranking[0]
                active_class = top_within["class_name"]
                active_idx = top_within["class_index"]
                active_crop = valid_confirmed_crop
                active_condition = top_within["condition"]
                active_healthy = top_within["is_healthy"]
                active_confidence = top_within["confidence"]
                active_confidence_percent = top_within["confidence_percent"]
                active_top_predictions = within_crop_ranking[:top_k]

    return {
        "predicted_class": active_class,
        "class_index": active_idx,
        "crop": active_crop,
        "condition": active_condition,
        "is_healthy": active_healthy,
        "confidence": active_confidence,
        "confidence_percent": active_confidence_percent,
        "top_predictions": active_top_predictions,
        "all_probabilities": probs,
        "input_tensor": batch_tensor,
        "display_image": display_rgb,
        "raw_predicted_class": raw_top_class,
        "raw_class_index": raw_top_idx,
        "raw_crop": raw_crop,
        "raw_condition": raw_condition,
        "raw_is_healthy": raw_healthy,
        "raw_confidence": raw_top_conf,
        "raw_confidence_percent": raw_top_conf * 100.0,
        "raw_top_predictions": raw_top_predictions,
        "cross_crop_ambiguous": cross_crop_ambiguous,
        "conflicting_crops": conflicting_crops,
        "confirmed_crop": valid_confirmed_crop,
        "within_crop_ranking": within_crop_ranking,
        "crop_probability_mass": crop_prob_sum,
        "center_crop_applied": center_crop,
    }

