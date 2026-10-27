"""
cnn/predict_image.py
--------------------
Predict plant disease or healthy status for a single leaf image using
the trained Smart Agriculture Convolutional Neural Network (CNN).

Usage Example:
    python cnn/predict_image.py "data/raw/test/example.jpg"
"""

import argparse
import os
import sys
from pathlib import Path

# Suppress excessive TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure workspace root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Validate required dependencies with beginner-friendly error messages
try:
    import numpy as np
    import tensorflow as tf
except ImportError as err:
    print(
        f"\n[ERROR] Missing required library: {err}\n"
        "Please ensure numpy and tensorflow are installed in your active Python environment."
    )
    sys.exit(1)

# Import existing verified dataset configuration and class detection
from cnn.data_loader import IMAGE_SIZE, TRAIN_DIR, detect_classes

# Model path
MODEL_PATH = Path("cnn/best_model.keras")
EXPECTED_CLASSES_COUNT = 38


def preprocess_single_image(image_path: Path, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Load and preprocess one image for CNN inference matching the training pipeline:
      1. Loads image in RGB format and resizes to (256, 256) without modifying the original file.
      2. Converts image to float32 NumPy array.
      3. Normalizes pixel values from [0, 255] to [0.0, 1.0].
      4. Adds batch dimension yielding shape (1, 256, 256, 3).
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: '{image_path}'")

    try:
        # Load image via TensorFlow/Keras utilities in RGB mode resized to 256x256
        pil_image = tf.keras.utils.load_img(
            str(image_path),
            color_mode="rgb",
            target_size=target_size,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to decode image file '{image_path}'. Ensure it is a valid image format (.jpg, .jpeg, .png). "
            f"Details: {exc}"
        ) from exc

    # Convert PIL Image to float32 NumPy array
    img_array = tf.keras.utils.img_to_array(pil_image, dtype="float32")

    # Normalize pixel values to [0.0, 1.0] exactly matching training pipeline
    normalized_array = img_array / 255.0

    # Expand dims to create batch of size 1: (1, 256, 256, 3)
    batch_array = np.expand_dims(normalized_array, axis=0)
    return batch_array


def predict(image_path_str: str) -> None:
    """
    Load the trained model and execute prediction on a single leaf image.
    """
    image_path = Path(image_path_str)

    print("=" * 70)
    print("SMART AGRICULTURE: SINGLE IMAGE LEAF PREDICTION")
    print("=" * 70)

    # 1. Validate and load trained model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model file '{MODEL_PATH}' was not found. "
            "Please train the model first by running: python cnn/train.py"
        )

    print(f"\n[1/3] Loading trained model from '{MODEL_PATH}'...")
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {exc}") from exc

    # 2. Retrieve verified class names using the data loader's public function
    print("[2/3] Retrieving dataset class mapping...")
    class_names = detect_classes(TRAIN_DIR)
    num_classes = len(class_names)

    # Verify model output alignment with 38 dataset classes
    model_output_dim = model.output_shape[-1]
    if model_output_dim != num_classes or model_output_dim != EXPECTED_CLASSES_COUNT:
        raise ValueError(
            f"Class count mismatch: Model outputs {model_output_dim} classes, "
            f"but detected dataset has {num_classes} classes."
        )

    # 3. Load and preprocess the target image
    print(f"[3/3] Preprocessing image '{image_path}'...")
    input_tensor = preprocess_single_image(image_path, target_size=IMAGE_SIZE)

    # 4. Run model inference (verbose=0 for clean console output)
    probabilities = model.predict(input_tensor, verbose=0)[0]

    # Verify probability distribution output length
    if len(probabilities) != EXPECTED_CLASSES_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES_COUNT} class prediction scores, but got {len(probabilities)}."
        )

    # 5. Extract top prediction and top 3 candidates
    top_index = int(np.argmax(probabilities))
    top_class_name = class_names[top_index]
    top_confidence = float(probabilities[top_index]) * 100.0

    # Sort indices in descending order of confidence
    sorted_indices = np.argsort(probabilities)[::-1][:3]

    # 6. Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Image Path:            {image_path.resolve()}")
    print(f"Predicted Class:       {top_class_name}")
    print(f"Confidence:            {top_confidence:.2f}%")

    print("\n--- Top 3 Predicted Candidates ---")
    print(f"{'Rank':<6} {'Confidence':<12} {'Class Name'}")
    print("-" * 70)
    for rank, idx in enumerate(sorted_indices, start=1):
        class_name = class_names[idx]
        conf_val = float(probabilities[idx]) * 100.0
        print(f"#{rank:<5} {conf_val:>6.2f}%      {class_name}")

    print("\n" + "=" * 70)
    print("SCOPE & USAGE NOTE")
    print("=" * 70)
    print(
        "Note: This model is trained exclusively to classify among 38 specific crop\n"
        "conditions (14 crops). It does not provide medical or chemical treatment\n"
        "recommendations, nor can it identify diseases outside its 38 trained classes."
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Predict plant disease or healthy status from a single leaf image using the trained CNN model.",
        epilog="Example: python cnn/predict_image.py \"data/raw/test/test/AppleScab1.JPG\"",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input leaf image (.jpg, .jpeg, .png).",
    )
    args = parser.parse_args()

    try:
        predict(args.image_path)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"\n[ERROR] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
