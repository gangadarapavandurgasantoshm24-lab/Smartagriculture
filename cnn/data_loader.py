import os
from pathlib import Path

# Suppress excessive TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
DATASET_ROOT = Path(
    "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
)
TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
COLOR_MODE = "rgb"
BATCH_SIZE = 32
CHANNELS = 3

EXPECTED_CLASSES_COUNT = 38
EXPECTED_TRAIN_COUNT = 70295
EXPECTED_VALID_COUNT = 17572


def detect_classes(train_dir: Path) -> list[str]:
    """
    Automatically detect all class directories in alphabetical order.
    Alphabetical sorting guarantees a deterministic, reproducible index mapping.
    """
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return class_names


def build_class_mapping(class_names: list[str]) -> dict[str, int]:
    """
    Create a deterministic zero-based label mapping (0 to N-1).
    """
    return {name: idx for idx, name in enumerate(class_names)}


def load_dataset_splits(
    train_dir: Path = TRAIN_DIR,
    valid_dir: Path = VALID_DIR,
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
):
    """
    Load training and validation datasets using tf.keras.utils.image_dataset_from_directory
    with normalized pixel values (0.0 to 1.0) and deterministic class indexing.
    """
    # 1. Detect class names
    class_names = detect_classes(train_dir)
    class_to_idx = build_class_mapping(class_names)

    # 2. Create raw training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=42,
    )

    # 3. Create raw validation dataset (shuffle=False for deterministic evaluation)
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    train_image_count = len(train_ds.file_paths)
    valid_image_count = len(valid_ds.file_paths)

    # 4. Normalize pixel values from [0, 255] to [0.0, 1.0]
    rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

    train_ds = train_ds.map(
        lambda x, y: (rescaling_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    valid_ds = valid_ds.map(
        lambda x, y: (rescaling_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return (
        train_ds,
        valid_ds,
        class_names,
        class_to_idx,
        train_image_count,
        valid_image_count,
    )


def main():
    print("=" * 70)
    print("SMART AGRICULTURE: CNN DATA LOADER & LABEL MAPPING PIPELINE")
    print("=" * 70)

    # 1. Load datasets and metadata
    (
        train_ds,
        valid_ds,
        class_names,
        class_to_idx,
        train_count,
        valid_count,
    ) = load_dataset_splits()

    num_classes = len(class_names)

    # 2. Print dataset overview
    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Number of classes: {num_classes}")
    print(f"Number of training images: {train_count:,}")
    print(f"Number of validation images: {valid_count:,}")
    print(f"Total dataset images: {train_count + valid_count:,}")
    print(f"Target image shape: ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, {CHANNELS})")
    print(f"Batch size: {BATCH_SIZE}")

    print("\n--- Detected Class Names (Total: 38) ---")
    for idx, name in enumerate(class_names):
        print(f"  [{idx:02d}] {name}")

    print("\n--- Class-to-Index Mapping (Zero-Based: 0 to 37) ---")
    for name, idx in class_to_idx.items():
        print(f"  '{name}': {idx}")

    # 3. Load exactly one batch and inspect
    print("\n" + "=" * 70)
    print("SINGLE BATCH INSPECTION")
    print("=" * 70)

    batch_images = None
    batch_labels = None
    for images, labels in train_ds.take(1):
        batch_images = images
        batch_labels = labels
        break

    if batch_images is None:
        raise RuntimeError("Failed to load a batch from the training dataset.")

    batch_image_shape = tuple(batch_images.shape)
    batch_label_shape = tuple(batch_labels.shape)
    min_pixel = float(tf.reduce_min(batch_images).numpy())
    max_pixel = float(tf.reduce_max(batch_images).numpy())
    sample_labels = batch_labels.numpy()[:10].tolist()

    print(f"Batch image shape: {batch_image_shape}")
    print(f"Batch label shape: {batch_label_shape}")
    print(f"Batch image dtype: {batch_images.dtype.name}")
    print(f"Minimum pixel value: {min_pixel:.4f}")
    print(f"Maximum pixel value: {max_pixel:.4f}")
    print(f"First 10 labels in batch: {sample_labels}")

    # 4. Verification checks
    classes_pass = num_classes == EXPECTED_CLASSES_COUNT
    mapping_pass = (
        len(class_to_idx) == EXPECTED_CLASSES_COUNT
        and min(class_to_idx.values()) == 0
        and max(class_to_idx.values()) == EXPECTED_CLASSES_COUNT - 1
    )
    train_pass = train_count == EXPECTED_TRAIN_COUNT
    valid_pass = valid_count == EXPECTED_VALID_COUNT
    size_pass = batch_image_shape[1:3] == IMAGE_SIZE
    rgb_pass = batch_image_shape[3] == CHANNELS
    norm_pass = 0.0 <= min_pixel <= 1.0 and 0.0 <= max_pixel <= 1.0 and max_pixel > 0.0
    batch_pass = (
        batch_image_shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
        and batch_label_shape == (BATCH_SIZE,)
    )

    all_passed = (
        classes_pass
        and mapping_pass
        and train_pass
        and valid_pass
        and size_pass
        and rgb_pass
        and norm_pass
        and batch_pass
    )

    # 5. Verification Report
    print("\n" + "=" * 70)
    print("CNN DATA LOADING VERIFICATION")
    print("=" * 70)
    print(f"Classes:           {'PASS' if classes_pass else 'FAIL'}")
    print(f"Class mapping:     {'PASS' if mapping_pass else 'FAIL'}")
    print(f"Training images:   {'PASS' if train_pass else 'FAIL'}")
    print(f"Validation images: {'PASS' if valid_pass else 'FAIL'}")
    print(f"Image size:        {'PASS' if size_pass else 'FAIL'}")
    print(f"RGB format:        {'PASS' if rgb_pass else 'FAIL'}")
    print(f"Normalization:     {'PASS' if norm_pass else 'FAIL'}")
    print(f"Batch loading:     {'PASS' if batch_pass else 'FAIL'}")
    print("-" * 70)
    print(f"Overall:           {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)

    if not all_passed:
        print("\n[WARNING] Discrepancies detected during data loading verification:")
        if not classes_pass:
            print(f"  - Classes: Expected {EXPECTED_CLASSES_COUNT}, got {num_classes}")
        if not mapping_pass:
            print("  - Class mapping is invalid or incomplete.")
        if not train_pass:
            print(f"  - Train images: Expected {EXPECTED_TRAIN_COUNT}, got {train_count}")
        if not valid_pass:
            print(f"  - Valid images: Expected {EXPECTED_VALID_COUNT}, got {valid_count}")
        if not size_pass:
            print(f"  - Image size: Expected {IMAGE_SIZE}, got {batch_image_shape[1:3]}")
        if not rgb_pass:
            print(f"  - Color channels: Expected {CHANNELS}, got {batch_image_shape[3]}")
        if not norm_pass:
            print(f"  - Normalization: Range [{min_pixel}, {max_pixel}] outside expected [0.0, 1.0]")
        if not batch_pass:
            print(f"  - Batch shape: Got {batch_image_shape}, labels {batch_label_shape}")


if __name__ == "__main__":
    main()
