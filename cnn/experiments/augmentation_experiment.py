"""Single controlled online-augmentation experiment for the existing CNN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cnn.data_loader import (  # noqa: E402
    BATCH_SIZE, EXPECTED_TRAIN_COUNT, EXPECTED_VALID_COUNT, IMAGE_SIZE,
    TRAIN_DIR, VALID_DIR, detect_classes, load_dataset_splits,
)
from cnn.model import build_cnn_model  # noqa: E402

BASELINE_MODEL = ROOT / "cnn" / "best_model.keras"
BASELINE_REPORT = ROOT / "reports" / "classification_report.csv"
EXPERIMENT_DIR = ROOT / "cnn" / "experiments" / "augmentation_v1"
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
HISTORY_PATH = EXPERIMENT_DIR / "training_history.csv"
CHECKPOINT_PATH = EXPERIMENT_DIR / "augmentation_model.keras"
CLASS_REPORT_PATH = EXPERIMENT_DIR / "classification_report.csv"
CONFUSION_PATH = EXPERIMENT_DIR / "confusion_matrix.csv"
COMPARISON_PATH = REPORTS_DIR / "experiment_comparison.csv"
SUMMARY_PATH = REPORTS_DIR / "augmentation_experiment.md"
SEED = 42
WEAK_CLASSES = [
    "Peach___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Early_blight",
    "Tomato___Target_Spot",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_augmentation() -> tf.keras.Sequential:
    """Mild leaf-preserving transforms; input pixel range is [0, 1]."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=SEED + 1),
        tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect", seed=SEED + 2),
        tf.keras.layers.RandomZoom(0.08, fill_mode="reflect", seed=SEED + 3),
        tf.keras.layers.RandomBrightness(0.08, value_range=(0.0, 1.0), seed=SEED + 4),
    ], name="online_leaf_augmentation")


def configuration() -> dict:
    return {
        "experiment": "augmentation_v1",
        "intervention": "online training-only image augmentation",
        "baseline_checkpoint": str(BASELINE_MODEL.relative_to(ROOT)),
        "experimental_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "train_directory": str(TRAIN_DIR),
        "validation_directory": str(VALID_DIR),
        "class_order": detect_classes(TRAIN_DIR),
        "input_shape": [256, 256, 3],
        "preprocessing": "RGB resize to 256x256; float32 divided by 255",
        "augmentation": {
            "horizontal_and_vertical_flip": True,
            "rotation_factor": 0.04,
            "translation_height_factor": 0.04,
            "translation_width_factor": 0.04,
            "zoom_factor": 0.08,
            "brightness_factor": 0.08,
            "fill_mode": "reflect",
            "application": "training batches only; validation is unchanged",
        },
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": 20,
        "optimizer": "Adam (same Keras default configuration as baseline)",
        "loss": "sparse_categorical_crossentropy",
        "model_architecture": "cnn.model.build_cnn_model(num_classes=38), unchanged",
        "early_stopping": {"monitor": "val_loss", "patience": 3, "restore_best_weights": True},
        "checkpoint_selection": {"monitor": "val_accuracy", "mode": "max", "save_best_only": True},
        "learning_rate_handling": "Adam default; no scheduler in baseline, so none added",
        "data_counts_expected": {"train": EXPECTED_TRAIN_COUNT, "validation": EXPECTED_VALID_COUNT},
        "baseline_sha256": sha256(BASELINE_MODEL),
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "devices": [d.device_type for d in tf.config.list_physical_devices()],
            "platform": platform.platform(),
        },
    }


def read_baseline() -> pd.DataFrame:
    if not BASELINE_REPORT.is_file():
        raise FileNotFoundError(f"Verified Phase 1 report is required: {BASELINE_REPORT}")
    frame = pd.read_csv(BASELINE_REPORT)
    if "class" not in frame or "f1_score" not in frame:
        raise ValueError(f"Unexpected Phase 1 report format: {BASELINE_REPORT}")
    return frame


def write_comparison(baseline: pd.DataFrame, experimental: pd.DataFrame | None = None) -> None:
    summary = baseline.set_index("class")
    records = []
    for split in ("baseline", "augmentation_v1"):
        data = summary if split == "baseline" else (experimental.set_index("class") if experimental is not None else None)
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            if split == "baseline":
                if metric == "accuracy":
                    value = float(summary.loc["weighted avg", "recall"])
                else:
                    value = float(summary.loc["macro avg" if metric != "f1_score" else "macro avg", metric])
            elif data is None:
                value = np.nan
            elif metric == "accuracy":
                value = float(data.loc["weighted avg", "recall"])
            else:
                value = float(data.loc["macro avg", metric])
            records.append({"model": split, "metric": metric, "class": "overall", "value": value})
        for class_name in WEAK_CLASSES:
            baseline_recall = float(summary.loc[class_name, "recall"])
            exp_recall = np.nan if experimental is None else float(experimental.set_index("class").loc[class_name, "recall"])
            records.extend([
                {"model": "baseline", "metric": "recall", "class": class_name, "value": baseline_recall},
                {"model": "augmentation_v1", "metric": "recall", "class": class_name, "value": exp_recall},
            ])
    if experimental is not None:
        wide = pd.DataFrame(records).pivot_table(index=["metric", "class"], columns="model", values="value", aggfunc="first")
        for (metric, class_name), row in wide.iterrows():
            if pd.notna(row.get("baseline")) and pd.notna(row.get("augmentation_v1")):
                records.append({"model": "delta_experimental_minus_baseline", "metric": metric,
                                "class": class_name, "value": float(row["augmentation_v1"] - row["baseline"])})
    pd.DataFrame(records).to_csv(COMPARISON_PATH, index=False)


def write_summary(status: str, extra: str = "") -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    baseline = read_baseline().set_index("class")
    weak = "\n".join(f"- {name}: {baseline.loc[name, 'recall']:.4%}" for name in WEAK_CLASSES)
    text = (
        "# Controlled online augmentation experiment\n\n"
        f"Status: **{status}**\n\n"
        "## Baseline pipeline findings\n\n"
        "The training tree already contains offline augmented image files (for example, filenames with `_90deg`, `_270deg`, and `_new30degFlipLR`). "
        "The loader itself has no online augmentation. It resizes RGB images to 256×256, divides float32 pixels by 255, shuffles training data with seed 42, and keeps validation unshuffled and deterministic.\n\n"
        "The CNN is unchanged: three Conv2D/MaxPooling blocks (32, 64, 128 filters), Flatten, Dense(128), Dropout(0.5), and 38-way softmax. "
        "It uses Adam defaults and sparse categorical cross-entropy. Baseline training uses batch size 32, a 20 epoch maximum, early stopping on `val_loss` (patience 3, restore best weights), and best-only checkpointing on `val_accuracy`. "
        "Only the training shuffle has an explicit baseline seed; model initialization is not seeded. The existing history contains 10 epochs, with best recorded `val_accuracy` at epoch 7.\n\n"
        "## Single intervention\n\n"
        "Training batches receive mild online horizontal/vertical flips, ±0.04 random rotation, ±0.04 translations, 0.08 zoom, and 0.08 brightness variation. "
        "Validation receives only the baseline resize and 1/255 normalization. Architecture, loss, optimizer, batch size, class mapping, data split, epoch cap, and callbacks are held constant. "
        f"Seed: {cfg['seed']}. Full settings are in `{CONFIG_PATH.relative_to(ROOT)}`.\n\n"
        "## Baseline reference\n\n"
        "- Accuracy: 93.2165%\n- Macro F1: 0.931868\n- Weighted F1: 0.931973\n\n"
        f"Weak class baseline recalls:\n{weak}\n\n"
        f"## Run status and comparison\n\n{extra}\n\n"
        "No experimental metric is reported until a complete training run and evaluation on all 17,572 validation images finish. "
        "See `reports/experiment_comparison.csv` for the baseline values and experimental fields (blank until evaluation).\n\n"
        f"## Reproduction\n\nTrain and evaluate: `python cnn/experiments/augmentation_experiment.py train`\n\n"
        f"Baseline SHA-256 recorded before the experiment: `{cfg['baseline_sha256']}`.\n"
    )
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def verify() -> None:
    if not BASELINE_MODEL.is_file():
        raise FileNotFoundError(BASELINE_MODEL)
    before_hash = sha256(BASELINE_MODEL)
    cfg = configuration()
    classes = cfg["class_order"]
    train_classes = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    valid_classes = sorted(p.name for p in VALID_DIR.iterdir() if p.is_dir())
    if len(classes) != 38 or classes != train_classes or classes != valid_classes:
        raise ValueError("Class directory mapping differs from expected 38-class train/validation mapping")
    model = tf.keras.models.load_model(BASELINE_MODEL, compile=False)
    if tuple(model.input_shape[1:]) != (256, 256, 3) or model.output_shape[-1] != 38:
        raise ValueError(f"Unexpected baseline model shapes: {model.input_shape}, {model.output_shape}")
    (train_ds, valid_ds, loaded_classes, _, train_n, valid_n) = load_dataset_splits(batch_size=BATCH_SIZE)
    if loaded_classes != classes or train_n != EXPECTED_TRAIN_COUNT or valid_n != EXPECTED_VALID_COUNT:
        raise ValueError(f"Split or class mismatch: classes={len(loaded_classes)}, train={train_n}, valid={valid_n}")
    augment = build_augmentation()
    x_train, y_train = next(iter(train_ds.take(1)))
    aug_batch = augment(x_train, training=True)
    x_valid, _ = next(iter(valid_ds.take(1)))
    if tuple(aug_batch.shape) != tuple(x_train.shape) or not bool(tf.reduce_all(tf.math.is_finite(aug_batch)).numpy()):
        raise ValueError("Augmentation smoke check failed shape/finite-value checks")
    if float(tf.reduce_min(aug_batch)) < -0.25 or float(tf.reduce_max(aug_batch)) > 1.25:
        raise ValueError("Augmentation smoke check produced an unexpected pixel range")
    if tuple(x_valid.shape[1:]) != (256, 256, 3) or float(tf.reduce_min(x_valid)) < 0 or float(tf.reduce_max(x_valid)) > 1:
        raise ValueError("Validation preprocessing smoke check failed")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_comparison(read_baseline())
    after_hash = sha256(BASELINE_MODEL)
    if before_hash != after_hash:
        raise RuntimeError("Baseline checkpoint changed during verification")
    write_summary("Implemented and smoke-verified; training not run in this verification step.",
                  "A full run is compute intensive on this native Windows CPU-only TensorFlow environment. The augmentation and deterministic validation pipelines were smoke-tested on a batch; the full training run remains available via the command below.")
    print(f"Verification passed: input={model.input_shape}, output={model.output_shape}, classes={len(classes)}, train={train_n}, valid={valid_n}")
    print(f"Augmented batch: {tuple(aug_batch.shape)}, range=[{float(tf.reduce_min(aug_batch)):.4f}, {float(tf.reduce_max(aug_batch)):.4f}]")
    print(f"Baseline SHA-256 unchanged: {after_hash}")
    print(f"Configuration: {CONFIG_PATH.relative_to(ROOT)}")


def evaluate_checkpoint(baseline: pd.DataFrame, training: dict) -> pd.DataFrame:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Experimental checkpoint not found: {CHECKPOINT_PATH}")
    classes = detect_classes(TRAIN_DIR)
    valid_classes = sorted(p.name for p in VALID_DIR.iterdir() if p.is_dir())
    if classes != valid_classes:
        raise ValueError("Validation class order does not match training class order")
    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)
    if tuple(model.input_shape[1:]) != (256, 256, 3) or model.output_shape[-1] != len(classes):
        raise ValueError(f"Experimental model shape mismatch: {model.input_shape}, {model.output_shape}")
    raw = tf.keras.utils.image_dataset_from_directory(VALID_DIR, labels="inferred", label_mode="int",
        class_names=classes, color_mode="rgb", batch_size=32, image_size=IMAGE_SIZE, shuffle=False)
    valid = raw.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    y_true = np.concatenate([y.numpy() for _, y in valid])
    probs = model.predict(valid, verbose=0)
    y_pred = probs.argmax(axis=1)
    if len(y_true) != EXPECTED_VALID_COUNT or len(y_pred) != EXPECTED_VALID_COUNT:
        raise ValueError(f"Expected {EXPECTED_VALID_COUNT} validation images, got {len(y_true)} and {len(y_pred)}")
    metrics = classification_report(y_true, y_pred, labels=np.arange(len(classes)), target_names=classes, output_dict=True, zero_division=0)
    exp_rows = [{"class": c, "precision": metrics[c]["precision"], "recall": metrics[c]["recall"],
                 "f1_score": metrics[c]["f1-score"], "support": int(metrics[c]["support"])} for c in classes]
    exp_rows.extend({"class": key, "precision": metrics[key]["precision"], "recall": metrics[key]["recall"],
                     "f1_score": metrics[key]["f1-score"], "support": EXPECTED_VALID_COUNT} for key in ("macro avg", "weighted avg"))
    experimental = pd.DataFrame(exp_rows)
    experimental.to_csv(CLASS_REPORT_PATH, index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(len(classes))), index=classes, columns=classes).to_csv(CONFUSION_PATH, index_label="actual_class")
    write_comparison(baseline, experimental)
    model_summary = {"accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": metrics["macro avg"]["precision"], "macro_recall": metrics["macro avg"]["recall"],
        "macro_f1": metrics["macro avg"]["f1-score"], "weighted_precision": metrics["weighted avg"]["precision"],
        "weighted_recall": metrics["weighted avg"]["recall"], "weighted_f1": metrics["weighted avg"]["f1-score"],
        "epochs_completed": training["epochs_completed"], "best_epoch": training["best_epoch"],
        "training_duration_seconds": training["training_duration_seconds"]}
    (EXPERIMENT_DIR / "evaluation_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = read_baseline().set_index("class")
    regressed = []
    weak_comparison = []
    for row in exp_rows:
        name = row["class"]
        if name in base.index and row["class"] not in ("macro avg", "weighted avg"):
            delta = row["recall"] - float(base.loc[name, "recall"])
            if delta < 0:
                regressed.append(f"- {name}: recall {delta:+.2%}")
    exp_index = experimental.set_index("class")
    for name in WEAK_CLASSES:
        before = float(base.loc[name, "recall"])
        after = float(exp_index.loc[name, "recall"])
        weak_comparison.append(f"- {name}: {before:.4%} → {after:.4%} ({after-before:+.2%})")
    accuracy_delta = model_summary["accuracy"] - float(base.loc["weighted avg", "recall"])
    macro_f1_delta = model_summary["macro_f1"] - float(base.loc["macro avg", "f1_score"])
    weighted_f1_delta = model_summary["weighted_f1"] - float(base.loc["weighted avg", "f1_score"])
    details = (f"Training completed: {training['epochs_completed']} epochs; best epoch {training['best_epoch']} by validation accuracy; "
        f"duration {training['training_duration_seconds']:.0f} seconds.\n\n"
        f"Experimental accuracy {model_summary['accuracy']:.6f} ({accuracy_delta:+.6f}), macro F1 {model_summary['macro_f1']:.6f} ({macro_f1_delta:+.6f}), "
        f"weighted F1 {model_summary['weighted_f1']:.6f} ({weighted_f1_delta:+.6f}); deltas are experimental minus baseline.\n\n"
        f"Previously weak class recall, before → after:\n" + "\n".join(weak_comparison) +
        "\n\nAll classes with lower recall:\n" + ("\n".join(regressed) if regressed else "- None") +
        "\n\nExperimental per-class metrics and confusion matrix are saved in `cnn/experiments/augmentation_v1/`. The comparison CSV includes deltas.")
    write_summary("Training and full validation evaluation completed.", details)
    return experimental


def train() -> None:
    if CHECKPOINT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite prior experimental checkpoint: {CHECKPOINT_PATH}")
    before_hash = sha256(BASELINE_MODEL)
    tf.keras.utils.set_random_seed(SEED)
    cfg = configuration()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    baseline = read_baseline()
    train_ds, valid_ds, class_names, _, train_count, valid_count = load_dataset_splits(batch_size=BATCH_SIZE)
    if len(class_names) != 38 or train_count != EXPECTED_TRAIN_COUNT or valid_count != EXPECTED_VALID_COUNT:
        raise ValueError(f"Unexpected data split: classes={len(class_names)}, train={train_count}, valid={valid_count}")
    augmentation = build_augmentation()
    augmented_train = train_ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    model = build_cnn_model(num_classes=len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(CHECKPOINT_PATH), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    ]
    started = time.perf_counter()
    history_obj = model.fit(augmented_train, validation_data=valid_ds, epochs=20, callbacks=callbacks)
    duration = time.perf_counter() - started
    history = pd.DataFrame(history_obj.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    history.to_csv(HISTORY_PATH, index=False)
    best_epoch = int(history["val_accuracy"].idxmax() + 1)
    training = {"epochs_completed": len(history), "best_epoch": best_epoch, "training_duration_seconds": duration}
    (EXPERIMENT_DIR / "training_summary.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    if sha256(BASELINE_MODEL) != before_hash:
        raise RuntimeError("Baseline checkpoint changed during training; aborting evaluation")
    evaluate_checkpoint(baseline, training)
    print(f"Experiment complete: {training}; baseline checkpoint hash unchanged.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "train"), help="verify pipeline safely, or run the full experiment")
    args = parser.parse_args()
    if not BASELINE_MODEL.is_file():
        raise FileNotFoundError(f"Baseline checkpoint missing: {BASELINE_MODEL}")
    if args.command == "verify":
        verify()
    else:
        train()


if __name__ == "__main__":
    main()
