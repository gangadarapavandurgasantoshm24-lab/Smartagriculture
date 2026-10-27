"""Run a reproducible, inference-only analysis of the saved CNN."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cnn.data_loader import IMAGE_SIZE, TRAIN_DIR, VALID_DIR, detect_classes

MODEL = ROOT / "cnn" / "best_model.keras"
REPORTS = ROOT / "reports"
OUTPUTS = ["error_analysis.md", "classification_report.csv", "confusion_matrix.csv",
           "confusion_matrix.png", "normalized_confusion_matrix.png",
           "misclassified_samples.csv", "misclassified_gallery.png"]


def unused_path(name: str) -> Path:
    path = REPORTS / name
    if not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No unused filename available for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for p in (MODEL, TRAIN_DIR, VALID_DIR):
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")
    classes = detect_classes(TRAIN_DIR)
    valid_classes = sorted(p.name for p in VALID_DIR.iterdir() if p.is_dir())
    if classes != valid_classes:
        raise ValueError("Train and validation class directories do not match")

    model = tf.keras.models.load_model(MODEL, compile=False)
    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if model.output_shape[-1] != len(classes):
        raise ValueError(f"Model outputs {model.output_shape[-1]} classes, dataset has {len(classes)}")

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR, labels="inferred", label_mode="int", class_names=classes,
        color_mode="rgb", batch_size=args.batch_size, image_size=IMAGE_SIZE,
        shuffle=False,
    )
    paths = [str(Path(p).resolve()) for p in raw_ds.file_paths]
    ds = raw_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    truth = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)
    if not (len(paths) == len(truth) == len(probs)):
        raise RuntimeError("Validation paths, labels, and predictions have different lengths")
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    ids = np.arange(len(classes))
    cm = confusion_matrix(truth, pred, labels=ids)
    norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    report = classification_report(truth, pred, labels=ids, target_names=classes, output_dict=True, zero_division=0)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = {name: unused_path(name) for name in OUTPUTS}

    rows = [{"class": c, "precision": report[c]["precision"], "recall": report[c]["recall"],
             "f1_score": report[c]["f1-score"], "support": int(report[c]["support"])} for c in classes]
    summary_rows = [{"class": key, "precision": report[key]["precision"], "recall": report[key]["recall"],
                     "f1_score": report[key]["f1-score"], "support": len(truth)} for key in ("macro avg", "weighted avg")]
    pd.DataFrame(rows + summary_rows).to_csv(out[OUTPUTS[1]], index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out[OUTPUTS[2]], index_label="actual_class")

    for matrix, title, dest, fmt in ((cm, "Confusion matrix (count)", out[OUTPUTS[3]], "d"),
                                      (norm, "Normalized confusion matrix (row proportion)", out[OUTPUTS[4]], ".2f")):
        fig, ax = plt.subplots(figsize=(22, 19))
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set(title=title, xlabel="Predicted class", ylabel="Actual class")
        ax.set_xticks(ids, classes, rotation=90, fontsize=7)
        ax.set_yticks(ids, classes, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
        fig.tight_layout()
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        plt.close(fig)

    err = pd.DataFrame({"image_path": paths, "actual_class": [classes[i] for i in truth],
                        "predicted_class": [classes[i] for i in pred], "confidence": conf})
    err = err.loc[truth != pred].copy()
    err.to_csv(out[OUTPUTS[5]], index=False)
    chosen = err.head(24)
    cols = 4
    fig, axes = plt.subplots(max(1, (len(chosen) + cols - 1) // cols), cols,
                             figsize=(16, max(4, 4 * ((len(chosen) + cols - 1) // cols))), squeeze=False)
    if chosen.empty:
        axes.flat[0].text(.5, .5, "No validation errors", ha="center", va="center")
    for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
        try:
            with Image.open(row.image_path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"True: {row.actual_class}\nPred: {row.predicted_class}\nConfidence: {row.confidence:.1%}", fontsize=8)
        except Exception as exc:
            ax.text(.5, .5, f"Could not display image: {exc}", ha="center")
        ax.axis("off")
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Representative validation errors; confidence is predicted-class softmax")
    fig.tight_layout()
    fig.savefig(out[OUTPUTS[6]], dpi=160, bbox_inches="tight")
    plt.close(fig)

    pairs = Counter((classes[a], classes[b]) for a, b in zip(truth, pred) if a != b)
    weakest = sorted(rows, key=lambda row: (row["recall"], row["class"]))[:5]
    pair_lines = "\n".join(f"{i}. `{a}` → `{b}`: {n}" for i, ((a, b), n) in enumerate(pairs.most_common(10), 1)) or "No errors."
    weak_lines = "\n".join(f"{i}. `{r['class']}`: {r['recall']:.4%} (support {r['support']})" for i, r in enumerate(weakest, 1))
    mapping = "\n".join(f"- {i}: `{c}`" for i, c in enumerate(classes))
    prior = ROOT / "cnn" / "classification_report.txt"
    accuracy = float(np.mean(truth == pred))
    text = ("# CNN validation error analysis\n\n"
        f"Model: `{MODEL.relative_to(ROOT)}`; validation split: `{VALID_DIR}`.\n\n"
        "## Verified setup\n\n"
        f"- Input/output shapes: `{model.input_shape}` / `{model.output_shape}`.\n"
        f"- Preprocessing: RGB, resize to {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}, float32 divided by 255.\n"
        f"- Training and validation directories: `{TRAIN_DIR}` and `{VALID_DIR}`.\n"
        f"- Validation images: {len(truth):,}; classes: {len(classes)}; labels are alphabetical, as in `cnn/data_loader.py`.\n\n"
        "## Metrics\n\n"
        f"- Accuracy: {accuracy:.6f} ({accuracy:.2%})\n"
        f"- Macro precision/recall/F1: {report['macro avg']['precision']:.6f} / {report['macro avg']['recall']:.6f} / {report['macro avg']['f1-score']:.6f}\n"
        f"- Weighted precision/recall/F1: {report['weighted avg']['precision']:.6f} / {report['weighted avg']['recall']:.6f} / {report['weighted avg']['f1-score']:.6f}\n\n"
        f"## Five weakest classes by recall\n\n{weak_lines}\n\n## Frequent directed confusions\n\n{pair_lines}\n\n"
        "## Error observations\n\nThe confusion pairs above are verified counts on this validation split. A plausible hypothesis is overlapping visible lesion or texture patterns across some diseases, but these metrics alone cannot establish a biological cause.\n\n"
        f"## Index-to-label mapping\n\n{mapping}\n\n"
        "## Comparison with existing report\n\n"
        f"The existing `{prior.relative_to(ROOT)}` reports 17,572 validation images, 93.22% accuracy, macro F1 0.9319, and weighted F1 0.9320. This run independently evaluates the saved model; compare the results above for consistency.\n\n"
        "CSV confidence is the softmax score for the predicted class. The gallery shows up to 24 misclassified examples.\n")
    out[OUTPUTS[0]].write_text(text, encoding="utf-8")
    for path in out.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Output missing or empty: {path}")
    print(f"Accuracy: {accuracy:.6f}; macro F1: {report['macro avg']['f1-score']:.6f}; weighted F1: {report['weighted avg']['f1-score']:.6f}")
    print("Weakest recalls:")
    for r in weakest:
        print(f"  {r['class']}: {r['recall']:.6f}")
    print("Top confusion pairs:")
    for (a, b), n in pairs.most_common(5):
        print(f"  {a} -> {b}: {n}")
    for path in out.values():
        print(f"Generated: {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
