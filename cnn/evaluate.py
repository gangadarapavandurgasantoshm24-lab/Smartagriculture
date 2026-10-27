import os
import sys
from pathlib import Path

# Suppress excessive TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is in the module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Dependency validation with clear error messages
try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ImportError as err:
    print(
        f"\n[ERROR] Missing required library: {err}\n"
        "Please ensure numpy, pandas, tensorflow, and scikit-learn are installed in your environment."
    )
    sys.exit(1)

from cnn.data_loader import load_dataset_splits

# ==============================================================================
# CONFIGURATION & FILE PATHS
# ==============================================================================
MODEL_PATH = Path("cnn/best_model.keras")
REPORT_SAVE_PATH = Path("cnn/classification_report.txt")
CONFUSION_MATRIX_SAVE_PATH = Path("cnn/confusion_matrix.csv")


def evaluate_cnn():
    """
    Evaluate the saved best CNN model on the validation dataset.
    Computes overall accuracy, macro precision/recall/F1, per-class metrics,
    and confusion matrix without modifying images or retraining the model.
    """
    print("=" * 70)
    print("SMART AGRICULTURE: CNN MODEL EVALUATION")
    print("=" * 70)

    # 1. Validate existence of saved model file
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please train the model first by running: python cnn/train.py"
        )

    # 2. Load the trained model
    print(f"\n[Step 1/6] Loading saved best model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Model loaded successfully.")
    print(f"Model input shape:  {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # 3. Load the validation dataset using the verified data loader
    print("\n[Step 2/6] Loading validation dataset via cnn/data_loader.py...")
    _, valid_ds, class_names, class_to_idx, _, valid_count = load_dataset_splits()

    num_classes = len(class_names)
    print(f"Validation images:       {valid_count:,}")
    print(f"Number of classes:       {num_classes}")
    print(f"Model output dimension:  {model.output_shape[-1]}")

    # Verify class mapping alignment
    if model.output_shape[-1] != num_classes:
        raise ValueError(
            f"Class count mismatch: Model output ({model.output_shape[-1]}) does "
            f"not match detected dataset classes ({num_classes})."
        )
    print("Class mapping alignment: PASS (Model classes match dataset classes 1:1)")

    # 4. Evaluate on validation dataset using model.evaluate()
    print("\n[Step 3/6] Computing validation loss and accuracy (model.evaluate)...")
    eval_results = model.evaluate(valid_ds, verbose=1)
    val_loss = float(eval_results[0])
    val_acc = float(eval_results[1])

    print("\n--- Validation Split Results ---")
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

    # 5. Extract ground truth labels and generate predictions
    print("\n[Step 4/6] Extracting labels and generating batch predictions...")
    y_true_list = []
    for _, labels in valid_ds:
        y_true_list.extend(labels.numpy().tolist())
    y_true = np.array(y_true_list)

    # Model inference across all validation batches
    y_pred_probs = model.predict(valid_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Sample count mismatch: {len(y_true)} ground-truth labels vs "
            f"{len(y_pred)} predictions."
        )

    # 6. Compute classification metrics
    print("\n[Step 5/6] Calculating evaluation metrics...")
    overall_acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE SUMMARY (VALIDATION SPLIT)")
    print("=" * 70)
    print(f"Evaluated Samples:   {len(y_true):,}")
    print(f"Overall Accuracy:    {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    print(f"Macro Precision:     {macro_precision:.4f}")
    print(f"Macro Recall:        {macro_recall:.4f}")
    print(f"Macro F1-Score:      {macro_f1:.4f}")
    print("=" * 70)

    # Generate classification report
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    # Save classification report to file
    REPORT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SMART AGRICULTURE CNN CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model File:         {MODEL_PATH}\n")
        f.write(f"Validation Samples: {len(y_true):,}\n")
        f.write(f"Overall Accuracy:   {overall_acc:.4f}\n")
        f.write(f"Macro Precision:    {macro_precision:.4f}\n")
        f.write(f"Macro Recall:       {macro_recall:.4f}\n")
        f.write(f"Macro F1-Score:     {macro_f1:.4f}\n\n")
        f.write(report_str)
    print(f"\nClassification report saved: {REPORT_SAVE_PATH}")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    CONFUSION_MATRIX_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(CONFUSION_MATRIX_SAVE_PATH)
    print(f"Confusion matrix saved:      {CONFUSION_MATRIX_SAVE_PATH}")

    # 7. Identify Top 10 classes with lowest recall
    per_class_recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_recall_pairs = [
        (class_names[i], float(per_class_recalls[i]))
        for i in range(num_classes)
    ]
    lowest_10_recall = sorted(class_recall_pairs, key=lambda item: item[1])[:10]

    print("\n" + "=" * 70)
    print("TOP 10 CLASSES WITH LOWEST RECALL")
    print("=" * 70)
    print(f"{'Rank':<6} {'Recall':<10} {'Class Name'}")
    print("-" * 70)
    for rank, (name, rec_val) in enumerate(lowest_10_recall, start=1):
        print(f"#{rank:<5} {rec_val:<10.4f} {name}")
    print("=" * 70)

    # 8. Real-world performance disclaimer
    print("\n" + "=" * 70)
    print("CRITICAL NOTE ON REAL-WORLD GENERALIZATION")
    print("=" * 70)
    print(
        "Important: High validation performance on this augmented dataset indicates\n"
        "strong pattern recognition on benchmark lab conditions, but does NOT prove\n"
        "identical real-world field accuracy. Field conditions introduce natural domain\n"
        "shifts: complex outdoor lighting, shadows, occlusions, varying camera sensors,\n"
        "soil/background interference, and co-occurring pest/disease symptoms."
    )
    print("=" * 70)


def main():
    evaluate_cnn()


if __name__ == "__main__":
    main()
