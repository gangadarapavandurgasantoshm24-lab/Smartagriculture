import argparse
import os
import sys
from pathlib import Path

# Suppress excessive TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure the project root is in the module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import tensorflow as tf

from cnn.data_loader import load_dataset_splits
from cnn.model import build_cnn_model

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
NUM_CLASSES = 38

MODEL_SAVE_PATH = Path("cnn/best_model.keras")
HISTORY_SAVE_PATH = Path("cnn/training_history.csv")


def train_cnn(epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Execute CNN model training with early stopping and best-model checkpointing.
    Saves the best model weights and exports training history to CSV.
    """
    print("=" * 70)
    print("SMART AGRICULTURE: CNN TRAINING PIPELINE")
    print("=" * 70)
    print(f"Epochs:              {epochs}")
    print(f"Batch Size:          {batch_size}")
    print(f"Classes:             {NUM_CLASSES}")
    print(f"Model Checkpoint:    {MODEL_SAVE_PATH}")
    print(f"History Output:      {HISTORY_SAVE_PATH}")
    print("=" * 70)

    # 1. Load training and validation datasets
    print("\n[Step 1/5] Loading datasets via cnn/data_loader.py...")
    (
        train_ds,
        valid_ds,
        class_names,
        class_to_idx,
        train_count,
        valid_count,
    ) = load_dataset_splits(batch_size=batch_size)

    print(f"Training images:     {train_count:,}")
    print(f"Validation images:   {valid_count:,}")
    print(f"Total classes:       {len(class_names)}")

    # 2. Build and compile the CNN model
    print("\n[Step 2/5] Building CNN architecture via cnn/model.py...")
    model = build_cnn_model(num_classes=NUM_CLASSES)
    model.summary()

    # 3. Configure training callbacks
    print("\n[Step 3/5] Setting up callbacks...")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    callbacks = [early_stopping, checkpoint]
    print("  - EarlyStopping: monitor='val_loss', patience=3, restore_best_weights=True")
    print("  - ModelCheckpoint: monitor='val_accuracy', mode='max', save_best_only=True")

    # 4. Train the model
    print("\n[Step 4/5] Starting training with model.fit()...\n")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # 5. Extract training statistics
    history_dict = history.history
    epochs_completed = len(history_dict["loss"])
    best_val_acc = max(history_dict["val_accuracy"])
    best_val_loss = min(history_dict["val_loss"])
    best_acc_epoch = history_dict["val_accuracy"].index(best_val_acc) + 1
    best_loss_epoch = history_dict["val_loss"].index(best_val_loss) + 1

    # 6. Save training history to CSV
    print("\n[Step 5/5] Saving training history to CSV...")
    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, epochs_completed + 1)),
            "loss": history_dict["loss"],
            "accuracy": history_dict["accuracy"],
            "val_loss": history_dict["val_loss"],
            "val_accuracy": history_dict["val_accuracy"],
        }
    )
    HISTORY_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(HISTORY_SAVE_PATH, index=False)
    print(f"Training history saved: {HISTORY_SAVE_PATH}")

    # 7. Print final training summary
    print("\n" + "=" * 70)
    print("CNN TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total Epochs Completed:       {epochs_completed} / {epochs}")
    print(f"Best Validation Accuracy:     {best_val_acc:.4f} (Epoch {best_acc_epoch})")
    print(f"Best Validation Loss:         {best_val_loss:.4f} (Epoch {best_loss_epoch})")
    print(f"Best Model File:              {MODEL_SAVE_PATH}")
    print(f"History File:                 {HISTORY_SAVE_PATH}")
    print("=" * 70)

    return model, history_df


def main():
    parser = argparse.ArgumentParser(
        description="Train the Smart Agriculture CNN for plant disease classification."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for data loader (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    train_cnn(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
