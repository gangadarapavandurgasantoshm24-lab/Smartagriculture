# Controlled online augmentation experiment

Status: **Implemented and smoke-verified; training not run in this verification step.**

## Baseline pipeline findings

The training tree already contains offline augmented image files (for example, filenames with `_90deg`, `_270deg`, and `_new30degFlipLR`). The loader itself has no online augmentation. It resizes RGB images to 256×256, divides float32 pixels by 255, shuffles training data with seed 42, and keeps validation unshuffled and deterministic.

The CNN is unchanged: three Conv2D/MaxPooling blocks (32, 64, 128 filters), Flatten, Dense(128), Dropout(0.5), and 38-way softmax. It uses Adam defaults and sparse categorical cross-entropy. Baseline training uses batch size 32, a 20 epoch maximum, early stopping on `val_loss` (patience 3, restore best weights), and best-only checkpointing on `val_accuracy`. Only the training shuffle has an explicit baseline seed; model initialization is not seeded. The existing history contains 10 epochs, with best recorded `val_accuracy` at epoch 7.

## Single intervention

Training batches receive mild online horizontal/vertical flips, ±0.04 random rotation, ±0.04 translations, 0.08 zoom, and 0.08 brightness variation. Validation receives only the baseline resize and 1/255 normalization. Architecture, loss, optimizer, batch size, class mapping, data split, epoch cap, and callbacks are held constant. Seed: 42. Full settings are in `cnn\experiments\augmentation_v1\config.json`.

## Baseline reference

- Accuracy: 93.2165%
- Macro F1: 0.931868
- Weighted F1: 0.931973

Weak class baseline recalls:
- Peach___Bacterial_spot: 77.3420%
- Tomato___Septoria_leaf_spot: 78.2110%
- Tomato___Early_blight: 78.3333%
- Tomato___Target_Spot: 81.6193%
- Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot: 82.9268%

## Run status and comparison

A full run is compute intensive on this native Windows CPU-only TensorFlow environment. The augmentation and deterministic validation pipelines were smoke-tested on a batch; the full training run remains available via the command below.

No experimental metric is reported until a complete training run and evaluation on all 17,572 validation images finish. See `reports/experiment_comparison.csv` for the baseline values and experimental fields (blank until evaluation).

## Reproduction

Train and evaluate: `python cnn/experiments/augmentation_experiment.py train`

Baseline SHA-256 recorded before the experiment: `b3b3a1621e3c37fcf7b901a24c3f2548b66cac82b6a1c4d04102dc1abc1788e0`.
