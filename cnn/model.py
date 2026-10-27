import os

# Suppress excessive TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
INPUT_SHAPE = (256, 256, 3)
DEFAULT_NUM_CLASSES = 38


def build_cnn_model(num_classes: int = DEFAULT_NUM_CLASSES) -> tf.keras.Model:
    """
    Build and compile a beginner-friendly Convolutional Neural Network (CNN)
    for plant disease classification across 38 crop condition classes.

    Architecture Overview:
      Input Layer       : (256, 256, 3) RGB plant leaf image
      Conv Block 1      : Conv2D(32, 3x3, relu) -> MaxPooling2D(2x2)
      Conv Block 2      : Conv2D(64, 3x3, relu) -> MaxPooling2D(2x2)
      Conv Block 3      : Conv2D(128, 3x3, relu) -> MaxPooling2D(2x2)
      Feature Flatten   : Flatten 2D spatial features to 1D vector
      Dense Classifier  : Dense(128, relu)
      Regularization    : Dropout(0.5) to reduce overfitting
      Output Layer      : Dense(num_classes, softmax) class probabilities
    """
    model = models.Sequential(
        [
            # Explicit input specification
            layers.Input(shape=INPUT_SHAPE, name="input_image"),
            # Block 1: Basic edge and texture feature extraction
            layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_1"),
            layers.MaxPooling2D((2, 2), name="maxpool_1"),
            # Block 2: Intermediate pattern and leaf spot extraction
            layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
            layers.MaxPooling2D((2, 2), name="maxpool_2"),
            # Block 3: High-level plant disease symptom extraction
            layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_3"),
            layers.MaxPooling2D((2, 2), name="maxpool_3"),
            # Flatten multi-dimensional feature maps into a 1D feature vector
            layers.Flatten(name="flatten"),
            # Dense hidden layer for feature combinations and classification
            layers.Dense(128, activation="relu", name="dense_features"),
            # Dropout randomly deactivates 50% of neurons to prevent overfitting
            layers.Dropout(0.5, name="dropout_regularization"),
            # Output layer: 38 probability scores (one for each class)
            layers.Dense(num_classes, activation="softmax", name="output_predictions"),
        ],
        name="smart_agriculture_cnn",
    )

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    print("=" * 70)
    print("SMART AGRICULTURE: CNN MODEL ARCHITECTURE")
    print("=" * 70)

    # 1. Build the model
    num_classes = DEFAULT_NUM_CLASSES
    model = build_cnn_model(num_classes=num_classes)
    print("\nModel successfully built and compiled.")

    # 2. Print model architecture summary
    print("\n--- Model Summary ---")
    model.summary()

    # 3. Inspect input and output shapes
    input_shape = model.input_shape
    output_shape = model.output_shape
    print("\n--- Model Tensor Shapes ---")
    print(f"Input shape:  {input_shape}")
    print(f"Output shape: {output_shape}")

    # 4. Count trainable parameters
    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    total_count = trainable_count + non_trainable_count

    print("\n--- Parameter Counts ---")
    print(f"Trainable parameters:     {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print(f"Total parameters:         {total_count:,}")

    # 5. Verification checks
    input_pass = input_shape == (None, 256, 256, 3)
    output_pass = output_shape == (None, num_classes)
    compile_pass = (
        model.loss == "sparse_categorical_crossentropy"
        and "accuracy" in model.metrics_names
    )

    print("\n" + "=" * 70)
    print("CNN MODEL VERIFICATION")
    print("=" * 70)
    print(f"Architecture creation:  PASS")
    print(f"Input shape check:      {'PASS' if input_pass else 'FAIL'} ({input_shape})")
    print(f"Output shape check:     {'PASS' if output_pass else 'FAIL'} ({output_shape})")
    print(f"Loss function:          {model.loss}")
    print(f"Optimizer:              adam")
    print(f"Metrics:                ['accuracy']")
    print("-" * 70)
    print(f"Overall Status:         {'PASS' if (input_pass and output_pass) else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
