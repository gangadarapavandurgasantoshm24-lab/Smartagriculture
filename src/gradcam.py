from typing import Optional, Tuple
import cv2
import keras
import numpy as np
import streamlit as st
import tensorflow as tf

from src.config import IMAGE_SIZE, TARGET_CONV_LAYER
from src.inference import load_model


COLORMAP_OPTIONS = {
    "JET (Standard Rainbow)": cv2.COLORMAP_JET,
    "VIRIDIS (Perceptually Uniform)": cv2.COLORMAP_VIRIDIS,
    "PLASMA (Warm Glow)": cv2.COLORMAP_PLASMA,
    "INFERNO (High Contrast)": cv2.COLORMAP_INFERNO,
    "TURBO (Enhanced Detail)": cv2.COLORMAP_TURBO,
}


@st.cache_resource(show_spinner=False)
def get_grad_model(target_layer_name: str = TARGET_CONV_LAYER) -> keras.Model:
    """
    Construct and cache a multi-output gradient extraction model.
    In Keras 3, Sequential models loaded from disk are reconstructed into
    a Functional graph to ensure intermediate layer tensors are differentiable.

    Returns:
        keras.Model with inputs = model.inputs and outputs = [conv_output, predictions]
    """
    base_model = load_model()

    # Verify target layer exists
    layer_names = [l.name for l in base_model.layers]
    if target_layer_name not in layer_names:
        raise ValueError(
            f"Target layer '{target_layer_name}' not found in model layers: {layer_names}"
        )

    # Reconstruct functional graph
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="gradcam_input")
    x = inputs
    target_activation = None

    for layer in base_model.layers:
        x = layer(x)
        if layer.name == target_layer_name:
            target_activation = x

    if target_activation is None:
        raise RuntimeError(f"Failed to bind target layer '{target_layer_name}'")

    grad_model = keras.Model(inputs=inputs, outputs=[target_activation, x])
    return grad_model


def generate_gradcam_heatmap(
    input_tensor: np.ndarray,
    target_class_index: Optional[int] = None,
    target_layer_name: str = TARGET_CONV_LAYER,
) -> np.ndarray:
    """
    Calculate the Gradient-weighted Class Activation Mapping (Grad-CAM)
    for a given input image tensor targeting the specified convolutional layer.

    Args:
        input_tensor: Normalized (1, 256, 256, 3) float32 numpy array.
        target_class_index: Optional integer index of target class (defaults to top predicted).
        target_layer_name: Name of convolutional layer to target ('conv2d_3').

    Returns:
        2D numpy array (256, 256) of normalized heatmap activations [0.0, 1.0].
    """
    grad_model = get_grad_model(target_layer_name=target_layer_name)
    tf_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf_tensor)

        if target_class_index is None:
            target_class_index = int(tf.argmax(predictions[0]))

        loss = predictions[:, target_class_index]

    # Compute gradient of target class score with respect to conv_outputs
    grads = tape.gradient(loss, conv_outputs)

    # Verification: Verify that gradients are not None
    if grads is None:
        raise RuntimeError(
            f"Grad-CAM error: tape.gradient returned None for layer '{target_layer_name}'. "
            "Ensure the gradient flow path is unbroken."
        )

    # Global Average Pooling of gradients across spatial dimensions (height, width)
    # Shape: (num_channels,) = (128,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolutional feature maps by pooled gradients
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: we are only interested in features that have a positive influence on the target class
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize heatmap between 0.0 and 1.0
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy()

    # Resize coarse heatmap (e.g. 60x60) up to original image resolution (256, 256)
    heatmap_resized = cv2.resize(heatmap_np, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

    # Clip to valid [0.0, 1.0] range
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    return heatmap_resized


def apply_gradcam_overlay(
    display_image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_code: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlay the Grad-CAM heatmap onto the original RGB image.

    Args:
        display_image_rgb: (256, 256, 3) uint8 image in RGB format.
        heatmap: (256, 256) float32 normalized array in [0.0, 1.0].
        alpha: Blending weight for heatmap overlay (0.0 = original, 1.0 = heatmap only).
        colormap_code: OpenCV colormap constant.

    Returns:
        (colored_heatmap_rgb, blended_overlay_rgb) both uint8 arrays.
    """
    # Scale heatmap to [0, 255] uint8
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply OpenCV color map (returns BGR)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap_code)

    # Convert to RGB
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Ensure display image is uint8 (256, 256, 3)
    if display_image_rgb.dtype != np.uint8:
        display_image_rgb = np.uint8(display_image_rgb)

    # Alpha blending: (1 - alpha) * image + alpha * heatmap
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    blended = cv2.addWeighted(
        display_image_rgb,
        1.0 - alpha_clamped,
        colored_heatmap_rgb,
        alpha_clamped,
        0,
    )

    return colored_heatmap_rgb, blended
