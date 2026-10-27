import argparse
from pathlib import Path

import cv2


# The dataset review found that sample images are 256 x 256 pixels.
# This script checks for that size, but it does not resize the image.
EXPECTED_SIZE = (256, 256)


def preprocess_image(image_path):
    """Preprocess one image and return a normalized RGB array."""
    image_path = Path(image_path)

    # cv2.imread() reads the image from disk in BGR color order.
    # This does not modify the original image file.
    image_bgr = cv2.imread(str(image_path))

    # If OpenCV cannot read the file, it returns None instead of an image array.
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read this image: {image_path}")

    # OpenCV image shape is stored as: height, width, color_channels.
    original_height, original_width = image_bgr.shape[:2]
    print(f"Original dimensions: {original_width} x {original_height}")

    # Keep the original file unchanged. Only warn if the size is different.
    if (original_width, original_height) == EXPECTED_SIZE:
        print("Size check: image is 256 x 256.")
    else:
        print(
            "Size warning: image is not 256 x 256. "
            "Resizing may be required later depending on the model."
        )

    # Convert BGR to RGB in memory because most deep learning workflows expect RGB.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert integer pixels from 0-255 into floating point values from 0.0-1.0.
    # This normalized array is for model input, not for saving as a JPG.
    normalized_rgb = image_rgb.astype("float32") / 255.0

    print(f"Original shape: {image_bgr.shape}")
    print(f"RGB shape: {image_rgb.shape}")
    print(f"Original data type: {image_bgr.dtype}")
    print(f"Normalized data type: {normalized_rgb.dtype}")
    print(f"Normalized minimum pixel value: {normalized_rgb.min():.4f}")
    print(f"Normalized maximum pixel value: {normalized_rgb.max():.4f}")
    print("The normalized RGB image is ready for model input.")
    print("No image file was saved or modified.")

    # Return the array so future model training code can use it.
    return normalized_rgb


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess one plant leaf image without changing the raw dataset."
    )
    parser.add_argument(
        "image_path",
        help="Path to one input image, for example: data/raw/test/test/AppleScab1.JPG",
    )
    args = parser.parse_args()

    preprocess_image(args.image_path)


if __name__ == "__main__":
    main()
