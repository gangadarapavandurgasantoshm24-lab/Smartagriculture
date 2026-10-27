import io
import unittest
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from src.config import CLASS_NAMES_PATH, NUM_CLASSES, TEST_IMAGES_DIR
from src.inference import (
    load_class_names,
    load_model,
    parse_class_label,
    predict_leaf,
    preprocess_image,
)


class TestInference(unittest.TestCase):
    def test_class_names_count_and_order(self):
        """Verify that class names map to models/class_names.json with exactly 38 deterministic classes."""
        classes = load_class_names()
        self.assertEqual(len(classes), NUM_CLASSES)
        self.assertEqual(len(classes), 38)
        self.assertEqual(classes[0], "Apple___Apple_scab")
        self.assertEqual(classes[-1], "Tomato___healthy")
        # Ensure classes are sorted alphabetically, identical to training data loader
        self.assertEqual(classes, sorted(classes))

    def test_model_loading_and_architecture(self):
        model = load_model()
        self.assertEqual(model.input_shape, (None, 256, 256, 3))
        self.assertEqual(model.output_shape, (None, NUM_CLASSES))
        layer_names = [l.name for l in model.layers]
        self.assertIn("conv2d_3", layer_names)
        self.assertIn("output_predictions", layer_names)

    def test_preprocess_image_uint8_and_dimensions(self):
        # Create a dummy RGB image of arbitrary size
        dummy_img = Image.new("RGB", (320, 240), color=(120, 180, 70))
        tensor, display = preprocess_image(dummy_img)

        self.assertEqual(tensor.shape, (1, 256, 256, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)

        self.assertEqual(display.shape, (256, 256, 3))
        self.assertEqual(display.dtype, np.uint8)

    def test_preprocess_image_no_duplicate_normalization(self):
        """Verify that already normalized float [0.0, 1.0] arrays are NOT divided again."""
        float_arr = np.full((256, 256, 3), 0.5, dtype=np.float32)
        tensor, display = preprocess_image(float_arr)
        # Should remain ~0.5, not 0.5 / 255 = 0.00196
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 0.5, places=4)
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_image_unnormalized_float_fixed(self):
        """Verify that unnormalized float arrays [0, 255] are normalized to [0, 1]."""
        float_arr = np.full((256, 256, 3), 255.0, dtype=np.float32)
        tensor, display = preprocess_image(float_arr)
        self.assertAlmostEqual(float(tensor.max()), 1.0, places=4)
        self.assertAlmostEqual(float(tensor.min()), 1.0, places=4)

    def test_preprocess_image_bytes_and_exif(self):
        """Verify that byte buffers from file upload decode properly to RGB."""
        img = Image.new("RGB", (100, 100), color=(50, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        tensor, display = preprocess_image(raw_bytes)
        self.assertEqual(tensor.shape, (1, 256, 256, 3))
        self.assertEqual(display.shape, (256, 256, 3))
        # Check channel order: R ~ 50/255=0.196, G ~ 150/255=0.588, B ~ 200/255=0.784
        self.assertAlmostEqual(float(tensor[0, 50, 50, 0]), 50 / 255.0, places=1)
        self.assertAlmostEqual(float(tensor[0, 50, 50, 1]), 150 / 255.0, places=1)
        self.assertAlmostEqual(float(tensor[0, 50, 50, 2]), 200 / 255.0, places=1)

    def test_plotly_margin_property_validity(self):
        """BUG 1 Regression: Verify Plotly margin does not accept 'bottom', but accepts 'b'."""
        fig = go.Figure(go.Bar(x=[10, 20], y=["A", "B"], orientation="h"))
        # Valid margin keys: l, r, t, b, pad
        fig.update_layout(margin=dict(l=10, r=20, t=10, b=10))

        # 'bottom' raises ValueError in Plotly
        with self.assertRaises(ValueError):
            fig.update_layout(margin=dict(l=10, r=20, t=10, bottom=10))

    def test_parse_class_label(self):
        crop, cond, healthy = parse_class_label("Corn_(maize)___Northern_Leaf_Blight")
        self.assertEqual(crop, "Corn (maize)")
        self.assertEqual(cond, "Northern Leaf Blight")
        self.assertFalse(healthy)

        crop_h, cond_h, healthy_h = parse_class_label("Tomato___healthy")
        self.assertEqual(crop_h, "Tomato")
        self.assertEqual(cond_h, "healthy")
        self.assertTrue(healthy_h)

        crop_c, cond_c, healthy_c = parse_class_label("Corn_(maize)___healthy")
        self.assertEqual(crop_c, "Corn (maize)")
        self.assertEqual(cond_c, "healthy")
        self.assertTrue(healthy_c)

    def test_predict_leaf_top_3_structure(self):
        dummy_img = np.full((256, 256, 3), 100, dtype=np.uint8)
        result = predict_leaf(dummy_img, top_k=3)

        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertEqual(len(result["all_probabilities"]), NUM_CLASSES)

        # Check top_predictions keys
        for p in result["top_predictions"]:
            self.assertIn("rank", p)
            self.assertIn("class_name", p)
            self.assertIn("confidence", p)
            self.assertIn("confidence_percent", p)
            self.assertIn("is_healthy", p)

    def test_prediction_samples_on_known_images(self):
        """BUG 2 Verification: Test predictions for known maize, pepper, and tomato images."""
        valid_dir = TEST_IMAGES_DIR
        if not valid_dir.exists():
            self.skipTest("Test images directory not found.")

        # Test any available validation images
        sample_images = list(valid_dir.glob("*.jpg")) + list(valid_dir.glob("*.JPG"))
        if not sample_images:
            self.skipTest("No sample images found in TEST_IMAGES_DIR.")

        for img_path in sample_images[:4]:
            res = predict_leaf(img_path, top_k=3)
            self.assertIn("predicted_class", res)
            self.assertIn(res["predicted_class"], load_class_names())
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)

    def test_center_crop_preprocessing_square(self):
        """Verify center_crop extracts central square region preserving aspect ratio."""
        # Create non-square rectangular test image (100 height x 200 width)
        rect_img = Image.new("RGB", (200, 100), color=(100, 150, 200))
        tensor, display = preprocess_image(rect_img, center_crop=True)
        self.assertEqual(tensor.shape, (1, 256, 256, 3))
        self.assertEqual(display.shape, (256, 256, 3))

    def test_reproduce_and_disambiguate_apple2_failure(self):
        """Verify reproduction of images_apple2.jpg: raw ambiguity and Apple disambiguation."""
        from pathlib import Path
        img_path = Path("data/user_failure_cases/images_apple2.jpg")
        if not img_path.exists():
            self.skipTest("images_apple2.jpg not found in user failure test cases.")

        # 1. Raw prediction: Cross-crop ambiguity is detected
        res_raw = predict_leaf(img_path, top_k=3)
        self.assertEqual(res_raw["raw_predicted_class"], "Peach___Bacterial_spot")
        self.assertAlmostEqual(res_raw["raw_confidence_percent"], 72.40, delta=1.0)
        self.assertTrue(res_raw["cross_crop_ambiguous"])
        self.assertIn("Peach", res_raw["conflicting_crops"])
        self.assertIn("Corn (maize)", res_raw["conflicting_crops"])

        # 2. Confirmed Crop: Disambiguates to Apple Black rot (> 95% within Apple)
        res_apple = predict_leaf(img_path, top_k=3, confirmed_crop="Apple")
        self.assertEqual(res_apple["predicted_class"], "Apple___Black_rot")
        self.assertEqual(res_apple["crop"], "Apple")
        self.assertEqual(res_apple["condition"], "Black rot")
        self.assertGreater(res_apple["confidence_percent"], 95.0)

    def test_reproduce_and_resolve_corn_health_failure(self):
        """Verify reproduction of images_corn health.jpg: squashed vs center-crop resolution."""
        from pathlib import Path
        img_path = Path("data/user_failure_cases/images_corn health.jpg")
        if not img_path.exists():
            self.skipTest("images_corn health.jpg not found in user failure test cases.")

        # 1. Direct uncropped squash resize reproduces Pepper bell artifact
        res_squash = predict_leaf(img_path, top_k=3, center_crop=False)
        self.assertEqual(res_squash["predicted_class"], "Pepper,_bell___healthy")
        self.assertGreater(res_squash["confidence_percent"], 99.0)

        # 2. Center leaf focus resolves framing distortion -> 100% Corn healthy
        res_center = predict_leaf(img_path, top_k=3, center_crop=True)
        self.assertEqual(res_center["predicted_class"], "Corn_(maize)___healthy")
        self.assertEqual(res_center["crop"], "Corn (maize)")
        self.assertTrue(res_center["is_healthy"])
        self.assertGreater(res_center["confidence_percent"], 99.0)


if __name__ == "__main__":
    unittest.main()


