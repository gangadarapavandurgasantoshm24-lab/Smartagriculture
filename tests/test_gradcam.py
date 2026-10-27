import unittest
import numpy as np

from src.gradcam import (
    apply_gradcam_overlay,
    generate_gradcam_heatmap,
    get_grad_model,
)


class TestGradCAM(unittest.TestCase):
    def test_grad_model_target_layer(self):
        grad_model = get_grad_model("conv2d_3")
        self.assertIsNotNone(grad_model)
        # Check outputs
        self.assertEqual(len(grad_model.outputs), 2)
        # First output is conv2d_3 activation, second is predictions
        conv_output = grad_model.outputs[0]
        preds_output = grad_model.outputs[1]
        self.assertEqual(conv_output.shape[-1], 128)
        self.assertEqual(preds_output.shape[-1], 38)

    def test_gradcam_heatmap_computation(self):
        # Create dummy normalized input tensor
        dummy_tensor = np.random.uniform(0.0, 1.0, size=(1, 256, 256, 3)).astype(
            np.float32
        )
        heatmap = generate_gradcam_heatmap(
            dummy_tensor, target_class_index=2, target_layer_name="conv2d_3"
        )

        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.shape, (256, 256))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_apply_gradcam_overlay(self):
        display_img = np.full((256, 256, 3), 120, dtype=np.uint8)
        heatmap = np.random.uniform(0.0, 1.0, size=(256, 256)).astype(np.float32)

        colored, blended = apply_gradcam_overlay(display_img, heatmap, alpha=0.5)

        self.assertEqual(colored.shape, (256, 256, 3))
        self.assertEqual(colored.dtype, np.uint8)
        self.assertEqual(blended.shape, (256, 256, 3))
        self.assertEqual(blended.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
