import unittest
import numpy as np

from src.database import (
    clear_all_scans,
    delete_scan,
    get_all_scans,
    get_scan_stats,
    init_db,
    save_scan,
)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_save_and_retrieve_scan(self):
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_top = [
            {
                "rank": 1,
                "class_name": "Apple___Apple_scab",
                "condition": "Apple scab",
                "confidence_percent": 96.5,
            }
        ]

        scan_id = save_scan(
            filename="unit_test_leaf.jpg",
            predicted_class="Apple___Apple_scab",
            crop="Apple",
            condition="Apple scab",
            is_healthy=False,
            confidence=0.965,
            top_predictions=dummy_top,
            display_image_rgb=dummy_img,
            notes="Automated test note",
        )

        self.assertGreater(scan_id, 0)

        # Retrieve
        scans = get_all_scans(crop_filter="Apple")
        matching = [s for s in scans if s["id"] == scan_id]
        self.assertEqual(len(matching), 1)
        record = matching[0]
        self.assertEqual(record["filename"], "unit_test_leaf.jpg")
        self.assertEqual(record["crop"], "Apple")
        self.assertFalse(record["is_healthy"])
        self.assertAlmostEqual(record["confidence"], 0.965, places=3)
        self.assertEqual(record["notes"], "Automated test note")

        # Delete
        deleted = delete_scan(scan_id)
        self.assertTrue(deleted)

    def test_get_scan_stats(self):
        stats = get_scan_stats()
        self.assertIn("total_scans", stats)
        self.assertIn("healthy_scans", stats)
        self.assertIn("diseased_scans", stats)


if __name__ == "__main__":
    unittest.main()
