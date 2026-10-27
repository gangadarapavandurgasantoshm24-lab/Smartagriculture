import os
import unittest
import numpy as np
import streamlit as st

from src.config import CLASS_NAMES_PATH
from src.inference import load_class_names
from src.pesticides import (
    get_pesticide_audit_summary,
    get_verified_products,
    load_verified_pesticides_catalog,
)
from src.report_generator import generate_printable_html_report
from src.session_history import (
    add_session_scan,
    clear_session_history,
    get_session_scan_by_id,
    get_session_scan_stats,
    get_session_scans,
    init_session_history,
)


class TestSmartAgriRefinement(unittest.TestCase):
    """
    Comprehensive test suite for the SmartAgri AI application refinement:
      - Final 5-page sidebar navigation
      - In-memory session scan history
      - Verified commercial pesticide product catalog (CIBRC India)
      - Print report generation across all required clinical scenarios
    """

    def setUp(self):
        init_session_history()
        clear_session_history()

    # =========================================================================
    # 1. NAVIGATION & 5-PAGE STRUCTURE TESTS
    # =========================================================================
    def test_sidebar_navigation_exactly_five_pages(self):
        """Verify app.py defines exactly five sidebar pages in the specified order."""
        import app
        import inspect

        # Inspect main function source code
        src = inspect.getsource(app.main)

        expected_pages = [
            "Dashboard",
            "Disease Detection",
            "AI Explainability",
            "Scan History",
            "Settings/About",
        ]

        # Ensure all 5 expected pages are in nav_pages
        for p in expected_pages:
            self.assertIn(f'"{p}"', src, f"Expected page '{p}' must be present in app.py navigation.")

        # Ensure Model Performance and Disease Library are removed from nav_pages
        self.assertNotIn(
            '"Model Performance"',
            src,
            "'Model Performance' must be removed from app.py sidebar navigation.",
        )
        self.assertNotIn(
            '"Disease Library"',
            src,
            "'Disease Library' must be removed from app.py sidebar navigation.",
        )

    def test_original_38_classes_unaltered(self):
        """Verify that the 38 model classes and class_names.json are strictly preserved."""
        classes = load_class_names()
        self.assertEqual(len(classes), 38, "Model class count must remain exactly 38.")
        self.assertEqual(classes[0], "Apple___Apple_scab")
        self.assertEqual(classes[-1], "Tomato___healthy")

    # =========================================================================
    # 2. SESSION-BASED IN-MEMORY SCAN HISTORY TESTS
    # =========================================================================
    def test_session_history_in_memory_storage(self):
        """Verify that scans are stored in-memory in st.session_state with no persistent SQLite writes."""
        from src.database import get_all_scans as get_sqlite_scans

        sqlite_scans_before = len(get_sqlite_scans())

        dummy_img = np.full((256, 256, 3), 120, dtype=np.uint8)
        dummy_top = [
            {
                "rank": 1,
                "class_name": "Tomato___Late_blight",
                "crop": "Tomato",
                "condition": "Late blight",
                "confidence_percent": 94.2,
            }
        ]

        # Add scan to session history
        scan = add_session_scan(
            filename="session_test_tomato.jpg",
            crop="Tomato",
            condition="Late blight",
            is_healthy=False,
            predicted_class="Tomato___Late_blight",
            confidence=0.942,
            top_predictions=dummy_top,
            display_image=dummy_img,
            notes="Session test note",
        )

        self.assertTrue(scan["id"].startswith("SCAN-"))
        self.assertEqual(scan["crop"], "Tomato")
        self.assertAlmostEqual(scan["confidence_percent"], 94.2, places=2)
        self.assertTrue(scan["has_images"])
        self.assertIsNotNone(scan["image_b64"])

        # Check in-memory list
        session_scans = get_session_scans()
        self.assertEqual(len(session_scans), 1)
        self.assertEqual(session_scans[0]["id"], scan["id"])

        # VERIFY ZERO PERSISTENT SQLITE WRITES OCCURRED
        sqlite_scans_after = len(get_sqlite_scans())
        self.assertEqual(
            sqlite_scans_before,
            sqlite_scans_after,
            "add_session_scan must NOT write persistent records to SQLite database.",
        )

    def test_session_history_isolation_and_clear(self):
        """Verify that multiple scans retain independent data and clear_session_history clears in-memory state."""
        # Add Scan 1: Maize healthy
        scan1 = add_session_scan(
            filename="maize_healthy.jpg",
            crop="Corn (maize)",
            condition="healthy",
            is_healthy=True,
            predicted_class="Corn_(maize)___healthy",
            confidence=0.999,
            top_predictions=[],
        )

        # Add Scan 2: Apple scab
        scan2 = add_session_scan(
            filename="apple_scab.jpg",
            crop="Apple",
            condition="Apple scab",
            is_healthy=False,
            predicted_class="Apple___Apple_scab",
            confidence=0.885,
            top_predictions=[],
        )

        scans = get_session_scans()
        self.assertEqual(len(scans), 2)

        # Verify scan 1 and scan 2 independence
        retrieved1 = get_session_scan_by_id(scan1["id"])
        retrieved2 = get_session_scan_by_id(scan2["id"])

        self.assertIsNotNone(retrieved1)
        self.assertIsNotNone(retrieved2)
        self.assertEqual(retrieved1["crop"], "Corn (maize)")
        self.assertTrue(retrieved1["is_healthy"])
        self.assertEqual(retrieved2["crop"], "Apple")
        self.assertFalse(retrieved2["is_healthy"])

        # Verify stats
        stats = get_session_scan_stats()
        self.assertEqual(stats["total_scans"], 2)
        self.assertEqual(stats["healthy_scans"], 1)
        self.assertEqual(stats["diseased_scans"], 1)

        # Clear session history
        clear_session_history()
        self.assertEqual(len(get_session_scans()), 0)
        self.assertIsNone(get_session_scan_by_id(scan1["id"]))
        self.assertIsNone(get_session_scan_by_id(scan2["id"]))

    # =========================================================================
    # 3. VERIFIED COMMERCIAL PESTICIDE PRODUCT TESTS
    # =========================================================================
    def test_verified_commercial_pesticides_audit_and_fields(self):
        """Verify CIBRC commercial trade names and mandatory regulatory fields."""
        summary = get_pesticide_audit_summary()
        self.assertEqual(summary["total_classes"], 38)
        self.assertEqual(summary["classes_with_verified_products"], 10)
        self.assertEqual(summary["healthy_classes_no_treatment"], 12)
        self.assertEqual(summary["classes_requiring_expert_confirmation"], 16)
        self.assertGreaterEqual(summary["total_commercial_products"], 15)

        # Check Tomato Late Blight commercial product
        tomato_lb_prods = get_verified_products("Tomato___Late_blight")
        self.assertGreaterEqual(len(tomato_lb_prods), 1)
        trade_names = [p["trade_name"] for p in tomato_lb_prods]
        self.assertTrue(any("Curzate" in t for t in trade_names))

        for p in tomato_lb_prods:
            self.assertTrue(p["active_ingredients"])
            self.assertTrue(p["formulation"])
            self.assertTrue(p["manufacturer"])
            self.assertTrue(p["dosage"])
            self.assertTrue(p["pre_harvest_interval"])
            self.assertTrue(p["re_entry_interval"])
            self.assertEqual(p["verification_status"], "verified")
            self.assertIn("CIBRC", p["country_registration"])

        # Check Potato Late Blight commercial product
        potato_lb_prods = get_verified_products("Potato___Late_blight")
        self.assertTrue(any("Ridomil Gold" in p["trade_name"] for p in potato_lb_prods))

        # Check Corn Northern Leaf Blight commercial product
        corn_nlb_prods = get_verified_products("Corn_(maize)___Northern_Leaf_Blight")
        self.assertTrue(any("Amistar Top" in p["trade_name"] for p in corn_nlb_prods))

    def test_unverified_classes_have_zero_commercial_products(self):
        """Verify that classes without CIBRC verification return empty product list."""
        # Apple Black rot requires expert confirmation
        self.assertEqual(len(get_verified_products("Apple___Black_rot")), 0)
        # Peach Bacterial spot requires expert confirmation
        self.assertEqual(len(get_verified_products("Peach___Bacterial_spot")), 0)
        # Grape Esca requires expert confirmation
        self.assertEqual(len(get_verified_products("Grape___Esca_(Black_Measles)")), 0)

    # =========================================================================
    # 4. PRINT REPORT GENERATOR TESTS (ALL REQUIRED SCENARIOS)
    # =========================================================================
    def test_print_report_healthy_maize_no_pesticides(self):
        """SCENARIO 1: Healthy maize report must show prominent message and zero pesticides."""
        scan = {
            "id": "SCAN-20261026-0001",
            "timestamp": "2026-10-26 14:00:00",
            "filename": "maize_healthy_leaf.jpg",
            "crop": "Corn (maize)",
            "condition": "healthy",
            "is_healthy": True,
            "predicted_class": "Corn_(maize)___healthy",
            "confidence_percent": 99.98,
            "top_predictions": [
                {"rank": 1, "condition": "healthy", "crop": "Corn (maize)", "confidence_percent": 99.98}
            ],
            "image_b64": None,
            "overlay_b64": None,
        }

        html = generate_printable_html_report(scan)

        # Check mandatory header & identity
        self.assertIn("SmartAgri AI — Plant Disease Diagnostics", html)
        self.assertIn("PLANT LEAF ASSESSMENT REPORT", html)
        self.assertIn("SCAN-20261026-0001", html)
        self.assertIn("Corn (maize): healthy", html)

        # Check prominent healthy message
        self.assertIn(
            "Healthy Leaf Detected — No Disease Treatment or Pesticide is Required Based on this Prediction",
            html,
        )

        # STRICT REQUIREMENT: No pesticide recommendations or commercial products
        self.assertNotIn("Curzate", html)
        self.assertNotIn("Ridomil", html)
        self.assertNotIn("Commercial Trade Name", html)
        self.assertNotIn("Verified Commercial Pesticide Options", html)

    def test_print_report_diseased_maize_with_verified_guidance(self):
        """SCENARIO 2 & 3: Diseased maize / verified chemical guidance (Amistar Top)."""
        scan = {
            "id": "SCAN-20261026-0002",
            "timestamp": "2026-10-26 14:05:00",
            "filename": "corn_nlb.jpg",
            "crop": "Corn (maize)",
            "condition": "Northern Leaf Blight",
            "is_healthy": False,
            "predicted_class": "Corn_(maize)___Northern_Leaf_Blight",
            "confidence_percent": 96.40,
            "top_predictions": [
                {"rank": 1, "condition": "Northern Leaf Blight", "crop": "Corn (maize)", "confidence_percent": 96.40},
                {"rank": 2, "condition": "Common Rust", "crop": "Corn (maize)", "confidence_percent": 3.10},
            ],
            "image_b64": None,
            "overlay_b64": None,
        }

        html = generate_printable_html_report(scan)

        self.assertIn("Corn (maize): Northern Leaf Blight", html)
        self.assertIn("96.40%", html)
        self.assertIn("Amistar Top", html)
        self.assertIn("Azoxystrobin + Difenoconazole", html)
        self.assertIn("Syngenta India Ltd", html)
        self.assertIn("CIBRC VERIFIED", html)

    def test_print_report_unverified_disease_requires_expert_confirmation(self):
        """SCENARIO 4: Disease requiring expert confirmation (Apple Black rot)."""
        scan = {
            "id": "SCAN-20261026-0003",
            "timestamp": "2026-10-26 14:10:00",
            "filename": "apple_frogeye.jpg",
            "crop": "Apple",
            "condition": "Black rot",
            "is_healthy": False,
            "predicted_class": "Apple___Black_rot",
            "confidence_percent": 99.37,
            "top_predictions": [
                {"rank": 1, "condition": "Black rot", "crop": "Apple", "confidence_percent": 99.37}
            ],
            "image_b64": None,
            "overlay_b64": None,
        }

        html = generate_printable_html_report(scan)

        self.assertIn("Apple: Black rot", html)
        self.assertIn("Verified Indian Trade Name Unavailable", html)
        self.assertIn("A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer.", html)

    def test_print_report_missing_image_graceful_handling(self):
        """SCENARIO 5: Missing image data generates complete report gracefully without crashing."""
        scan = {
            "id": "SCAN-20261026-0004",
            "timestamp": "2026-10-26 14:15:00",
            "filename": "no_image_scan.jpg",
            "crop": "Tomato",
            "condition": "Early blight",
            "is_healthy": False,
            "predicted_class": "Tomato___Early_blight",
            "confidence_percent": 88.0,
            "top_predictions": [],
            "image_b64": None,
            "overlay_b64": None,
        }

        html = generate_printable_html_report(scan)
        self.assertIn("SCAN-20261026-0004", html)
        self.assertIn("Tomato: Early blight", html)
        self.assertIn("Leaf photograph and Grad-CAM overlay are unavailable for this session entry", html)

    def test_print_report_multiple_history_isolation(self):
        """SCENARIO 6: Reports from different scans are strictly isolated to their own data."""
        scan_a = add_session_scan(
            filename="scan_a.jpg",
            crop="Apple",
            condition="Apple scab",
            is_healthy=False,
            predicted_class="Apple___Apple_scab",
            confidence=0.91,
            top_predictions=[],
        )

        scan_b = add_session_scan(
            filename="scan_b.jpg",
            crop="Tomato",
            condition="healthy",
            is_healthy=True,
            predicted_class="Tomato___healthy",
            confidence=0.99,
            top_predictions=[],
        )

        html_a = generate_printable_html_report(scan_a)
        html_b = generate_printable_html_report(scan_b)

        self.assertIn(scan_a["id"], html_a)
        self.assertIn("Apple: Apple scab", html_a)
        self.assertNotIn("Tomato: healthy", html_a)

        self.assertIn(scan_b["id"], html_b)
        self.assertIn("Tomato: healthy", html_b)
        self.assertNotIn("Apple: Apple scab", html_b)

    def test_print_report_print_media_css(self):
        """SCENARIO 7: Generated report contains proper @media print CSS for A4 printing/PDF."""
        scan = {
            "id": "SCAN-PRINT-001",
            "timestamp": "2026-10-26 14:20:00",
            "filename": "test.jpg",
            "crop": "Grape",
            "condition": "Black rot",
            "is_healthy": False,
            "predicted_class": "Grape___Black_rot",
            "confidence_percent": 85.0,
            "top_predictions": [],
        }

        html = generate_printable_html_report(scan)
        self.assertIn("@media print", html)
        self.assertIn("size: A4 portrait", html)
        self.assertIn("window.print()", html)


if __name__ == "__main__":
    unittest.main()
