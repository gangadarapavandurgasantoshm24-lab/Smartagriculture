import unittest

from src.disease_data import (
    DISEASE_KNOWLEDGE_BASE,
    filter_diseases,
    get_all_crops,
    get_disease_info,
)
from src.inference import load_class_names


class TestDiseaseData(unittest.TestCase):
    def test_all_38_classes_covered(self):
        class_names = load_class_names()
        self.assertEqual(len(class_names), 38)

        for c_name in class_names:
            self.assertIn(
                c_name,
                DISEASE_KNOWLEDGE_BASE,
                f"Missing knowledge base entry for class: {c_name}",
            )
            info = DISEASE_KNOWLEDGE_BASE[c_name]
            self.assertTrue(info["crop"])
            self.assertTrue(info["common_name"])
            self.assertIn(info["category"], ["Healthy", "Diseased"])
            self.assertTrue(info["symptoms"])
            self.assertTrue(info["ipm_practices"])

    def test_get_all_crops(self):
        crops = get_all_crops()
        self.assertEqual(len(crops), 14)
        self.assertIn("Tomato", crops)
        self.assertIn("Apple", crops)
        self.assertIn("Corn (maize)", crops)

    def test_filter_diseases(self):
        tomato_diseased = filter_diseases(crop="Tomato", category="Diseased")
        self.assertEqual(len(tomato_diseased), 9)

        tomato_healthy = filter_diseases(crop="Tomato", category="Healthy")
        self.assertEqual(len(tomato_healthy), 1)

        search_rust = filter_diseases(search_query="rust")
        self.assertGreaterEqual(len(search_rust), 2)


if __name__ == "__main__":
    unittest.main()
