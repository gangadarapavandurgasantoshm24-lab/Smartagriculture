import json
import re
import pytest

from src.config import CLASS_NAMES_PATH, DISEASE_GUIDANCE_PATH
from src.guidance import get_all_guidance, get_guidance, validate_guidance_against_model


class TestDiseaseGuidance:
    """Test suite for the source-grounded Disease Precautions & Treatment Guidance module.
    
    IMPORTANT AGRONOMIC LIMITATION NOTICE:
    Automated software unit tests verify JSON data structure, schema compliance, deterministic
    38-class mapping to the CNN model, metadata completeness, and regex dosage formats.
    Automated tests CANNOT verify agronomic correctness, bio-efficacy in the field, localized
    soil/weather interactions, or evolving statutory pesticide regulations. Physical field
    inspection by a certified plant pathologist or local Krishi Vigyan Kendra (KVK) is essential.
    """

    @pytest.fixture(scope="module")
    def guidance_catalog(self):
        """Load the complete 38-class guidance catalog."""
        with open(DISEASE_GUIDANCE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture(scope="module")
    def model_classes(self):
        """Load the verified 38 model class names."""
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_class_names_exact_count_and_mapping(self, guidance_catalog, model_classes):
        """Verify that the guidance catalog matches the exact 38 model classes with 0 missing or extra."""
        assert len(model_classes) == 38, "Model classes must be exactly 38"
        assert len(guidance_catalog) == 38, "Guidance catalog must contain exactly 38 classes"

        guidance_keys = list(guidance_catalog.keys())
        assert guidance_keys == model_classes, "Class order and keys must match model class_names.json exactly"

        # Check self-validation function
        val_result = validate_guidance_against_model()
        assert val_result["valid"] is True
        assert val_result["class_count"] == 38
        assert len(val_result["errors"]) == 0

    def test_all_classes_have_complete_required_fields(self, guidance_catalog):
        """Verify that every class contains all mandatory schema fields."""
        required_top_keys = [
            "class_name", "crop", "disease_name", "etiology", "causal_agent",
            "overview", "plain_language_symptoms", "visual_signs",
            "preventive_measures", "immediate_non_chemical_management",
            "sanitation_and_irrigation", "chemical_treatment_status",
            "chemical_guidance", "sources"
        ]
        required_chem_keys = [
            "available", "active_ingredient", "formulation_and_dosage",
            "application_method", "safety_precautions", "re_entry_interval",
            "pre_harvest_interval", "disclaimer"
        ]

        for cname, record in guidance_catalog.items():
            for k in required_top_keys:
                assert k in record, f"Class '{cname}' missing top-level field '{k}'"
                assert record[k] is not None, f"Class '{cname}' has null field '{k}'"

            chem = record["chemical_guidance"]
            for ck in required_chem_keys:
                assert ck in chem, f"Class '{cname}' missing chemical field '{ck}'"

            assert len(record["preventive_measures"]) > 0, f"Class '{cname}' has empty preventive measures"
            assert len(record["immediate_non_chemical_management"]) > 0, f"Class '{cname}' has empty immediate actions"

    def test_healthy_classes_have_no_chemical_treatment(self, guidance_catalog):
        """Verify that all 12 healthy classes strictly prohibit chemical intervention."""
        healthy_classes = [k for k in guidance_catalog.keys() if "healthy" in k]
        assert len(healthy_classes) == 12, "Must identify exactly 12 healthy classes"

        for cname in healthy_classes:
            record = guidance_catalog[cname]
            assert "Inappropriate" in record["chemical_treatment_status"], (
                f"Healthy class '{cname}' must have Inappropriate chemical status"
            )
            chem = record["chemical_guidance"]
            assert chem["available"] is False, f"Healthy class '{cname}' must not have chemical options available"
            assert chem["active_ingredient"] is None, f"Healthy class '{cname}' must have null active ingredient"
            assert chem["formulation_and_dosage"] is None, f"Healthy class '{cname}' must have null dosage"
            assert "inappropriate" in chem["disclaimer"].lower(), (
                f"Healthy class '{cname}' disclaimer must state chemical treatment is inappropriate"
            )

    def test_viral_classes_clarify_incurability(self, guidance_catalog):
        """Verify that viral diseases state that pesticides do not cure viruses."""
        viral_classes = [k for k in guidance_catalog.keys() if "virus" in k.lower()]
        assert len(viral_classes) >= 2, "Must test at least Tomato Mosaic and TYLCV"

        for cname in viral_classes:
            record = guidance_catalog[cname]
            assert "Viral" in record["etiology"]
            combined_text = (
                str(record.get("overview", "")) + " " +
                str(record.get("immediate_non_chemical_management", "")) + " " +
                str(record["chemical_guidance"].get("disclaimer", ""))
            ).lower()
            assert ("not cure" in combined_text or "incurable" in combined_text or "cannot be cured" in combined_text), (
                f"Viral class '{cname}' must clarify that pesticides do not cure viral infections"
            )

    def test_no_unsupported_antibiotic_recommendations(self, guidance_catalog):
        """Verify that no class recommends Streptomycin or Streptocycline on vegetables."""
        for cname, record in guidance_catalog.items():
            chem = record["chemical_guidance"]
            ai = str(chem.get("active_ingredient", "") or "")
            dose = str(chem.get("formulation_and_dosage", "") or "")
            assert "streptomycin" not in ai.lower() and "streptocycline" not in ai.lower(), (
                f"Class '{cname}' must not recommend antibiotics: {ai}"
            )
            assert "streptomycin" not in dose.lower() and "streptocycline" not in dose.lower(), (
                f"Class '{cname}' dosage must not recommend antibiotics: {dose}"
            )

        # Confirm Bacterial Spot classes require expert confirmation
        assert guidance_catalog["Tomato___Bacterial_spot"]["chemical_guidance"]["available"] is False
        assert guidance_catalog["Pepper,_bell___Bacterial_spot"]["chemical_guidance"]["available"] is False
        assert guidance_catalog["Peach___Bacterial_spot"]["chemical_guidance"]["available"] is False

    def test_verified_chemicals_count_and_data_integrity(self, guidance_catalog):
        """Verify that exactly 8 chemical classes and 2 vector-management classes are available with verified data."""
        available_classes = [
            cname for cname, rec in guidance_catalog.items()
            if rec["chemical_guidance"]["available"]
        ]
        assert len(available_classes) == 10, f"Expected exactly 10 available classes (8 chem + 2 vector), got {len(available_classes)}"

        dosage_regex = re.compile(r"(\d+(\.\d+)?\s*(g|ml|kg|l|%|WP|EC|SC|WG|SL|litre|acre|ha))", re.IGNORECASE)

        for cname in available_classes:
            record = guidance_catalog[cname]
            chem = record["chemical_guidance"]
            ai = chem["active_ingredient"]
            assert ai and len(ai) > 3, f"Class '{cname}' has empty active ingredient"

            formulation_dosage = chem["formulation_and_dosage"]
            assert formulation_dosage and len(formulation_dosage) > 5, (
                f"Class '{cname}' has missing formulation/dosage"
            )
            assert dosage_regex.search(formulation_dosage), (
                f"Class '{cname}' dosage must include valid units: '{formulation_dosage}'"
            )

            # Safety intervals
            assert chem["pre_harvest_interval"] is not None and "N/A" not in chem["pre_harvest_interval"], (
                f"Class '{cname}' has unverified PHI"
            )
            assert chem["re_entry_interval"] is not None and "N/A" not in chem["re_entry_interval"], (
                f"Class '{cname}' has unverified REI"
            )

            # PPE requirement
            sp = chem["safety_precautions"].lower()
            assert "glove" in sp or "mask" in sp or "protective" in sp, (
                f"Class '{cname}' safety precautions must mention personal protective equipment"
            )

    def test_unverified_classes_have_suppressed_dosages_and_mandatory_notice(self, guidance_catalog):
        """Verify that all 28 classes with available=False have null dosages and include the mandatory KVK notice."""
        mandatory_notice = "A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer."

        unverified_count = 0
        for cname, record in guidance_catalog.items():
            chem = record["chemical_guidance"]
            if not chem["available"]:
                unverified_count += 1
                assert chem["active_ingredient"] is None, (
                    f"Class '{cname}' has available=False but non-null active ingredient: {chem['active_ingredient']}"
                )
                assert chem["formulation_and_dosage"] is None, (
                    f"Class '{cname}' has available=False but non-null dosage: {chem['formulation_and_dosage']}"
                )

                # For diseased classes without verified chemicals, verify the mandatory notice is present
                if "healthy" not in cname:
                    combined_disclaimer = (
                        str(chem.get("disclaimer", "")) + " " +
                        str(record.get("overview", ""))
                    )
                    assert mandatory_notice in chem.get("disclaimer", "") or "inappropriate" in chem.get("disclaimer", "").lower(), (
                        f"Class '{cname}' must display the mandatory KVK notice or explicit Inappropriate disclaimer"
                    )

        assert unverified_count == 28, f"Expected exactly 28 unverified/inappropriate classes, got {unverified_count}"

    def test_sources_and_references_integrity(self, guidance_catalog):
        """Verify that every class has valid, trustworthy source citations with valid official URLs."""
        for cname, record in guidance_catalog.items():
            sources = record["sources"]
            assert len(sources) >= 1, f"Class '{cname}' must have at least one source citation"

            for src in sources:
                assert "title" in src and len(src["title"]) > 0, f"Class '{cname}' source missing title"
                assert "official_url" in src and src["official_url"].startswith("http"), (
                    f"Class '{cname}' source URL must be a valid HTTP/HTTPS link: {src.get('official_url')}"
                )
                assert "source_type" in src and len(src["source_type"]) > 0, f"Class '{cname}' source missing type"
                assert "last_verified" in src and len(src["last_verified"]) > 0, f"Class '{cname}' source missing last_verified date"
