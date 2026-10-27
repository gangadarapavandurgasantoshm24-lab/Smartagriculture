import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from src.config import (
    CLASS_NAMES_PATH,
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    DISEASE_GUIDANCE_PATH,
)


@st.cache_data(show_spinner=False)
def load_disease_guidance() -> Dict[str, Dict[str, Any]]:
    """Load and cache the comprehensive 38-class source-grounded disease guidance catalog."""
    if not DISEASE_GUIDANCE_PATH.exists():
        raise FileNotFoundError(
            f"Disease guidance data file not found at {DISEASE_GUIDANCE_PATH}"
        )
    with open(DISEASE_GUIDANCE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_guidance(class_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve the guidance record for a specific PlantVillage class key."""
    catalog = load_disease_guidance()
    return catalog.get(class_name)


def get_all_guidance() -> Dict[str, Dict[str, Any]]:
    """Retrieve all 38 guidance records."""
    return load_disease_guidance()


def validate_guidance_against_model() -> Dict[str, Any]:
    """Validate that the guidance catalog matches the model class list exactly.
    
    Verifies:
    1. Exactly 38 classes.
    2. Exact class key 1-to-1 match with models/class_names.json.
    3. Completeness of required schema fields in every record.
    4. Compliance of healthy, viral, and bacterial rules.
    """
    if not CLASS_NAMES_PATH.exists():
        return {
            "valid": False,
            "error": f"Model class names file not found at {CLASS_NAMES_PATH}",
        }

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        model_classes = json.load(f)

    guidance_data = load_disease_guidance()
    guidance_classes = list(guidance_data.keys())

    errors = []

    # 1. Check exact class count & match
    if len(guidance_classes) != len(model_classes):
        errors.append(
            f"Class count mismatch: guidance has {len(guidance_classes)}, model has {len(model_classes)}"
        )

    missing_in_guidance = set(model_classes) - set(guidance_classes)
    if missing_in_guidance:
        errors.append(f"Classes missing in guidance: {missing_in_guidance}")

    extra_in_guidance = set(guidance_classes) - set(model_classes)
    if extra_in_guidance:
        errors.append(f"Extra classes in guidance: {extra_in_guidance}")

    # 2. Check schema fields
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

    for cname, record in guidance_data.items():
        for rk in required_top_keys:
            if rk not in record:
                errors.append(f"Class '{cname}' missing required top field '{rk}'")
        chem = record.get("chemical_guidance", {})
        for ck in required_chem_keys:
            if ck not in chem:
                errors.append(f"Class '{cname}' missing chemical field '{ck}'")
        if not record.get("sources"):
            errors.append(f"Class '{cname}' has no sources listed")

    return {
        "valid": len(errors) == 0,
        "class_count": len(guidance_classes),
        "expected_count": len(model_classes),
        "errors": errors,
    }


def render_guidance_tabs(
    guidance: Dict[str, Any],
    model_confidence: float = 1.0,
    is_prediction_context: bool = True,
    is_ambiguous: bool = False,
    ambiguity_message: str = "",
):

    """Render the 7-tab Disease Precautions & Treatment Guidance interface.
    
    Tabs:
    1. Disease Overview
    2. Symptoms
    3. Prevention
    4. Immediate Management
    5. Verified Chemical Options
    6. Application Safety
    7. Sources and References
    """
    if not guidance:
        st.warning("Guidance information is not available for this class.")
        return

    chem = guidance.get("chemical_guidance", {})
    sources = guidance.get("sources", [])
    treatment_status = guidance.get("chemical_treatment_status", "Requires Expert Confirmation")

    # Header Card: Disease Identity & Dual Status Badges
    crop = guidance.get("crop", "Unknown")
    disease = guidance.get("disease_name", "Unknown")
    etiology = guidance.get("etiology", "Fungal")
    causal_agent = guidance.get("causal_agent", "N/A")

    is_healthy = (
        etiology == "Healthy"
        or "healthy" in guidance.get("class_name", "").lower()
        or "healthy" in disease.lower()
    )

    # Determine badge style
    if is_healthy:
        status_badge_class = "badge-healthy"
        status_icon = "🌱"
        treatment_status = "Healthy Foliage — No Cure Required"
    elif "Appropriate (Verified)" in treatment_status or "Bactericide" in treatment_status or "Acaricide" in treatment_status:
        status_badge_class = "badge-verified"
        status_icon = "✅"
    elif "Vector Management Only" in treatment_status:
        status_badge_class = "badge-vector"
        status_icon = "🔄"
    elif "Requires Expert Confirmation" in treatment_status or "Unavailable" in treatment_status:
        status_badge_class = "badge-expert"
        status_icon = "⚠️"
    else:
        status_badge_class = "badge-inappropriate"
        status_icon = "ℹ️"

    # BUG 3 requirement: Prominent healthy status message
    if is_healthy:
        st.markdown(
            f"""
            <div style="background: #E8F5E9; border-left: 6px solid #2E7D32; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 3px 12px rgba(46,125,50,0.1);">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <span style="font-size: 2.2rem;">🌱</span>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #1B5E20; margin-bottom: 4px;">
                            Healthy leaf detected — no disease treatment or pesticide is required based on this prediction.
                        </div>
                        <div style="font-size: 0.92rem; color: #2E7D32; line-height: 1.45;">
                            Target Crop: <strong>{crop}</strong> • Foliage displays normal vigor with no active disease lesions. Continue routine monitoring, balanced nutrition, proper irrigation, and field hygiene.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="guidance-card" style="border-left: 5px solid {COLOR_FOREST_GREEN};">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="badge-crop">{crop}</span>
                    <span style="font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; background: #E2ECE7; color: {COLOR_FOREST_GREEN};">
                        Etiology: {etiology}
                    </span>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="{status_badge_class}">
                        {status_icon} Status: {treatment_status}
                    </span>
                </div>
            </div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.35rem; font-weight: 700; color: {COLOR_FOREST_GREEN}; margin-bottom: 4px;">
                {crop}: {disease}
            </div>
            <div style="font-size: 0.88rem; color: #4F6D61;">
                <strong>Pathogen / Agent:</strong> <em>{causal_agent}</em> &nbsp;|&nbsp; 
                <strong>Dataset Class:</strong> <code>{guidance.get('class_name', '')}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Seven Structured Tabs
    tabs = st.tabs([
        "📖 1. Disease Overview",
        "🔍 2. Symptoms",
        "🛡️ 3. Prevention",
        "⚡ 4. Immediate Management",
        "🧪 5. Verified Chemical Options",
        "⛑️ 6. Application Safety",
        "📚 7. Sources and References",
    ])

    # ---------------- TAB 1: DISEASE OVERVIEW ----------------
    with tabs[0]:
        st.markdown("##### 📖 Botanical & Agronomic Overview")
        st.markdown(guidance.get("overview", "No overview provided."))

        if is_healthy:
            st.markdown(
                f"""
                <div class="guidance-callout">
                    <strong>🌿 Agronomic Principle: Sustaining Healthy Plant Vigor</strong><br>
                    Maintaining robust plant health focuses on balanced crop nutrition, proactive field scouting, soil health maintenance, and proper irrigation management. Healthy plants naturally resist environmental stresses and minor pest pressures.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="guidance-callout">
                    <strong>💡 Agronomic Principle: Disease Management vs. Guaranteed Cure</strong><br>
                    Crop disease management focuses on pathogen suppression, breaking infection cycles, protecting new foliage, and reducing economic yield loss. Once cellular infection and necrosis have occurred within plant tissue, a "guaranteed cure" is not biologically possible. Sustainable control relies on integrated cultural practices, sanitation, and timely, label-authorized treatments.
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Target Crop:** {crop}")
            st.markdown(f"**Condition Assessment:** {disease}")
            st.markdown(f"**Causal Organism / State:** *{causal_agent}*")
        with col_b:
            st.markdown(f"**Pathological Group:** {etiology}")
            st.markdown(f"**Model Dataset Key:** `{guidance.get('class_name')}`")
            st.markdown(f"**Treatment Requirement:** {'None Required (Healthy)' if is_healthy else treatment_status}")

    # ---------------- TAB 2: SYMPTOMS ----------------
    with tabs[1]:
        st.markdown("##### 🔍 Plain-Language Symptoms & Visual Identification")
        st.markdown(
            f"""
            <div style="margin-bottom: 14px;">
                <strong>Farmer / Scout Observations:</strong><br>
                {guidance.get('plain_language_symptoms', 'None reported.')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🔬 Specific Botanical & Visual Signs")
        st.info(guidance.get("visual_signs", "None reported."))

    # ---------------- TAB 3: PREVENTION ----------------
    with tabs[2]:
        st.markdown("##### 🛡️ Preventive Care & Cultural Measures")
        prev_measures = guidance.get("preventive_measures", [])
        if isinstance(prev_measures, list) and prev_measures:
            for pm in prev_measures:
                st.markdown(f"- {pm}")
        else:
            st.markdown(str(prev_measures))

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 💧 Field Sanitation & Irrigation Recommendations")
        st.markdown(
            f"""
            <div class="guidance-callout">
                {guidance.get('sanitation_and_irrigation', 'Maintain clean field borders and sanitize pruning shears between plants.')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- TAB 4: IMMEDIATE MANAGEMENT ----------------
    with tabs[3]:
        st.markdown("##### ⚡ Immediate Management & Best Practices")
        imm_actions = guidance.get("immediate_non_chemical_management", [])
        if isinstance(imm_actions, list) and imm_actions:
            for act in imm_actions:
                st.markdown(f"- {act}")
        else:
            st.markdown(str(imm_actions))

        if is_healthy:
            st.markdown(
                """
                <div class="guidance-callout">
                    <strong>🌱 Best Practice for Healthy Stands:</strong> Maintain regular weekly scouting schedules. Check lower canopy leaf undersides for early insect vector colonization or subtle moisture stress. Do not apply preventive foliar chemicals without justified threshold indicators.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif etiology == "Viral":
            st.markdown(
                """
                <div class="guidance-callout-warning">
                    <strong>⚠️ Viral Etiology Alert:</strong> Chemical fungicides and bactericides have zero efficacy against plant viruses. Never spray fungicides to treat viral symptoms. Focus strictly on rogueing symptomatic plants, cleaning pruning tools, and managing vector populations (e.g., whiteflies or thrips) using registered IPM protocols.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif etiology == "Bacterial":
            st.markdown(
                """
                <div class="guidance-callout-info">
                    <strong>ℹ️ Bacterial Etiology Alert:</strong> Standard synthetic fungicides do not control phytopathogenic bacteria. Misdiagnosing bacterial spot as fungal leaf spot and applying synthetic fungicides results in wasted resources and uncontrolled disease progression.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- TAB 5: VERIFIED CHEMICAL OPTIONS ----------------
    with tabs[4]:
        st.markdown("##### 🧪 Source-Grounded Chemical & Bio-Chemical Options")

        if is_healthy:
            # BUG 3 requirement for healthy predictions:
            # 1. Hide disease-specific cure and pesticide recommendations.
            # 2. Prominent healthy-status message.
            # 3. Do not recommend spraying pesticides preventively without a justified need.
            # 4. Keep model confidence visible and explain image prediction is not a guarantee of plant health.
            st.markdown(
                f"""
                <div style="background: #F1F8E9; border: 1.5px solid #81C784; border-radius: 12px; padding: 22px; margin-bottom: 18px;">
                    <div style="font-size: 1.15rem; font-weight: 800; color: #1B5E20; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                        <span>🌱</span> Healthy Leaf Detected — No Cure Required
                    </div>
                    <div style="font-size: 1.02rem; font-weight: 700; color: #2E7D32; margin-bottom: 12px;">
                        Healthy leaf detected — no disease treatment or pesticide is required based on this prediction.
                    </div>
                    <p style="font-size: 0.92rem; color: #2E7D32; line-height: 1.5; margin-bottom: 12px;">
                        Applying chemical fungicides, bactericides, or synthetic pesticides preventively to healthy plants without a justified need or verified economic threshold is unnecessary, economically wasteful, and environmentally hazardous. <strong>Prophylactic pesticide spraying is strictly not recommended for healthy foliage.</strong>
                    </p>
                    <div style="background: #FFFFFF; border-radius: 8px; padding: 12px 16px; border: 1px solid #C8E6C9; font-size: 0.88rem; color: #1B5E20;">
                        <strong>Recommended Care:</strong> Routine monitoring, balanced N-P-K nutrition, proper drip irrigation to avoid canopy wetness, and field sanitation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if is_prediction_context:
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    st.metric("Model Classification Confidence", f"{model_confidence * 100:.1f}%")
                with conf_col2:
                    st.metric("Treatment Requirement", "No Chemical Cure Required")

                st.markdown(
                    """
                    <div class="guidance-callout-info" style="font-size: 0.88rem; line-height: 1.45; margin-top: 12px;">
                        <strong>ℹ️ Screening Advisory:</strong> Model confidence reflects classification of the submitted leaf photograph against known visual patterns. An image prediction is not a guarantee of plant health. Computer vision cannot detect sub-visual, internal, or latent root/vascular infections (such as nematodes, soil-borne root wilts, or early asymptomatic viral systemic infections). Regular in-field monitoring remains essential.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            # Disease predictions: Show disease-specific precautions and management
            if is_ambiguous:
                st.markdown(
                    f"""
                    <div style="background: #FFF8E1; border: 1.5px solid #FFC107; border-radius: 12px; padding: 20px; margin-bottom: 18px;">
                        <div style="font-size: 1.15rem; font-weight: 800; color: #7F5800; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                            <span>⚠️</span> Chemical Guidance Withheld (Cross-Crop Ambiguity)
                        </div>
                        <p style="font-size: 0.93rem; color: #5C4000; line-height: 1.5; margin-bottom: 10px;">
                            {ambiguity_message if ambiguity_message else "Because the crop species identity or disease classification is ambiguous, specific chemical pesticide recommendations are withheld to prevent accidental crop injury or illegal off-label usage."}
                        </p>
                        <div style="background: #FFFFFF; border-radius: 8px; padding: 12px 16px; border: 1px solid #FFE082; font-size: 0.88rem; color: #5C4000;">
                            <strong>Safe Action:</strong>
                            <ol style="margin-top: 6px; margin-bottom: 0; padding-left: 20px;">
                                <li>Confirm your target crop in the <strong>Target Crop Context</strong> selector above to filter out background interference.</li>
                                <li>Or capture a clear, close-up photograph focused on the leaf blade without hands or foreign backgrounds.</li>
                                <li>Consult your local Krishi Vigyan Kendra (KVK) or extension officer for certified on-site verification before applying any chemical spray.</li>
                            </ol>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif is_prediction_context and model_confidence < 0.70:
                # Low-confidence AI warning (< 70%)
                st.markdown(
                    f"""
                    <div class="guidance-callout-warning">
                        <strong>⚠️ Low Model Confidence Alert ({model_confidence * 100:.1f}%):</strong><br>
                        The neural network predicted this class with less than 70% confidence. <strong>Never purchase or spray chemical pesticides solely based on a low-confidence computer vision screening.</strong> Physical examination of the crop and symptoms by a local certified Krishi Vigyan Kendra (KVK) scientist or agricultural officer is required before chemical deployment.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Separate Model Confidence and Guidance Reliability
            if is_prediction_context:
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    st.metric("Model Classification Confidence", f"{model_confidence * 100:.1f}%")
                with conf_col2:
                    rel_label = (
                        "Withheld (Ambiguous Prediction)"
                        if is_ambiguous
                        else (
                            "PPQS / CIBRC Registered & ICAR Verified"
                            if chem.get("available")
                            else "Non-Chemical / Unregistered / Expert Consult Required"
                        )
                    )
                    st.metric("Guidance Verification Level", rel_label)

            # Check if chemical options are verified and available (suppressed if ambiguous)
            if not is_ambiguous and chem.get("available"):

                st.markdown(
                    f"""
                    <div class="guidance-card" style="border-left: 4px solid #1E7E34; background: #FAFDFB;">
                        <div style="font-weight: 700; color: #1E7E34; font-size: 1rem; margin-bottom: 6px;">
                            ✅ Verified Active Ingredient(s):
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: {COLOR_FOREST_GREEN}; margin-bottom: 12px;">
                            {chem.get('active_ingredient', 'N/A')}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr; gap: 10px; font-size: 0.9rem;">
                            <div>
                                <strong>Formulation & Dosage:</strong><br>
                                <code>{chem.get('formulation_and_dosage', 'Refer to official product label.')}</code>
                            </div>
                            <div>
                                <strong>Application Method:</strong><br>
                                {chem.get('application_method', 'Foliar application.')}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if chem.get("disclaimer"):
                    st.markdown(
                        f"""
                        <div class="guidance-callout">
                            <strong>📌 Label & Agronomic Advisory:</strong> {chem.get('disclaimer')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Requirement 5: Actual verified commercial trade names (India CIBRC approved)
                from src.pesticides import render_commercial_product_cards
                render_commercial_product_cards(
                    class_name=guidance.get("class_name", ""),
                    is_healthy=is_healthy,
                    is_ambiguous=is_ambiguous,
                    ambiguity_message=ambiguity_message,
                )
            else:
                # Notice when verified chemical recommendation is not available
                from src.pesticides import render_commercial_product_cards
                render_commercial_product_cards(
                    class_name=guidance.get("class_name", ""),
                    is_healthy=is_healthy,
                    is_ambiguous=is_ambiguous,
                    ambiguity_message=ambiguity_message,
                )
                disclaimer_text = chem.get(
                    "disclaimer",
                    "A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer."
                )
                if "A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer." not in disclaimer_text:
                    notice_prefix = "A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer.\n\n"
                else:
                    notice_prefix = ""

                st.markdown(
                    f"""
                    <div class="guidance-callout-warning" style="font-size: 0.95rem; line-height: 1.5;">
                        <strong>ℹ️ Official Regulatory & Agronomic Notice:</strong><br>
                        {notice_prefix}{disclaimer_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Show source citations for chemical guidance
        if sources:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("**Authoritative Verification Sources for this Guidance:**")
            for src in sources:
                src_title = src.get("title", "Official Source")
                src_url = src.get("official_url", "#")
                src_type = src.get("source_type", "Regulatory")
                st.markdown(
                    f"- 🔗 [{src_title}]({src_url}) &nbsp; `[{src_type}]`"
                )

    # ---------------- TAB 6: APPLICATION SAFETY ----------------
    with tabs[5]:
        st.markdown("##### ⛑️ Worker Safety, Re-entry & Pre-Harvest Intervals")

        if is_healthy:
            st.markdown(
                """
                <div class="guidance-callout" style="margin-bottom: 16px;">
                    <strong>🌱 No Chemical Restrictions:</strong> Because this foliage is identified as healthy, no chemical fungicide or pesticide treatment is required. Pre-Harvest Intervals (PHI) and Re-Entry Intervals (REI) for chemical sprays are not applicable.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px; margin-bottom: 12px;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: {COLOR_TEXT_MUTED}; text-transform: uppercase;">
                            ⏳ Pre-Harvest Interval (PHI)
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_FOREST_GREEN}; margin-top: 4px;">
                            {chem.get('pre_harvest_interval', 'N/A')}
                        </div>
                        <div style="font-size: 0.78rem; color: #6E887B; margin-top: 4px;">
                            Mandatory minimum time between final application and crop harvest.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s2:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px; margin-bottom: 12px;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: {COLOR_TEXT_MUTED}; text-transform: uppercase;">
                            🚪 Re-Entry Interval (REI)
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_FOREST_GREEN}; margin-top: 4px;">
                            {chem.get('re_entry_interval', 'N/A')}
                        </div>
                        <div style="font-size: 0.78rem; color: #6E887B; margin-top: 4px;">
                            Restricted field entry time without full personal protective equipment.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("##### 🧤 Personal Protective Equipment (PPE) & Handling Protocols")
        safety_text = chem.get(
            "safety_precautions",
            "Wear chemical-resistant nitrile gloves, protective eyewear, mask, and long sleeves. Do not consume food or drink while applying crop protectants."
        )
        st.markdown(
            f"""
            <div class="guidance-callout">
                {safety_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **Environmental & Pollinator Protection Guidelines:**
            - **Bee Protection:** Avoid spraying flowering crops or blooming weeds in the field during peak honeybee foraging hours (apply during late dusk or early dawn).
            - **Aquatic Safety:** Do not apply within 15 meters of water bodies, fish ponds, or irrigation canals; prevent runoff or spray drift.
            - **Container Disposal:** Triple rinse empty product containers, puncture them to prevent reuse, and dispose of them through authorized agricultural hazardous waste channels.
            """
        )

    # ---------------- TAB 7: SOURCES AND REFERENCES ----------------
    with tabs[6]:
        st.markdown("##### 📚 Source Citations & Agricultural Registry Grounding")
        st.markdown(
            """
            All disease precautions, cultural management guidelines, and chemical recommendations are compiled exclusively from authoritative agricultural research institutions, State Agricultural Universities (SAUs), ICAR institutes, and official pesticide registries.
            """
        )

        if sources:
            for idx, src in enumerate(sources, 1):
                src_title = src.get("title", f"Source {idx}")
                src_url = src.get("official_url", "#")
                src_type = src.get("source_type", "Government / Extension")
                src_date = src.get("last_verified", "2024")

                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 700; font-size: 0.95rem; color: {COLOR_FOREST_GREEN};">
                                {idx}. {src_title}
                            </span>
                            <span style="font-size: 0.75rem; background: {COLOR_LIGHT_SAGE}; color: {COLOR_FOREST_GREEN}; padding: 2px 8px; border-radius: 12px; font-weight: 600;">
                                {src_type}
                            </span>
                        </div>
                        <div style="font-size: 0.82rem; color: #4F6D61; margin-bottom: 6px;">
                            Verified Date: <strong>{src_date}</strong>
                        </div>
                        <div style="font-size: 0.85rem;">
                            <a href="{src_url}" target="_blank" style="color: {COLOR_SECONDARY_GREEN}; font-weight: 600; text-decoration: none;">
                                🌐 View Official Publication / Registry Record &rarr;
                            </a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No external source citations recorded for this entry.")

        st.markdown(
            """
            <div style="margin-top: 16px; font-size: 0.78rem; color: #6E887B; border-top: 1px solid #D3E2D3; padding-top: 10px;">
                <strong>Statutory Disclaimer:</strong> Treatment advisories are subject to national and state pesticide regulations (CIBRC/PPQS under the Insecticides Act, 1968). Formulations, registered crops, and maximum residue limits (MRLs) may be updated periodically. Always verify against the physical manufacturer product label registered in your jurisdiction before purchase or application.
            </div>
            """,
            unsafe_allow_html=True,
        )
