from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COLOR_ACCENT_GREEN,
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
    TEST_IMAGES_DIR,
)
from src.session_history import add_session_scan
from src.disease_data import get_disease_info
from src.gradcam import apply_gradcam_overlay, generate_gradcam_heatmap
from src.guidance import get_guidance, render_guidance_tabs
from src.inference import SUPPORTED_CROPS, predict_leaf
from src.styles import render_hero


def render_detection():
    """Render the Disease Detection workflow page."""
    render_hero(
        title="Plant Leaf Disease Diagnostics",
        subtitle="Upload a leaf photograph or select a validation sample. The CNN inference pipeline computes class probabilities and live Grad-CAM visual attention.",
        badge="REAL-TIME INFERENCE",
    )

    # State tracking for image input
    current_image_bytes = None
    image_name = "custom_leaf.jpg"

    col_input, col_meta = st.columns([3, 2])

    with col_input:
        st.markdown("#### 📤 Input Plant Leaf Image")
        input_method = st.radio(
            "Select Image Source:",
            [
                "Upload Your Own File",
                "Benchmark Dataset Samples",
                "Field Failure Test Cases (Apple / Maize)",
            ],
            horizontal=True,
        )

        if input_method == "Upload Your Own File":
            uploaded_file = st.file_uploader(
                "Drag and drop or browse a leaf image (.jpg, .jpeg, .png):",
                type=["jpg", "jpeg", "png"],
                help="Image will be automatically normalized and resized to 256x256 RGB for the CNN.",
            )
            if uploaded_file is not None:
                current_image_bytes = uploaded_file.read()
                image_name = uploaded_file.name

        elif input_method == "Benchmark Dataset Samples":
            # Preset Sample Selector from PlantVillage test images
            if TEST_IMAGES_DIR.exists():
                sample_files = sorted(
                    list(TEST_IMAGES_DIR.glob("*.jpg"))
                    + list(TEST_IMAGES_DIR.glob("*.JPG"))
                    + list(TEST_IMAGES_DIR.glob("*.png"))
                )
                if sample_files:
                    sample_options = {f.name: f for f in sample_files}
                    default_idx = 0

                    if (
                        "selected_sample_image" in st.session_state
                        and st.session_state["selected_sample_image"]
                    ):
                        passed_path = Path(st.session_state["selected_sample_image"])
                        if passed_path.name in sample_options:
                            default_idx = list(sample_options.keys()).index(
                                passed_path.name
                            )

                    chosen_sample_name = st.selectbox(
                        "Select a verified dataset sample image:",
                        list(sample_options.keys()),
                        index=default_idx,
                    )
                    chosen_file_path = sample_options[chosen_sample_name]
                    with open(chosen_file_path, "rb") as f:
                        current_image_bytes = f.read()
                    image_name = chosen_sample_name

        else:
            # Field Failure Test Cases Reproducer
            failure_dir = Path("data/user_failure_cases")
            if failure_dir.exists():
                failure_files = sorted(list(failure_dir.glob("*.jpg")) + list(failure_dir.glob("*.png")))
                if failure_files:
                    fail_options = {f.name: f for f in failure_files}
                    chosen_fail_name = st.selectbox(
                        "Select a reported field failure image:",
                        list(fail_options.keys()),
                        index=0,
                        help="Exact failure cases reported by user (e.g. apple leaf with brown lesions, maize leaf with border)."
                    )
                    chosen_fail_path = fail_options[chosen_fail_name]
                    with open(chosen_fail_path, "rb") as f:
                        current_image_bytes = f.read()
                    image_name = chosen_fail_name

        # Diagnostic Pre-Filtering Controls
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            crop_options = ["All Crops (Auto-Detect)"] + SUPPORTED_CROPS
            selected_crop_context = st.selectbox(
                "🌾 Confirm Target Crop (Optional):",
                crop_options,
                index=0,
                help="If you know the crop being inspected (e.g. Apple orchard, Maize field), select it to filter out foreign background or skin-tone noise from field photography."
            )
            confirmed_crop = None if selected_crop_context == "All Crops (Auto-Detect)" else selected_crop_context

        with ctrl_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            center_crop_enabled = st.checkbox(
                "🎯 Center Leaf Focus (Square Crop)",
                value=False,
                help="Crops the square center of the photo before resizing to 256x256. Highly recommended for non-square smartphone photos or images with letterbox/white borders."
            )

    with col_meta:
        st.markdown("#### ⚙️ Pipeline Specification")
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 18px; font-size: 0.88rem; color: #3E544A;">
                <p><strong>Input Dimension:</strong> <code>256 × 256 × 3 (RGB)</code></p>
                <p><strong>Pixel Normalization:</strong> <code>float32 / 255.0</code> ([0.0, 1.0])</p>
                <p><strong>Model Engine:</strong> <code>smart_agriculture_cnn</code></p>
                <p><strong>Activation Layer:</strong> <code>conv2d_3 (128 filters)</code></p>
                <p><strong>Classification:</strong> 38-way Softmax Distribution</p>
                <p><strong>Crop Disambiguation:</strong> <code>{'Active (' + str(confirmed_crop) + ')' if confirmed_crop else 'Auto-Detect (Unconstrained)'}</code></p>
                <p><strong>Leaf Framing:</strong> <code>{'Center Square Crop' if center_crop_enabled else 'Full Image Direct Resize'}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Stop here if no image has been selected
    if current_image_bytes is None:
        st.info("👆 Please upload an image or select a sample above to run diagnosis.")
        return

    st.markdown("---")

    # --- Run Inference ---
    with st.spinner("Processing leaf image through CNN and calculating Grad-CAM..."):
        try:
            results = predict_leaf(
                current_image_bytes,
                top_k=3,
                confirmed_crop=confirmed_crop,
                center_crop=center_crop_enabled,
            )
            # Generate Grad-CAM targeting conv2d_3
            heatmap = generate_gradcam_heatmap(
                results["input_tensor"], target_class_index=results["class_index"]
            )
            _, overlay_img = apply_gradcam_overlay(
                results["display_image"], heatmap, alpha=0.45
            )

            # Store in session state for AI Explainability page deep-dive
            st.session_state["last_diagnostic_result"] = results
            st.session_state["last_heatmap"] = heatmap
            st.session_state["last_overlay"] = overlay_img
            st.session_state["last_image_name"] = image_name

        except Exception as e:
            st.error(f"Inference Pipeline Error: {e}")
            return

    # --- Display Diagnostic Output ---
    st.markdown("### 📋 Diagnostic Assessment")

    out_col1, out_col2 = st.columns([1, 1])

    with out_col1:
        st.markdown("##### 🍃 Analyzed Leaf & Model Attention")
        img_tab1, img_tab2 = st.tabs(["Original (256x256)", "Grad-CAM (conv2d_3)"])
        with img_tab1:
            st.image(
                results["display_image"],
                caption=f"Input: {image_name}{' [Center Cropped]' if results['center_crop_applied'] else ''}",
                use_container_width=True,
            )
        with img_tab2:
            st.image(
                overlay_img,
                caption="Grad-CAM heat activation on conv2d_3",
                use_container_width=True,
            )

    with out_col2:
        # Diagnostic Classification Card
        badge_class = (
            "badge-healthy" if results["is_healthy"] else "badge-diseased"
        )
        status_text = "HEALTHY" if results["is_healthy"] else "DISEASED"

        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 14px; padding: 22px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="badge-crop" style="font-size: 0.85rem;">{results['crop']}</span>
                    <span class="{badge_class}" style="font-size: 0.85rem;">{status_text}</span>
                </div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 800; color: {COLOR_FOREST_GREEN}; margin-bottom: 6px;">
                    {results['condition']}
                </div>
                <div style="font-size: 0.9rem; color: #556E63; margin-bottom: 14px;">
                    Class Identifier: <code>{results['predicted_class']}</code>
                </div>
                <div style="display: flex; align-items: baseline; gap: 8px;">
                    <span style="font-size: 1.8rem; font-weight: 800; color: {COLOR_SECONDARY_GREEN};">{results['confidence_percent']:.2f}%</span>
                    <span style="font-size: 0.85rem; color: #6E887B;">{'within-crop confidence' if results['confirmed_crop'] else 'prediction confidence'}</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #6E887B; margin-top: -6px; margin-bottom: 14px; font-style: italic;">
                ⚠️ Statistical model probability across trained patterns — not proof of biological certainty.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Context applied banner or cross-crop ambiguity warning
        if results["confirmed_crop"]:
            st.markdown(
                f"""
                <div style="background: #E8F5E9; border-left: 4px solid #2E7D32; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.88rem; color: #1B5E20;">
                    🎯 <strong>Target Crop Context Active: {results['confirmed_crop']}</strong><br>
                    Condition ranking is constrained to {results['confirmed_crop']} foliage ({results['confidence_percent']:.1f}% relative confidence).
                    <div style="font-size: 0.78rem; color: #2E7D32; margin-top: 4px;">
                        Raw unconstrained model prediction was <code>{results['raw_predicted_class']}</code> ({results['raw_confidence_percent']:.1f}%), activated by outdoor lighting, human hand, or background noise.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif results["cross_crop_ambiguous"]:
            crops_str = " vs ".join(results["conflicting_crops"][:3])
            st.warning(
                f"⚠️ **Cross-Species Ambiguity Detected ({crops_str}):**\n\n"
                f"The neural network is dividing probability across distinct crop species: "
                + ", ".join([f"**{p['crop']}** ({p['condition']} {p['confidence_percent']:.1f}%)" for p in results["raw_top_predictions"][:2]])
                + ".\n\n"
                "Because this CNN was trained on uniform laboratory leaves on flat boards (PlantVillage), field photos with hands or natural backgrounds can activate multiple crop filters.\n\n"
                "👉 **Action Required:** Please select your target crop in the **Confirm Target Crop** selector above to filter out background noise, or upload a closer photo focused strictly on the leaf blade."
            )

        # Top Candidates Bar Chart
        chart_title = (
            f"📊 Top Candidates within {results['confirmed_crop']}"
            if results["confirmed_crop"]
            else "📊 Top 3 Candidate Probabilities"
        )
        st.markdown(f"##### {chart_title}")
        top_preds = results["top_predictions"]

        chart_labels = [p["condition"] for p in reversed(top_preds)]
        chart_values = [p["confidence_percent"] for p in reversed(top_preds)]
        chart_colors = [
            COLOR_FOREST_GREEN if i == len(top_preds) - 1 else "#A3C4B6"
            for i in range(len(top_preds))
        ]

        fig = go.Figure(
            go.Bar(
                x=chart_values,
                y=chart_labels,
                orientation="h",
                marker_color=chart_colors,
                text=[f"{v:.1f}%" for v in chart_values],
                textposition="auto",
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=20, t=10, b=10),
            height=180,
            xaxis=dict(range=[0, 100], title="Confidence (%)", showgrid=True),
            yaxis=dict(title=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Uncertainty alert if unconstrained top candidate confidence is low (< 70%)
        if not results["confirmed_crop"] and results["confidence"] < 0.70:
            st.warning(
                f"⚠️ **Low Confidence / Uncertain Prediction ({results['confidence_percent']:.1f}%):** "
                "The model classification confidence is below 70%. Predictions on field-collected imagery, "
                "non-standard camera angles, or atypical leaf lesions can be ambiguous. Please evaluate the top candidates above "
                "and consult a local agricultural officer (KVK) for physical verification."
            )

    # --- Source-Grounded Disease Precautions & Treatment Guidance (7 Tabs) ---
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🌿 Disease Precautions & Treatment Guidance")

    # Prominent healthy-status message when a healthy leaf is detected
    if results["is_healthy"]:
        st.success(
            "🌱 **Healthy leaf detected — no disease treatment or pesticide is required based on this prediction.**"
        )

    # Determine if chemical recommendations should be safely withheld
    guidance_ambiguous = (results["cross_crop_ambiguous"] and not results["confirmed_crop"]) or (
        not results["confirmed_crop"] and results["confidence"] < 0.70
    )
    ambiguity_msg = (
        f"Predictions are currently dividing probability across multiple crop species ({', '.join(results['conflicting_crops'])}). "
        "Specific chemical recommendations are withheld to prevent accidental crop injury or illegal off-label spraying. "
        "Please select your target crop in 'Confirm Target Crop' above to unlock verified guidance."
        if results["cross_crop_ambiguous"]
        else "Model prediction confidence is below 70%. Chemical recommendations are withheld until physical examination by a certified agricultural expert."
    )

    guidance = get_guidance(results["predicted_class"])
    if guidance:
        render_guidance_tabs(
            guidance=guidance,
            model_confidence=results["confidence"],
            is_prediction_context=True,
            is_ambiguous=guidance_ambiguous,
            ambiguity_message=ambiguity_msg,
        )
    else:
        st.warning("Guidance catalog entry not found for this class.")


    # --- In-Memory Session History Logging ---
    st.markdown("---")
    st.markdown("##### 💾 Log Diagnostic Record to Session History")
    note_col, btn_col1, btn_col2 = st.columns([3, 1.5, 1.5])
    with note_col:
        custom_notes = st.text_input(
            "Add diagnostic field notes (optional):",
            placeholder="e.g., North greenhouse bench 4, early symptoms observed on lower canopy.",
            key="scan_note_input",
        )
    with btn_col1:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("💾 Log to Session", use_container_width=True):
            scan_record = add_session_scan(
                filename=image_name,
                crop=results["crop"],
                condition=results["condition"],
                is_healthy=results["is_healthy"],
                predicted_class=results["predicted_class"],
                confidence=results["confidence"],
                top_predictions=results["top_predictions"],
                display_image=results["display_image"],
                overlay_image=overlay_img,
                notes=custom_notes,
                confirmed_crop=results.get("confirmed_crop"),
                guidance_snapshot=guidance,
            )
            st.session_state["active_report_scan_id"] = scan_record["id"]
            st.success(
                f"Scan **{scan_record['id']}** successfully logged to session history!"
            )
    with btn_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🖨️ View & Print Report", use_container_width=True):
            scan_record = add_session_scan(
                filename=image_name,
                crop=results["crop"],
                condition=results["condition"],
                is_healthy=results["is_healthy"],
                predicted_class=results["predicted_class"],
                confidence=results["confidence"],
                top_predictions=results["top_predictions"],
                display_image=results["display_image"],
                overlay_image=overlay_img,
                notes=custom_notes,
                confirmed_crop=results.get("confirmed_crop"),
                guidance_snapshot=guidance,
            )
            st.session_state["active_report_scan_id"] = scan_record["id"]
            st.session_state["nav_selection"] = "Scan History"
            st.rerun()

