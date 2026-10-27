from pathlib import Path
import cv2
import numpy as np
import streamlit as st

from src.config import (
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
    TARGET_CONV_LAYER,
    TEST_IMAGES_DIR,
)
from src.gradcam import (
    COLORMAP_OPTIONS,
    apply_gradcam_overlay,
    generate_gradcam_heatmap,
)
from src.inference import predict_leaf
from src.styles import render_hero


def render_explainability():
    """Render the AI Explainability & Grad-CAM inspection page focused on layer conv2d_3."""
    render_hero(
        title="AI Explainability & Grad-CAM Inspection",
        subtitle="Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes the spatial attention of the convolutional neural network at layer conv2d_3 directly on leaf tissue.",
        badge="TRANSPARENT AI",
    )

    # Check if an active scan exists from the Detection workflow
    has_session_result = (
        "last_diagnostic_result" in st.session_state
        and st.session_state["last_diagnostic_result"] is not None
    )

    # 1. Image Source Selection
    source_mode = st.radio(
        "Select Leaf Image for Grad-CAM Inspection:",
        [
            "Active Diagnostic Scan from Session",
            "Choose From Benchmark Validation Samples",
            "Upload a New Leaf Image",
        ],
        horizontal=True,
        index=0 if has_session_result else 1,
    )

    res = None
    image_label = "Leaf Analysis"

    if source_mode == "Active Diagnostic Scan from Session":
        if not has_session_result:
            st.info(
                "📌 **No active leaf diagnosis in current session.**\n\n"
                "Please navigate to **Disease Detection** from the sidebar and upload a plant leaf image, "
                "or select **Choose From Benchmark Validation Samples** above to inspect sample attention."
            )
            return
        else:
            res = st.session_state["last_diagnostic_result"]
            image_label = st.session_state.get("last_image_name", "Current Session Leaf")
            display_img = res["display_image"]
            input_tensor = res["input_tensor"]
            top_class_idx = res["class_index"]
            top_class_name = res["predicted_class"]
            confidence = res["confidence_percent"]

    elif source_mode == "Choose From Benchmark Validation Samples":
        if TEST_IMAGES_DIR.exists():
            sample_files = sorted(
                list(TEST_IMAGES_DIR.glob("*.jpg"))
                + list(TEST_IMAGES_DIR.glob("*.JPG"))
                + list(TEST_IMAGES_DIR.glob("*.png"))
            )
            sample_name = st.selectbox(
                "Select a benchmark validation sample:", [f.name for f in sample_files]
            )
            sample_path = TEST_IMAGES_DIR / sample_name
            image_label = sample_name
            with open(sample_path, "rb") as f:
                selected_image_bytes = f.read()

            with st.spinner("Running CNN inference..."):
                res = predict_leaf(selected_image_bytes)
                display_img = res["display_image"]
                input_tensor = res["input_tensor"]
                top_class_idx = res["class_index"]
                top_class_name = res["predicted_class"]
                confidence = res["confidence_percent"]

    else:
        uploaded = st.file_uploader(
            "Upload a plant leaf image (.jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            image_label = uploaded.name
            with st.spinner("Running CNN inference..."):
                res = predict_leaf(uploaded.read())
                display_img = res["display_image"]
                input_tensor = res["input_tensor"]
                top_class_idx = res["class_index"]
                top_class_name = res["predicted_class"]
                confidence = res["confidence_percent"]
        else:
            st.info("👆 Please upload a plant leaf image to calculate Grad-CAM activations.")
            return

    if res is None:
        return

    st.markdown("---")

    # 2. Diagnostic Summary Banner
    badge_healthy = res.get("is_healthy", False)
    badge_text = "HEALTHY FOLIAGE" if badge_healthy else "DISEASED / PATHOGEN DETECTED"
    badge_bg = "#E8F5E9" if badge_healthy else "#FFEBEE"
    badge_color = "#1B5E20" if badge_healthy else "#B71C1C"

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: {badge_bg}; color: {badge_color}; font-weight: 700; font-size: 0.8rem; padding: 3px 8px; border-radius: 4px; border: 1px solid {badge_color};">
                        {badge_text}
                    </span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: {COLOR_FOREST_GREEN}; margin-left: 10px;">
                        {res['crop']}: {res['condition']}
                    </span>
                </div>
                <div style="font-size: 0.9rem; color: #556E63;">
                    Confidence: <strong style="color: {COLOR_SECONDARY_GREEN}; font-size: 1.1rem;">{res['confidence_percent']:.2f}%</strong> • Target Layer: <code>{TARGET_CONV_LAYER}</code>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Interactive Visualization Controls
    c_col1, c_col2, c_col3 = st.columns([1.5, 1.5, 2])
    with c_col1:
        chosen_colormap_name = st.selectbox(
            "Heatmap Palette:", list(COLORMAP_OPTIONS.keys()), index=0
        )
        colormap_code = COLORMAP_OPTIONS[chosen_colormap_name]
    with c_col2:
        alpha_val = st.slider(
            "Overlay Transparency (α):",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Higher values give stronger visual prominence to model heat activations.",
        )
    with c_col3:
        target_options = {
            f"#{p['rank']} {p['condition']} ({p['confidence_percent']:.1f}%)": p[
                "class_index"
            ]
            for p in res["top_predictions"]
        }
        chosen_target_label = st.selectbox(
            "Compute Gradient For Target Class:",
            list(target_options.keys()),
            index=0,
            help="Compare which spatial regions support the top prediction versus secondary candidates.",
        )
        target_class_idx = target_options[chosen_target_label]

    # 4. Compute Grad-CAM for chosen class index
    with st.spinner(f"Computing backpropagated gradients on layer {TARGET_CONV_LAYER}..."):
        heatmap = generate_gradcam_heatmap(
            input_tensor, target_class_index=target_class_idx
        )
        colored_heatmap, overlay_img = apply_gradcam_overlay(
            display_img, heatmap, alpha=alpha_val, colormap_code=colormap_code
        )

    # 5. Side-by-Side Responsive Display (Requirement 6)
    st.markdown("### 👁️ Model Attention Inspection: Side-by-Side View")

    col_orig, col_overlay = st.columns(2)

    with col_orig:
        st.markdown("##### 1. Original Leaf Image (256×256 RGB)")
        st.image(
            display_img,
            caption=f"Input: {image_label}",
            use_container_width=True,
        )

    with col_overlay:
        st.markdown(f"##### 2. Grad-CAM Overlay on Layer `{TARGET_CONV_LAYER}` (α={alpha_val:.2f})")
        st.image(
            overlay_img,
            caption=f"Grad-CAM Attention Heatmap (Target: {chosen_target_label})",
            use_container_width=True,
        )

    # 6. Clear Explanation of Highlighted Regions (Requirement 6)
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border-left: 4px solid {COLOR_SECONDARY_GREEN}; border-radius: 8px; padding: 16px 20px; margin-top: 16px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <div style="font-weight: 700; color: {COLOR_FOREST_GREEN}; font-size: 0.98rem; margin-bottom: 6px;">
                💡 Clinical Interpretation of Highlighted Regions
            </div>
            <p style="font-size: 0.9rem; color: #3E544A; line-height: 1.55; margin-bottom: 8px;">
                <strong>Warm Regions (Red / Orange / Yellow):</strong> Highlight the specific spatial pixels and tissue patterns 
                (such as chlorotic halos, necrotic spot margins, concentric target rings, or rust pustules) that provided the strongest positive 
                gradient activation toward predicting <code>{chosen_target_label}</code>.
            </p>
            <p style="font-size: 0.9rem; color: #3E544A; line-height: 1.55; margin-bottom: 8px;">
                <strong>Cool Regions (Blue / Dark Violet):</strong> Indicate background board, uninfected leaf margins, or veins that did not contribute 
                substantively to the classification score.
            </p>
            <div style="font-size: 0.86rem; color: #8F1D1E; background: #FFF5F5; padding: 8px 12px; border-radius: 6px; border: 1px solid #FED7D7;">
                ⚠️ <strong>Important Diagnostic Notice:</strong> Stronger highlighted regions indicate areas influencing the neural network's 
                statistical prediction. They do <strong>not</strong> independently prove that the disease diagnosis is biologically correct. 
                Always cross-verify symptoms with field pathology guidelines and local agricultural extension officers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 7. Targeted Layer Technical Insight Expander
    with st.expander("🔬 Why Layer 'conv2d_3' Was Selected for Explainability"):
        st.markdown(
            f"""
            - **Architectural Position:** The sequential CNN comprises 3 convolutional blocks:
              1. `conv2d_1` (32 filters): Detects low-level edge angles and basic color contrasts.
              2. `conv2d_2` (64 filters): Identifies intermediate textures, curves, and circular spot boundaries.
              3. **`conv2d_3` (128 filters):** Represents rich, high-level semantic disease motifs (e.g. pustule clusters, water-soaked margins) while still retaining 2D spatial coordinates ($60 \\times 60$ resolution) before spatial flattening into dense classification logits.
            - **Mathematical Operation:** Features in `conv2d_3` are weighted by pooled gradient backpropagation $\\alpha_k^c$ and gated through a rectified linear unit ($\\text{{ReLU}}$) to isolate strictly positive symptom evidence.
            """
        )
