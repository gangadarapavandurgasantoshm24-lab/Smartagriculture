import platform
import sys
from pathlib import Path
import keras
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from src.config import (
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
    MODEL_PATH,
    TARGET_CONV_LAYER,
)
from src.disease_data import filter_diseases, get_all_crops
from src.evaluation_data import (
    get_actual_metrics_summary,
    get_evaluation_chart_paths,
    load_classification_report_csv,
    load_training_history,
)
from src.guidance import get_all_guidance, get_guidance, render_guidance_tabs
from src.inference import load_class_names, load_model, parse_class_label
from src.pesticides import get_pesticide_audit_summary, get_verified_products
from src.styles import render_hero


def render_settings():
    """
    Render the comprehensive Settings, System Architecture & Technical Blueprint page.
    Consolidates Model Performance evaluation analytics and Disease Library encyclopedia into a
    unified, structured 9-section master reference.
    """
    render_hero(
        title="Settings, Technical Architecture & Master Reference",
        subtitle="Consolidated technical blueprint: deep neural network topology, empirical benchmark evaluations, complete 38-class pathology encyclopedia, and regulatory safety standards.",
        badge="TECHNICAL BLUEPRINT",
    )

    model = load_model()
    class_names = load_class_names()
    metrics = get_actual_metrics_summary()
    history_df = load_training_history()
    report_csv = load_classification_report_csv()
    chart_paths = get_evaluation_chart_paths()
    pesticide_summary = get_pesticide_audit_summary()

    # Calculate parameter counts
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total_params = trainable_count + non_trainable_count

    # =========================================================================
    # SECTION 1: APPLICATION OVERVIEW
    # =========================================================================
    st.markdown("### 1. 🌿 Application Overview")
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 1.15rem; font-weight: 800; color: {COLOR_FOREST_GREEN}; margin-bottom: 8px;">
                SmartAgri AI — Clinical Plant Pathology Screening & Diagnostic System
            </div>
            <p style="font-size: 0.92rem; color: #3E544A; line-height: 1.6; margin-bottom: 12px;">
                SmartAgri AI is an automated, explainable plant leaf disease screening framework developed using Python, Streamlit, 
                and TensorFlow/Keras. The system leverages a tailored sequential Convolutional Neural Network (CNN) trained on the 
                PlantVillage benchmark dataset to classify <strong>38 botanical conditions</strong> across <strong>14 agricultural crop species</strong>.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; padding: 12px 16px; background: {COLOR_LIGHT_SAGE}; border-radius: 8px; font-size: 0.88rem; color: {COLOR_FOREST_GREEN};">
                <div>🎯 <strong>Active Architecture:</strong> {model.name}</div>
                <div>🌿 <strong>Supported Crops:</strong> 14 Species</div>
                <div>🏷️ <strong>Taxonomic Scope:</strong> 38 Exact Classes</div>
                <div>👁️ <strong>Explainability Layer:</strong> {TARGET_CONV_LAYER}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION 2: MODEL ARCHITECTURE AND TECHNICAL SPECIFICATIONS
    # =========================================================================
    st.markdown("### 2. 🏛️ Model Architecture & Technical Specifications")

    col_spec1, col_spec2, col_spec3 = st.columns(3)
    with col_spec1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Model Topology</div>
                <div class="stat-value" style="font-size: 1.35rem;">{model.name}</div>
                <div class="stat-desc">Sequential 3-Block CNN</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_spec2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total Parameters</div>
                <div class="stat-value" style="font-size: 1.35rem;">{total_params:,}</div>
                <div class="stat-desc">Trainable: {trainable_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_spec3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Input / Output Dimensions</div>
                <div class="stat-value" style="font-size: 1.35rem;">256 × 256 × 3</div>
                <div class="stat-desc">Output: 38 Softmax Logits</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    with st.expander("🔬 View Detailed Layer-by-Layer Architectural Breakdown", expanded=False):
        layer_rows = []
        for idx, layer in enumerate(model.layers):
            p_count = sum(tf.keras.backend.count_params(w) for w in layer.weights)
            is_target = layer.name == TARGET_CONV_LAYER
            layer_rows.append(
                {
                    "Layer #": idx,
                    "Layer Name": layer.name + (" ⭐ (Grad-CAM Target)" if is_target else ""),
                    "Type": layer.__class__.__name__,
                    "Output Shape": str(getattr(layer, "output_shape", "N/A")),
                    "Parameters": f"{p_count:,}",
                }
            )
        st.dataframe(pd.DataFrame(layer_rows), use_container_width=True, height=350)

    # Runtime Environment Diagnostics
    st.markdown("##### 💻 Runtime Software & Execution Environment")
    env_c1, env_c2 = st.columns(2)
    with env_c1:
        st.markdown(f"- **Operating System:** `{platform.system()} {platform.release()}`")
        st.markdown(f"- **Python Runtime:** `{sys.version.split()[0]}`")
        st.markdown(f"- **Streamlit Version:** `{st.__version__}`")
        st.markdown(f"- **TensorFlow Engine:** `{tf.__version__}`")
        st.markdown(f"- **Keras Implementation:** `{keras.__version__}`")
    with env_c2:
        st.markdown(f"- **Active Checkpoint:** `{MODEL_PATH.resolve()}`")
        st.markdown(f"- **Target Explainer Layer:** `{TARGET_CONV_LAYER}`")
        st.markdown(f"- **Hardware Acceleration:** `Native Windows CPU (Optimized)`")
        st.markdown(f"- **Session History Storage:** `In-Memory Streamlit SessionState (No Disk Persistence)`")

    st.markdown("---")

    # =========================================================================
    # SECTION 3: SUPPORTED CROPS AND ALL 38 CLASSES
    # =========================================================================
    st.markdown("### 3. 🌾 Supported Crops & All 38 Classes")
    st.caption(
        "Complete taxonomy of the 14 agricultural crop species and 38 botanical conditions recognized by the CNN model. "
        "Notice: The model is bounded strictly to this 38-class taxonomy and does not claim comprehensive coverage of all regional or state crops."
    )

    all_crops = ["All"] + get_all_crops()
    c_col1, c_col2, c_col3 = st.columns([2, 2, 3])
    with c_col1:
        selected_crop = st.selectbox("Filter Encyclopedia by Crop:", all_crops, index=0)
    with c_col2:
        selected_status = st.selectbox("Filter by Status:", ["All", "Healthy", "Diseased"], index=0)
    with c_col3:
        search_kw = st.text_input("Search symptoms, pathogens, or class names:", placeholder="e.g. rust, scab, blight, healthy, virus...")

    filtered_records = filter_diseases(crop=selected_crop, category=selected_status, search_query=search_kw)

    st.markdown(f"<div style='font-size: 0.88rem; color: #556E63; margin-bottom: 12px;'>Showing <strong>{len(filtered_records)}</strong> of 38 classes:</div>", unsafe_allow_html=True)

    # Render Compact Searchable Table / Expanders
    for rec in filtered_records:
        is_h = rec["category"] == "Healthy"
        status_badge = "🟢 HEALTHY" if is_h else "🔴 DISEASED"
        guidance_entry = get_guidance(rec["class_key"]) or {}
        chem_status = guidance_entry.get("chemical_treatment_status", "Requires Expert Confirmation")
        
        with st.expander(f"{rec['crop']} — {rec['common_name']} ({status_badge})", expanded=False):
            info_left, info_right = st.columns([3, 2])
            with info_left:
                st.markdown(f"**Exact Dataset Class Key:** ` {rec['class_key']} `")
                st.markdown(f"**Pathogen / Agent:** *{rec['pathogen']}*")
                st.markdown(f"**Visual Symptoms:** {rec['symptoms']}")
                st.markdown(f"**Agronomic IPM Management:** {rec['ipm_practices']}")
            with info_right:
                st.markdown(f"**Category:** `{rec['category']}`")
                st.markdown(f"**Chemical Treatment Status:** `{chem_status}`")
                # Show verified commercial products if any
                v_prods = get_verified_products(rec["class_key"])
                if v_prods:
                    trade_names = ", ".join(f"**{p['trade_name']}** ({p['manufacturer']})" for p in v_prods)
                    st.markdown(f"**Verified CIBRC Products (India):** {trade_names}")
                elif not is_h:
                    st.caption("Commercial Indian trade names require local KVK confirmation.")

    st.markdown("---")

    # =========================================================================
    # SECTION 4: DATASET INFORMATION
    # =========================================================================
    st.markdown("### 4. 📂 Dataset Information & Laboratory Boundaries")
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 1.05rem; font-weight: 700; color: {COLOR_FOREST_GREEN}; margin-bottom: 6px;">
                PlantVillage Benchmark Dataset Specifications
            </div>
            <p style="font-size: 0.9rem; color: #3E544A; line-height: 1.55; margin-bottom: 12px;">
                The underlying neural network was trained on the standardized <strong>PlantVillage</strong> computer vision benchmark, 
                comprising <strong>54,305 total curated leaf images</strong> partitioned across 38 classes. An independent, held-out validation 
                partition of <strong>17,572 images</strong> was used to verify model convergence and generalization metrics.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.88rem; color: #26332D;">
                <div>📦 <strong>Total Images:</strong> 54,305 photos</div>
                <div>🧪 <strong>Validation Samples:</strong> 17,572 held-out images</div>
                <div>🌿 <strong>Host Crops:</strong> 14 distinct botanical species</div>
                <div>📐 <strong>Standard Resolution:</strong> 256 × 256 RGB</div>
            </div>
            <div style="background: #FFFDF7; border-left: 4px solid #D97706; padding: 12px 14px; border-radius: 6px; margin-top: 14px; font-size: 0.86rem; color: #78350F; line-height: 1.5;">
                <strong>⚠️ Known Dataset Domain Shift & Laboratory Limitations:</strong><br>
                PlantVillage images were collected under controlled laboratory illumination with excised leaf blades positioned flat against 
                uniform gray or black paper backgrounds. When the CNN encounters unconstrained real-world field photography containing 
                human hands, skin tones, direct sunlight glare, or busy orchard backgrounds, the model can experience cross-species probability 
                dispersion. The SmartAgri AI <em>Target Crop Context</em> feature was engineered to neutralize this background interference.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION 5: EVALUATION METRICS AND MODEL PERFORMANCE
    # =========================================================================
    st.markdown("### 5. 📊 Evaluation Metrics & Model Performance")
    st.caption("Empirical performance summary evaluated on the 17,572 held-out validation images.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Application Model Accuracy</div>
                <div class="stat-value">{metrics['application_model_val_accuracy']:.2f}%</div>
                <div class="stat-desc">Target accuracy across 38 classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Baseline Accuracy</div>
                <div class="stat-value">{metrics['baseline_val_accuracy']:.2f}%</div>
                <div class="stat-desc">Peak epoch evaluation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Macro F1-Score</div>
                <div class="stat-value">{metrics['macro_f1']:.4f}</div>
                <div class="stat-desc">Unweighted class average</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Weighted F1-Score</div>
                <div class="stat-value">{metrics['weighted_f1']:.4f}</div>
                <div class="stat-desc">Sample-weighted average</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Top & Bottom Recall Classes
    st.markdown("##### ⚖️ Per-Class Recall Extremes (Diagnostic Strengths vs Vulnerabilities)")
    col_str, col_vul = st.columns(2)
    with col_str:
        st.markdown("##### 🏆 Top 5 Strongest Classes (Highest Recall)")
        if "top_classes" in metrics:
            for item in metrics["top_classes"]:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border-left: 4px solid {COLOR_FOREST_GREEN}; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <div style="font-weight: 700; font-size: 0.9rem; color: {COLOR_FOREST_GREEN};">{item['class']}</div>
                        <div style="font-size: 0.82rem; color: #556E63;">
                            Recall: <strong>{float(item['recall'])*100:.2f}%</strong> | Precision: {float(item['precision'])*100:.2f}% | F1: {float(item['f1_score']):.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_vul:
        st.markdown("##### ⚠️ Top 5 Vulnerable Classes (Lowest Recall)")
        if "bottom_classes" in metrics:
            for item in metrics["bottom_classes"]:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border-left: 4px solid #D9383A; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <div style="font-weight: 700; font-size: 0.9rem; color: #8F1D1E;">{item['class']}</div>
                        <div style="font-size: 0.82rem; color: #556E63;">
                            Recall: <strong>{float(item['recall'])*100:.2f}%</strong> | Precision: {float(item['precision'])*100:.2f}% | F1: {float(item['f1_score']):.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if report_csv is not None:
        with st.expander("📄 View Complete 38-Class Classification Metrics Table (reports/classification_report.csv)"):
            st.dataframe(report_csv, use_container_width=True, height=350)

    st.markdown("---")

    # =========================================================================
    # SECTION 6: CONFUSION MATRIX AND TRAINING CHARTS
    # =========================================================================
    st.markdown("### 6. 🗺️ Confusion Matrix & Training Convergence")

    if history_df is not None:
        st.markdown("##### 📈 Training & Validation Convergence Trajectories (10 Epochs)")
        t_tab1, t_tab2 = st.tabs(["Accuracy Convergence Trajectory", "Categorical Cross-Entropy Loss"])

        with t_tab1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["accuracy"]*100, mode="lines+markers", name="Training Accuracy", line=dict(color=COLOR_SECONDARY_GREEN, width=3)))
            fig_acc.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["val_accuracy"]*100, mode="lines+markers", name="Validation Accuracy", line=dict(color=COLOR_FOREST_GREEN, width=3, dash="dot")))
            fig_acc.add_annotation(x=7, y=93.22, text="Best Epoch 7: 93.22%", showarrow=True, arrowhead=2, arrowcolor=COLOR_FOREST_GREEN, bgcolor="#FFFFFF")
            fig_acc.update_layout(title="Accuracy vs Epoch", xaxis_title="Epoch", yaxis_title="Accuracy (%)", template="plotly_white", height=340)
            st.plotly_chart(fig_acc, use_container_width=True)

        with t_tab2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["loss"], mode="lines+markers", name="Training Loss", line=dict(color="#D97706", width=3)))
            fig_loss.add_trace(go.Scatter(x=history_df["epoch"], y=history_df["val_loss"], mode="lines+markers", name="Validation Loss", line=dict(color="#B91C1C", width=3, dash="dot")))
            fig_loss.update_layout(title="Loss vs Epoch", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white", height=340)
            st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("##### 🗺️ Confusion Matrix Visualizations")
    cm_c1, cm_c2 = st.columns(2)

    with cm_c1:
        norm_png = chart_paths["normalized_confusion_matrix_png"]
        if norm_png.is_file():
            st.image(str(norm_png), caption="Normalized Confusion Matrix (Row Proportion)", use_container_width=True)
        else:
            st.warning("Normalized confusion matrix plot not found at `reports/normalized_confusion_matrix.png`.")

    with cm_c2:
        count_png = chart_paths["confusion_matrix_png"]
        if count_png.is_file():
            st.image(str(count_png), caption="Raw Frequency Confusion Matrix", use_container_width=True)
        else:
            st.warning("Raw confusion matrix plot not found at `reports/confusion_matrix.png`.")

    gallery_png = chart_paths["misclassified_gallery_png"]
    if gallery_png.is_file():
        with st.expander("🖼️ View Representative Validation Misclassifications Gallery"):
            st.image(str(gallery_png), caption="Representative Misclassified Validation Samples (reports/misclassified_gallery.png)", use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # SECTION 7: GRAD-CAM EXPLAINABILITY
    # =========================================================================
    st.markdown("### 7. 👁️ Grad-CAM Explainability & Visual Attribution")
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 1.05rem; font-weight: 700; color: {COLOR_FOREST_GREEN}; margin-bottom: 6px;">
                Gradient-Weighted Class Activation Mapping (Layer: <code>{TARGET_CONV_LAYER}</code>)
            </div>
            <p style="font-size: 0.9rem; color: #3E544A; line-height: 1.55; margin-bottom: 12px;">
                Deep neural networks are often perceived as non-interpretable black boxes. SmartAgri AI integrates 
                <strong>Grad-CAM</strong> to extract spatial gradient flow flowing from the top predicted class back into 
                the final convolutional layer <code>{TARGET_CONV_LAYER}</code> (128 feature maps).
            </p>
            <div style="font-size: 0.88rem; color: #2E5444; line-height: 1.6;">
                <strong>Why Layer {TARGET_CONV_LAYER}?</strong><br>
                Earlier layers (<code>conv2d_1</code>, <code>conv2d_2</code>) isolate primitive geometric edges and color transitions. 
                Layer <code>{TARGET_CONV_LAYER}</code> contains the richest semantic abstraction of pathogen morphology (lesion rings, chlorotic borders, fungal pustules) 
                while still preserving 2D spatial arrangement (60×60 resolution) prior to dense classification flattening.
            </div>
            <div style="background: #F8FAF8; border-radius: 6px; padding: 10px 14px; margin-top: 12px; font-size: 0.86rem; color: #556E63;">
                <strong>Clinical Takeaway:</strong> High activation values (warm red/yellow hues) reflect tissue regions that correlated positively with the model's prediction score. They do not constitute independent biological proof of disease.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION 8: DISEASE AND TREATMENT GUIDANCE COVERAGE
    # =========================================================================
    st.markdown("### 8. 🌿 Disease & Treatment Guidance Coverage")
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 1.05rem; font-weight: 700; color: {COLOR_FOREST_GREEN}; margin-bottom: 6px;">
                Audit of Source-Grounded Treatment & Pesticide Records
            </div>
            <p style="font-size: 0.9rem; color: #3E544A; line-height: 1.55; margin-bottom: 12px;">
                Every disease management entry in SmartAgri AI is grounded in official publications from the Central Insecticides Board & 
                Registration Committee (CIBRC), Directorate of Plant Protection, Quarantine & Storage (PPQS), and State Agricultural Universities (SAUs).
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; font-size: 0.88rem;">
                <div style="background: #E8F5E9; padding: 10px; border-radius: 6px; border: 1px solid #C8E6C9;">
                    <strong style="color: #1B5E20;">12 Healthy Classes</strong><br>
                    <span style="font-size: 0.82rem; color: #2E7D32;">Chemicals Inappropriate (Prohibited)</span>
                </div>
                <div style="background: #EBF3ED; padding: 10px; border-radius: 6px; border: 1px solid #A7D3B9;">
                    <strong style="color: #173F35;">10 Verified Classes</strong><br>
                    <span style="font-size: 0.82rem; color: #2F6B4F;">{pesticide_summary['total_commercial_products']} CIBRC Commercial Products</span>
                </div>
                <div style="background: #FFFDF7; padding: 10px; border-radius: 6px; border: 1px solid #FDE68A;">
                    <strong style="color: #B45309;">16 Expert Classes</strong><br>
                    <span style="font-size: 0.82rem; color: #78350F;">Trade Names Withheld (KVK Required)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION 9: SOURCES, LIMITATIONS AND SAFETY
    # =========================================================================
    st.markdown("### 9. ⚖️ Authoritative Sources, Operational Scope & Safety Disclaimers")
    st.warning(
        """
        **Official Agricultural Diagnostic Boundaries & Safety Directives:**
        1. **Authoritative Sources Cited:**
           - Central Insecticides Board & Registration Committee (CIBRC), Directorate of Plant Protection, Quarantine & Storage (PPQS), Ministry of Agriculture & Farmers Welfare, Government of India.
           - Indian Council of Agricultural Research (ICAR) National Institutes & Crop Protection Protocols.
           - State Agricultural Universities (e.g. Dr. YSP UHF Solan, TNAU Coimbatore, ANGRAU Andhra Pradesh).
        2. **Domain Boundaries:**
           - The neural network is trained exclusively on the 38 classes of the PlantVillage dataset across 14 crops. It will not identify non-crop foliage, forest trees, weeds, or unlisted pathogens.
           - Never transfer dosage recommendations across crops or formulations without statutory CIBRC authorization.
        3. **Advisory Decision Support Only:**
           - Machine learning inferences provide assistive screening support for early detection. They do not constitute certified biological proof.
           - Always verify symptoms with your local Krishi Vigyan Kendra (KVK) extension scientist or certified plant protection officer before purchasing chemical pesticides or making irreversible agronomic interventions.
        """
    )
