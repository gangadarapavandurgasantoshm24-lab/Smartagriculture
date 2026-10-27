from pathlib import Path
import streamlit as st
import pandas as pd

from src.config import (
    COLOR_FOREST_GREEN,
    COLOR_SECONDARY_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_BORDER,
    COLOR_ACCENT_GREEN,
    TEST_IMAGES_DIR,
)
from src.session_history import get_session_scans, get_session_scan_stats
from src.disease_data import get_all_crops
from src.evaluation_data import get_actual_metrics_summary
from src.styles import render_hero


def render_dashboard():
    """Render the main overview dashboard."""
    render_hero(
        title="Plant Pathology Intelligence & Explainability",
        subtitle="Autonomous leaf disease diagnosis across 14 crops and 38 conditions. Verified CNN architecture powered by real-time Grad-CAM visual attention at layer conv2d_3.",
        badge="APPLICATION MODEL: BEST_MODEL.KERAS",
    )

    metrics = get_actual_metrics_summary()
    session_stats = get_session_scan_stats()
    crops = get_all_crops()

    # --- KPI Metric Cards ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Model Accuracy</div>
                <div class="stat-value">{metrics['application_model_val_accuracy']:.2f}%</div>
                <div class="stat-desc">Best validation accuracy (38 classes)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Supported Scope</div>
                <div class="stat-value">{metrics['classes_count']} Classes</div>
                <div class="stat-desc">Across {len(crops)} crop species</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Explainability</div>
                <div class="stat-value">conv2d_3</div>
                <div class="stat-desc">128-channel Grad-CAM attention</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Session Scans</div>
                <div class="stat-value">{session_stats['total_scans']} Logged</div>
                <div class="stat-desc">In-memory session log</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # --- Quick Diagnostic Action & Dataset Overview ---
    row2_left, row2_right = st.columns([3, 2])

    with row2_left:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-header">
                    <span>🚀</span> Quick Diagnostic Workflow
                </div>
                <p style="color: #4A6357; font-size: 0.95rem; margin-bottom: 16px;">
                    Instantly analyze plant leaf images with end-to-end preprocessing, class probability distributions, and live Grad-CAM heatmaps.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "💡 **Ready to scan?** Navigate to **'Disease Detection'** in the sidebar to upload your own leaf photo, or test with built-in sample images from the PlantVillage dataset."
        )

        # Preset Sample Quick Launcher
        if TEST_IMAGES_DIR.exists():
            sample_files = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(
                TEST_IMAGES_DIR.glob("*.JPG")
            )
            if sample_files:
                st.markdown("##### 🧪 Fast Verification Samples from Dataset")
                sample_cols = st.columns(min(4, len(sample_files)))
                for idx, sample_path in enumerate(sample_files[:4]):
                    with sample_cols[idx]:
                        sample_name = sample_path.stem
                        if st.button(
                            f"🔍 {sample_name[:16]}...",
                            key=f"quick_sample_{idx}",
                            use_container_width=True,
                        ):
                            st.session_state["selected_sample_image"] = str(sample_path)
                            st.session_state["nav_selection"] = "Disease Detection"
                            st.rerun()

    with row2_right:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-header">
                    <span>🌾</span> Supported Crops Portfolio
                </div>
                <p style="color: #4A6357; font-size: 0.88rem; margin-bottom: 12px;">
                    Trained on 70,295 benchmark training images and 17,572 validation images:
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px;">
            """,
            unsafe_allow_html=True,
        )

        crops_html = "".join([f"<span class='badge-crop'>{c}</span>" for c in crops])
        st.markdown(
            f"<div style='display: flex; flex-wrap: wrap; gap: 6px;'>{crops_html}</div></div>",
            unsafe_allow_html=True,
        )

    # --- Recent Scans Feed ---
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🕒 Recent Field Diagnostic History")

    recent_scans = get_session_scans(limit=5)
    if recent_scans:
        scan_records = []
        for s in recent_scans:
            badge = "🟢 Healthy" if s["is_healthy"] else "🔴 Diseased"
            scan_records.append(
                {
                    "Scan ID": s["id"],
                    "Timestamp": s["timestamp"],
                    "Image File": s["filename"],
                    "Crop": s["crop"],
                    "Condition": s["condition"],
                    "Health Status": badge,
                    "Confidence": f"{s['confidence_percent']:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(scan_records), use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px dashed {COLOR_BORDER}; border-radius: 12px; padding: 24px; text-align: center; color: #556E63;">
                <p style="font-size: 1rem; margin-bottom: 6px;">No diagnostic scans recorded in current session yet.</p>
                <p style="font-size: 0.85rem;">Run your first diagnosis in the <strong>Disease Detection</strong> page to begin building session history.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
