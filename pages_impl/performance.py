from pathlib import Path
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
)
from src.evaluation_data import (
    get_actual_metrics_summary,
    get_evaluation_chart_paths,
    load_classification_report_csv,
    load_confusion_matrix_csv,
    load_misclassified_samples,
    load_training_history,
)
from src.styles import render_hero


def render_performance():
    """Render the Model Performance & Evaluation page using verified artifacts."""
    render_hero(
        title="Model Benchmark & Evaluation Analytics",
        subtitle="Empirical performance audit on 17,572 held-out validation images. Complete training dynamics, confusion matrices, and per-class precision/recall metrics.",
        badge="VALIDATED ARTIFACTS",
    )

    metrics = get_actual_metrics_summary()
    history_df = load_training_history()
    report_csv = load_classification_report_csv()
    chart_paths = get_evaluation_chart_paths()

    # --- Metrics Banner ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Application Model</div>
                <div class="stat-value">{metrics['application_model_val_accuracy']:.2f}%</div>
                <div class="stat-desc">Target accuracy on 38 classes</div>
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
                <div class="stat-desc">Verified benchmark on 17,572 samples</div>
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
                <div class="stat-desc">Balanced unweighted mean</div>
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
                <div class="stat-desc">Sample support weighted mean</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # --- Training Convergence Curves (Plotly) ---
    st.markdown("### 📈 Training History & Epoch Convergence")

    if history_df is not None:
        tab_acc, tab_loss = st.tabs(
            ["Accuracy Trajectory", "Cross-Entropy Loss Trajectory"]
        )

        with tab_acc:
            fig_acc = go.Figure()
            fig_acc.add_trace(
                go.Scatter(
                    x=history_df["epoch"],
                    y=history_df["accuracy"] * 100.0,
                    mode="lines+markers",
                    name="Training Accuracy",
                    line=dict(color=COLOR_SECONDARY_GREEN, width=3),
                    marker=dict(size=7),
                )
            )
            fig_acc.add_trace(
                go.Scatter(
                    x=history_df["epoch"],
                    y=history_df["val_accuracy"] * 100.0,
                    mode="lines+markers",
                    name="Validation Accuracy",
                    line=dict(color=COLOR_FOREST_GREEN, width=3, dash="dot"),
                    marker=dict(size=7),
                )
            )
            # Add annotation for best epoch
            fig_acc.add_annotation(
                x=7,
                y=93.22,
                text="Best Validation Accuracy: 93.22% (Epoch 7)",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLOR_FOREST_GREEN,
                font=dict(color=COLOR_FOREST_GREEN, size=11),
                bordercolor=COLOR_FOREST_GREEN,
                borderwidth=1,
                borderpad=4,
                bgcolor="#FFFFFF",
            )
            fig_acc.update_layout(
                title="Training vs Validation Accuracy Across 10 Epochs",
                xaxis_title="Epoch",
                yaxis_title="Accuracy (%)",
                hovermode="x unified",
                template="plotly_white",
                height=380,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with tab_loss:
            fig_loss = go.Figure()
            fig_loss.add_trace(
                go.Scatter(
                    x=history_df["epoch"],
                    y=history_df["loss"],
                    mode="lines+markers",
                    name="Training Loss",
                    line=dict(color="#D97706", width=3),
                    marker=dict(size=7),
                )
            )
            fig_loss.add_trace(
                go.Scatter(
                    x=history_df["epoch"],
                    y=history_df["val_loss"],
                    mode="lines+markers",
                    name="Validation Loss",
                    line=dict(color="#B91C1C", width=3, dash="dot"),
                    marker=dict(size=7),
                )
            )
            fig_loss.update_layout(
                title="Training vs Validation Sparse Categorical Cross-Entropy Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                hovermode="x unified",
                template="plotly_white",
                height=380,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    # --- Per-Class Diagnostic Strengths & Weaknesses ---
    st.markdown("---")
    st.markdown("### ⚖️ Per-Class Recall Extremes")
    st.caption(
        "Analysis of true positive detection rates from reports/classification_report.csv."
    )

    col_strong, col_weak = st.columns(2)

    with col_strong:
        st.markdown("##### 🏆 Top 5 Strongest Classes (Highest Recall)")
        if "top_classes" in metrics:
            for item in metrics["top_classes"]:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border-left: 4px solid {COLOR_FOREST_GREEN}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                        <div style="font-weight: 700; color: {COLOR_FOREST_GREEN};">{item['class']}</div>
                        <div style="font-size: 0.85rem; color: #556E63;">
                            Recall: <strong>{float(item['recall'])*100:.2f}%</strong> | Precision: {float(item['precision'])*100:.2f}% | Support: {int(item['support'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_weak:
        st.markdown("##### ⚠️ Top 5 Vulnerable Classes (Lowest Recall)")
        if "bottom_classes" in metrics:
            for item in metrics["bottom_classes"]:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border-left: 4px solid #D9383A; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                        <div style="font-weight: 700; color: #8F1D1E;">{item['class']}</div>
                        <div style="font-size: 0.85rem; color: #556E63;">
                            Recall: <strong>{float(item['recall'])*100:.2f}%</strong> | Precision: {float(item['precision'])*100:.2f}% | Support: {int(item['support'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --- Confusion Matrix Visualizations ---
    st.markdown("---")
    st.markdown("### 🗺️ Empirical Confusion Matrix Heatmaps")

    cm_tab1, cm_tab2, cm_tab3 = st.tabs(
        [
            "Normalized Heatmap (Row Proportion)",
            "Raw Count Heatmap",
            "Full 38×38 Metrics Table",
        ]
    )

    with cm_tab1:
        norm_png = chart_paths["normalized_confusion_matrix_png"]
        if norm_png.is_file():
            st.image(
                str(norm_png),
                caption="Normalized Confusion Matrix (reports/normalized_confusion_matrix.png)",
                use_container_width=True,
            )
        else:
            st.warning("Normalized confusion matrix plot not found.")

    with cm_tab2:
        count_png = chart_paths["confusion_matrix_png"]
        if count_png.is_file():
            st.image(
                str(count_png),
                caption="Raw Confusion Matrix (reports/confusion_matrix.png)",
                use_container_width=True,
            )
        else:
            st.warning("Raw confusion matrix plot not found.")

    with cm_tab3:
        if report_csv is not None:
            st.dataframe(report_csv, use_container_width=True, height=450)

    # --- Error Analysis Observations ---
    st.markdown("---")
    st.markdown("### 🔎 Frequent Validation Confusions & Field Takeaways")
    st.info(
        """
        **Top Directed Confusions Identified in Error Analysis (`reports/error_analysis.md`):**
        1. **`Corn Gray leaf spot` → `Corn Northern Leaf Blight`:** 67 misclassifications due to overlapping early rectangular chlorotic lesion margins.
        2. **`Tomato Early blight` → `Tomato Late blight`:** 37 misclassifications occurring when early concentric target rings expand into broad dark necrotic leaf tissue.
        3. **`Grape Black rot` → `Grape Esca`:** 31 misclassifications triggered by interveinal chlorosis before distinct black pycnidia form.
        4. **`Peach Bacterial spot` → `Tomato Late blight / Bacterial spot`:** Due to similar angular, water-soaked polygonal lesion shapes.
        """
    )

    gallery_png = chart_paths["misclassified_gallery_png"]
    if gallery_png.is_file():
        with st.expander(
            "🖼️ View Gallery of Representative Validation Misclassifications (24 samples)"
        ):
            st.image(
                str(gallery_png),
                caption="Representative Misclassified Validation Samples (reports/misclassified_gallery.png)",
                use_container_width=True,
            )
