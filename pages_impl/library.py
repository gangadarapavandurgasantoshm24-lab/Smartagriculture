import streamlit as st

from src.config import (
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
)
from src.disease_data import filter_diseases, get_all_crops
from src.guidance import get_guidance, render_guidance_tabs
from src.styles import render_hero


def render_library():
    """Render the searchable 38-class Disease Library encyclopedia."""
    render_hero(
        title="Crop Pathology & Disease Encyclopedia",
        subtitle="Complete reference catalog of the 14 crop species and 38 botanical conditions classified by the model, featuring symptoms and integrated pest management (IPM) practices.",
        badge="BOTANICAL KNOWLEDGE BASE",
    )

    all_crops = ["All"] + get_all_crops()

    # Search & Filter Controls
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 3])

    with filter_col1:
        selected_crop = st.selectbox("Filter by Crop:", all_crops, index=0)

    with filter_col2:
        selected_cat = st.selectbox(
            "Filter by Health Status:", ["All", "Diseased", "Healthy"], index=0
        )

    with filter_col3:
        search_kw = st.text_input(
            "Search keywords:",
            placeholder="e.g., blight, powdery mildew, fungus, mosaic, rust...",
        )

    # Perform filter
    records = filter_diseases(
        crop=selected_crop, category=selected_cat, search_query=search_kw
    )

    st.markdown(
        f"<div style='font-size: 0.9rem; color: #556E63; margin-bottom: 16px;'>Displaying <strong>{len(records)}</strong> matching classes out of 38 total:</div>",
        unsafe_allow_html=True,
    )

    if not records:
        st.warning("No disease or crop records matched your filter criteria.")
        return

    # Render results in a clean 2-column grid
    for i in range(0, len(records), 2):
        col_left, col_right = st.columns(2)

        for col, idx in [(col_left, i), (col_right, i + 1)]:
            if idx < len(records):
                item = records[idx]
                badge_class = (
                    "badge-healthy"
                    if item["category"] == "Healthy"
                    else "badge-diseased"
                )

                with col:
                    st.markdown(
                        f"""
                        <div style="background: #FFFFFF; border: 1px solid {COLOR_BORDER}; border-radius: 14px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span class="badge-crop">{item['crop']}</span>
                                <span class="{badge_class}">{item['category'].upper()}</span>
                            </div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: {COLOR_FOREST_GREEN}; margin-bottom: 4px;">
                                {item['common_name']}
                            </div>
                            <div style="font-size: 0.8rem; color: #6E887B; margin-bottom: 12px;">
                                Dataset Key: <code>{item['class_key']}</code>
                            </div>
                            
                            <p style="font-size: 0.88rem; margin-bottom: 6px;">
                                <strong>Pathogen / Agent:</strong> <em>{item['pathogen']}</em>
                            </p>
                            <p style="font-size: 0.88rem; margin-bottom: 8px; color: #2E453A;">
                                <strong>Symptoms:</strong> {item['symptoms']}
                            </p>
                            <div style="background: {COLOR_LIGHT_SAGE}; border-left: 3px solid {COLOR_SECONDARY_GREEN}; padding: 10px 12px; border-radius: 6px; font-size: 0.83rem; color: {COLOR_FOREST_GREEN};">
                                <strong>Agronomic IPM Management:</strong><br>
                                {item['ipm_practices']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    guidance = get_guidance(item["class_key"])
                    if guidance:
                        with st.expander(
                            f"📋 View 7-Tab Guidance: {item['common_name']}",
                            expanded=False,
                        ):
                            render_guidance_tabs(guidance, is_prediction_context=False)
