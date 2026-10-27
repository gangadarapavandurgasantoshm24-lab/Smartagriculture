import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.config import (
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
)
from src.disease_data import get_all_crops
from src.report_generator import generate_printable_html_report
from src.session_history import (
    clear_session_history,
    get_session_scan_by_id,
    get_session_scan_stats,
    get_session_scans,
    init_session_history,
)
from src.styles import render_hero


def render_history():
    """Render the in-memory session-based Scan History and Print Report workflow."""
    init_session_history()

    render_hero(
        title="Field Diagnostic Audit & Scan History",
        subtitle="In-memory session log of recent plant leaf analyses. Review diagnostic probabilities, Grad-CAM attention, and generate professional print-ready assessment reports.",
        badge="SESSION SCOPED",
    )

    stats = get_session_scan_stats()
    all_crops = ["All"] + get_all_crops()

    # --- Active Print Report Modal / Section ---
    active_report_id = st.session_state.get("active_report_scan_id")
    if active_report_id:
        selected_scan = get_session_scan_by_id(active_report_id)
        if selected_scan:
            st.markdown("### 🖨️ Plant Leaf Disease Assessment Report Preview")
            st.caption(
                f"Generating diagnostic report for **{selected_scan['id']}** ({selected_scan['crop']} - {selected_scan['condition']}). "
                "Use the buttons below to print directly or download the A4 report for your records."
            )

            html_report = generate_printable_html_report(selected_scan)

            act_col1, act_col2, act_col3 = st.columns([2, 2, 2])
            with act_col1:
                st.download_button(
                    label="📥 Download Report (HTML / Print-Ready)",
                    data=html_report,
                    file_name=f"SmartAgri_Report_{selected_scan['id']}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with act_col2:
                # Direct JavaScript print trigger via HTML button
                print_trigger_html = f"""
                <script>
                function triggerPrint() {{
                    var printWindow = window.open('', '_blank');
                    printWindow.document.write({repr(html_report)});
                    printWindow.document.close();
                    printWindow.focus();
                    printWindow.print();
                }}
                </script>
                <button onclick="triggerPrint()" style="
                    background-color: {COLOR_FOREST_GREEN};
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: bold;
                    width: 100%;
                    cursor: pointer;
                    height: 38px;
                ">🖨️ Open Print Dialog (PDF)</button>
                """
                components.html(print_trigger_html, height=45)

            with act_col3:
                if st.button("❌ Close Report View", use_container_width=True):
                    st.session_state["active_report_scan_id"] = None
                    st.rerun()

            # Render HTML report in embedded frame for preview
            components.html(html_report, height=750, scrolling=True)
            st.markdown("---")
        else:
            # Active report ID not found in current session
            st.session_state["active_report_scan_id"] = None

    # --- Filters Bar ---
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        crop_filter = st.selectbox("Filter by Crop:", all_crops, index=0)
    with f_col2:
        status_filter = st.selectbox(
            "Filter by Health Status:", ["All", "Healthy", "Diseased"], index=0
        )
    with f_col3:
        max_rows = st.selectbox("Display Limit:", [10, 25, 50], index=1)

    scans = get_session_scans(
        crop_filter=crop_filter, status_filter=status_filter, limit=max_rows
    )

    # --- Summary Stats Banner ---
    st.markdown(
        f"""
        <div style="display: flex; gap: 20px; margin-bottom: 20px; font-size: 0.9rem; color: #556E63; flex-wrap: wrap;">
            <div>Session Scans: <strong>{stats['total_scans']}</strong></div>
            <div>Healthy Foliage: <strong style="color: #197A3E;">{stats['healthy_scans']}</strong></div>
            <div>Diseased Records: <strong style="color: #D9383A;">{stats['diseased_scans']}</strong></div>
            <div>Average Confidence: <strong>{stats['average_confidence']*100:.1f}%</strong></div>
            <div style="font-size: 0.8rem; color: #888; font-style: italic;">(Scans are stored in-memory during the current browser session)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not scans:
        st.info(
            "📌 **No scans recorded in the current session.**\n\n"
            "Please navigate to **Disease Detection** from the sidebar and upload a plant leaf image. "
            "Each diagnosis will automatically appear here with a dedicated **Print Report** button."
        )
        return

    # --- Actions Row: Export CSV & Clear Session History ---
    act_col1, act_col2 = st.columns([3, 1])

    # Convert records to DataFrame for download
    df_export = pd.DataFrame(
        [
            {
                "Scan ID": s["id"],
                "Timestamp": s["timestamp"],
                "Image": s["filename"],
                "Crop": s["crop"],
                "Condition": s["condition"],
                "Status": "Healthy" if s["is_healthy"] else "Diseased",
                "Confidence (%)": round(s["confidence_percent"], 2),
                "Class Identifier": s["predicted_class"],
                "Notes": s.get("notes", ""),
            }
            for s in scans
        ]
    )

    with act_col1:
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Session Log (CSV)",
            data=csv_data,
            file_name="smart_agriculture_session_scans.csv",
            mime="text/csv",
        )

    with act_col2:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            clear_session_history()
            st.success("Session scan history and associated report data cleared.")
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --- Scan Entries List with Dedicated Print Report Buttons ---
    st.markdown("### 📋 Logged Diagnostic Scans")

    for scan in scans:
        status_badge = "🟢 HEALTHY" if scan["is_healthy"] else "🔴 DISEASED"
        card_header = f"#{scan['id']} • {scan['timestamp']} — {scan['crop']}: {scan['condition']} ({scan['confidence_percent']:.1f}%)"

        with st.expander(card_header, expanded=False):
            detail_left, detail_right = st.columns([3, 2])

            with detail_left:
                st.markdown(f"**Scan ID:** ` {scan['id']} `")
                st.markdown(f"**Image Filename:** `{scan['filename']}`")
                st.markdown(f"**Predicted Class:** `{scan['predicted_class']}`")
                st.markdown(f"**Health Assessment:** {status_badge}")
                st.markdown(
                    f"**Model Confidence:** `{scan['confidence_percent']:.2f}%` *(statistical softmax probability)*"
                )
                if scan.get("notes"):
                    st.info(f"📝 **Field Notes:** {scan['notes']}")

                # MANDATORY: Every individual scan must have its own clearly visible Print Report button
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                if st.button(
                    f"🖨️ Print Report for {scan['id']}",
                    key=f"btn_print_{scan['id']}",
                    type="primary",
                    help="Generate and print a professional A4 diagnostic assessment report for this exact scan.",
                ):
                    st.session_state["active_report_scan_id"] = scan["id"]
                    st.rerun()

            with detail_right:
                if scan.get("image_b64") or scan.get("overlay_b64"):
                    img_sub1, img_sub2 = st.columns(2)
                    if scan.get("image_b64"):
                        with img_sub1:
                            st.image(
                                scan["image_b64"],
                                caption="Analyzed Leaf",
                                use_container_width=True,
                            )
                    if scan.get("overlay_b64"):
                        with img_sub2:
                            st.image(
                                scan["overlay_b64"],
                                caption="Grad-CAM (conv2d_3)",
                                use_container_width=True,
                            )
                else:
                    st.caption("No image data retained for this entry.")
