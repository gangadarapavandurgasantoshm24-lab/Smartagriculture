import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Suppress verbose TensorFlow C++ messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st

from pages_impl.dashboard import render_dashboard
from pages_impl.detection import render_detection
from pages_impl.explainability import render_explainability
from pages_impl.history import render_history
from pages_impl.settings import render_settings
from src.styles import apply_custom_styles, render_footer, render_sidebar_header


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="SmartAgri - Plant Disease AI",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Inject Design System Styling
    apply_custom_styles()

    # 3. Sidebar Navigation & Branding
    render_sidebar_header()

    # EXACTLY FIVE NAVIGATION PAGES (Requirement 2)
    nav_pages = [
        "Dashboard",
        "Disease Detection",
        "AI Explainability",
        "Scan History",
        "Settings/About",
    ]

    # Initialize / validate navigation state
    if (
        "nav_selection" not in st.session_state
        or st.session_state["nav_selection"] not in nav_pages
    ):
        st.session_state["nav_selection"] = "Dashboard"

    # Map icons to page names
    nav_icons = {
        "Dashboard": "🏠 Dashboard",
        "Disease Detection": "🔍 Disease Detection",
        "AI Explainability": "👁️ AI Explainability",
        "Scan History": "🕒 Scan History",
        "Settings/About": "⚙️ Settings/About",
    }

    display_options = [nav_icons[p] for p in nav_pages]
    current_display = nav_icons[st.session_state["nav_selection"]]
    current_index = display_options.index(current_display)

    selected_display = st.sidebar.radio(
        "Navigation",
        options=display_options,
        index=current_index,
        label_visibility="collapsed",
    )

    # Reverse lookup page key
    selected_page = [k for k, v in nav_icons.items() if v == selected_display][0]
    st.session_state["nav_selection"] = selected_page

    # Sidebar Quick Info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.82rem; color: #FFFFFF !important; line-height: 1.6; padding: 10px 12px; background: rgba(255,255,255,0.08); border-radius: 10px; border: 1px solid rgba(255,255,255,0.18);">
            <strong style="color: #FFFFFF !important; font-size: 0.86rem;">System Status:</strong><br>
            <span style="color: #FFFFFF !important;">• Model: <code style="color: #FFFFFF !important; background: rgba(255,255,255,0.16) !important;">best_model.keras</code></span><br>
            <span style="color: #FFFFFF !important;">• Classes: 38 (14 crops)</span><br>
            <span style="color: #FFFFFF !important;">• Grad-CAM: <code style="color: #FFFFFF !important; background: rgba(255,255,255,0.16) !important;">conv2d_3</code> active</span><br>
            <span style="color: #FFFFFF !important;">• Engine: TensorFlow / Keras 3</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Dispatch to Page Implementations (Exactly 5 Pages)
    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "Disease Detection":
        render_detection()
    elif selected_page == "AI Explainability":
        render_explainability()
    elif selected_page == "Scan History":
        render_history()
    elif selected_page == "Settings/About":
        render_settings()

    # 5. Global Footer
    render_footer()



if __name__ == "__main__":
    main()
