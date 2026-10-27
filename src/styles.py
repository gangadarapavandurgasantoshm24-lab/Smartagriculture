import streamlit as st
from src.config import (
    COLOR_ACCENT_GREEN,
    COLOR_BORDER,
    COLOR_FOREST_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_SECONDARY_GREEN,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_WARM_OFFWHITE,
)

CUSTOM_CSS = f"""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset & Typography */
html, body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {COLOR_TEXT_PRIMARY};
}}

/* Headings in Main View - High Contrast Forest Green */
h1, h2, h3, h4, h5, h6,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3,
[data-testid="stHeadingWithActionElements"] h4,
[data-testid="stHeadingWithActionElements"] h5,
[data-testid="stHeadingWithActionElements"] h6,
.stApp [data-testid="stMainBlockContainer"] h1,
.stApp [data-testid="stMainBlockContainer"] h2,
.stApp [data-testid="stMainBlockContainer"] h3,
.stApp [data-testid="stMainBlockContainer"] h4,
.stApp [data-testid="stMainBlockContainer"] h5,
.stApp [data-testid="stMainBlockContainer"] h6 {{
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    color: {COLOR_FOREST_GREEN} !important;
    letter-spacing: -0.02em;
}}

/* App Container & Theme Variables */
.stApp {{
    background-color: {COLOR_WARM_OFFWHITE};
    color: {COLOR_TEXT_PRIMARY};
    --text-color: {COLOR_TEXT_PRIMARY};
}}

/* Ensure High-Contrast Text on Light Backgrounds strictly in Main View */
.stApp [data-testid="stMainBlockContainer"],
.stApp [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] p:not(.hero-title):not(.hero-subtitle),
.stApp [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] span:not(.badge-healthy):not(.badge-diseased):not(.badge-crop):not(.badge-verified):not(.badge-vector):not(.badge-expert):not(.badge-inappropriate):not(.sidebar-brand-badge):not(.hero-card span),
.stApp [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] label,
.stApp [data-testid="stMainBlockContainer"] p:not(.hero-title):not(.hero-subtitle),
.stApp [data-testid="stMainBlockContainer"] label,
.stApp [data-testid="stMainBlockContainer"] .stMarkdown p:not(.hero-title):not(.hero-subtitle) {{
    color: {COLOR_TEXT_PRIMARY} !important;
}}

/* Lists & Nested List Items in Main View */
.stApp [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] ul,
.stApp [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] ol,
.stApp [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMainBlockContainer"] .stMarkdown ul,
.stApp [data-testid="stMainBlockContainer"] .stMarkdown ol,
.stApp [data-testid="stMainBlockContainer"] .stMarkdown li {{
    color: {COLOR_TEXT_PRIMARY} !important;
}}

.stApp [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] li::marker,
.stApp [data-testid="stMainBlockContainer"] .stMarkdown li::marker {{
    color: {COLOR_SECONDARY_GREEN} !important;
}}

/* KaTeX Mathematical Formulas High Contrast */
.stApp .katex,
.stApp .katex-html,
.stApp .katex-display,
.stApp .katex *,
.stApp .katex .mord,
.stApp .katex .mbin,
.stApp .katex .mrel,
.stApp .katex .mopen,
.stApp .katex .mclose,
.stApp .katex .mpunct,
.stApp .katex .mop,
.stApp .katex .minner,
.stApp .katex .vlist-t,
.stApp .katex .vlist-r,
.stApp .katex .frac-line {{
    color: {COLOR_TEXT_PRIMARY} !important;
    border-color: {COLOR_TEXT_PRIMARY} !important;
}}

/* Inline Code High Contrast */
.stApp [data-testid="stMarkdownContainer"] code:not(section[data-testid="stSidebar"] code),
.stApp code:not(section[data-testid="stSidebar"] code) {{
    color: {COLOR_FOREST_GREEN} !important;
    background-color: #E2ECE5 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.88em !important;
    border: 1px solid #C4D9CD !important;
    font-weight: 500 !important;
}}

/* Expander Content & Body Styling (Preserving Header) */
[data-testid="stExpander"] details {{
    border-radius: 12px;
}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] li,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] div {{
    color: {COLOR_TEXT_PRIMARY} !important;
}}

/* Sidebar Styling & Dark Background High-Contrast Rules */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLOR_FOREST_GREEN} 0%, #0B221B 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #FFFFFF !important;
}}

/* Universal High Contrast for all elements inside Sidebar */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {{
    color: #FFFFFF !important;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
}}

section[data-testid="stSidebar"] hr {{
    border-color: rgba(255, 255, 255, 0.22) !important;
}}

section[data-testid="stSidebar"] code {{
    color: #FFFFFF !important;
    background-color: rgba(255, 255, 255, 0.16) !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    font-weight: 600 !important;
}}

/* Custom Sidebar Brand Card */
.sidebar-brand {{
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 14px;
    padding: 18px 16px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}}
.sidebar-brand-title {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}}
.sidebar-brand-badge {{
    display: inline-block;
    background: {COLOR_ACCENT_GREEN};
    color: #FFFFFF !important;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.sidebar-brand-subtitle {{
    font-size: 0.82rem;
    color: #CFE3D9 !important;
    margin-top: 6px;
    line-height: 1.4;
}}

/* Navigation Radio Styling in Sidebar */
section[data-testid="stSidebar"] div[role="radiogroup"] > label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover,
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: rgba(255, 255, 255, 0.18) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
    transform: translateX(4px) !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label span,
section[data-testid="stSidebar"] div[role="radiogroup"] > label div,
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span {{
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover span {{
    color: #FFFFFF !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
    background: {COLOR_SECONDARY_GREEN} !important;
    border: 1.5px solid #82E0AA !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] span,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) span,
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span {{
    color: #FFFFFF !important;
    font-weight: 800 !important;
}}

/* Main Hero Banner Card (Dark Green Background with Crisp White Text) */
.hero-card {{
    background: linear-gradient(135deg, {COLOR_FOREST_GREEN} 0%, {COLOR_SECONDARY_GREEN} 100%);
    border-radius: 18px;
    padding: 32px 36px;
    color: #FFFFFF !important;
    box-shadow: 0 10px 25px -5px rgba(23, 63, 53, 0.25);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}}
.hero-card::after {{
    content: "🌿";
    position: absolute;
    right: 20px;
    bottom: -15px;
    font-size: 110px;
    opacity: 0.12;
    pointer-events: none;
}}
.hero-title {{
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin-bottom: 8px !important;
    letter-spacing: -0.02em !important;
}}
.hero-subtitle {{
    font-size: 1.05rem !important;
    color: #F0F7F4 !important;
    max-width: 750px !important;
    line-height: 1.5 !important;
}}
.hero-card h1,
.hero-card h2,
.hero-card h3,
.hero-card h4,
.hero-card p,
.hero-card span,
.hero-card div {{
    color: #FFFFFF !important;
}}

/* Refined Stat Cards */
.stat-card {{
    background: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    transition: all 0.25s ease;
    height: 100%;
}}
.stat-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(23, 63, 53, 0.08);
    border-color: {COLOR_SECONDARY_GREEN};
}}
.stat-label {{
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLOR_FOREST_GREEN} !important;
    margin-bottom: 6px;
}}
.stat-value {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    color: {COLOR_FOREST_GREEN} !important;
    line-height: 1.2;
}}
.stat-desc {{
    font-size: 0.8rem;
    font-weight: 600;
    color: #26332D !important;
    margin-top: 6px;
}}

/* Feature Card */
.feature-card {{
    background: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    margin-bottom: 20px;
}}
.feature-header {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: {COLOR_FOREST_GREEN} !important;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* Disease Tag Badges */
.badge-healthy {{
    background-color: #E6F7EC !important;
    color: #197A3E !important;
    border: 1px solid #B8E8C7 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    display: inline-block !important;
}}
.badge-diseased {{
    background-color: #FDF2F2 !important;
    color: #D9383A !important;
    border: 1px solid #FAC7C7 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    display: inline-block !important;
}}
.badge-crop {{
    background-color: {COLOR_LIGHT_SAGE} !important;
    color: {COLOR_FOREST_GREEN} !important;
    border: 1px solid {COLOR_BORDER} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    display: inline-block !important;
}}
.badge-verified {{
    background-color: #EDF7F2 !important;
    color: #1E7E34 !important;
    border: 1px solid #B7E4C7 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    display: inline-block !important;
}}
.badge-vector {{
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
    border: 1px solid #BFDBFE !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    display: inline-block !important;
}}
.badge-expert {{
    background-color: #FFFBEB !important;
    color: #B45309 !important;
    border: 1px solid #FDE68A !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    display: inline-block !important;
}}
.badge-inappropriate {{
    background-color: #F3F4F6 !important;
    color: #4B5563 !important;
    border: 1px solid #E5E7EB !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 4px 10px !important;
    border-radius: 20px !important;
    display: inline-block !important;
}}
.guidance-card {{
    background: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}}
.guidance-callout {{
    background: {COLOR_LIGHT_SAGE};
    border-left: 4px solid {COLOR_SECONDARY_GREEN};
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: {COLOR_TEXT_PRIMARY};
}}
.guidance-callout-warning {{
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #78350F;
}}
.guidance-callout-danger {{
    background: #FDF2F2;
    border-left: 4px solid #D9383A;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #7F1D1D;
}}
.guidance-callout-info {{
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #1E3A8A;
}}

/* Primary & Secondary Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {COLOR_FOREST_GREEN} 0%, {COLOR_SECONDARY_GREEN} 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 12px rgba(23, 63, 53, 0.2) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(23, 63, 53, 0.3) !important;
    color: #FFFFFF !important;
}}

.stButton > button[kind="secondary"] {{
    background: #FFFFFF !important;
    color: {COLOR_FOREST_GREEN} !important;
    border: 1px solid {COLOR_BORDER} !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background-color: rgba(232, 242, 232, 0.5);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid {COLOR_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    padding: 8px 18px;
    color: {COLOR_TEXT_MUTED} !important;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    background-color: #FFFFFF !important;
    color: {COLOR_FOREST_GREEN} !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}}

/* Streamlit Alert boxes refinement */
div[data-testid="stAlert"] {{
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.06);
}}

/* File Uploader Refinement */
div[data-testid="stFileUploader"] {{
    background: #FFFFFF;
    border: 2px dashed {COLOR_BORDER};
    border-radius: 16px;
    padding: 18px;
    transition: border-color 0.2s ease;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: {COLOR_SECONDARY_GREEN};
}}

/* Footer */
.app-footer {{
    text-align: center;
    padding: 24px 0 12px 0;
    color: {COLOR_TEXT_MUTED};
    font-size: 0.8rem;
    border-top: 1px solid {COLOR_BORDER};
    margin-top: 40px;
}}
</style>
"""


def apply_custom_styles():
    """Inject the unified design system CSS into the current Streamlit app page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar_header():
    """Render high-end branded header in the sidebar."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                <span>🌿</span> SmartAgri AI
            </div>
            <span class="sidebar-brand-badge">CNN • 38 Classes</span>
            <div class="sidebar-brand-subtitle">
                Plant Leaf Disease Diagnostics & Grad-CAM Visual Explainability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badge: str = "PRODUCTION READY"):
    """Render the standard page hero banner."""
    st.markdown(
        f"""
        <div class="hero-card">
            <span style="background: rgba(255,255,255,0.2); font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; letter-spacing: 0.05em;">{badge}</span>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Render professional app footer."""
    st.markdown(
        """
        <div class="app-footer">
            <strong>SmartAgri AI Diagnostic System</strong> • Powered by TensorFlow & Keras 3 • Architectural Model: <code>smart_agriculture_cnn</code> (conv2d_3 target)<br>
            For agricultural research & advisory assistance. Always verify high-stakes treatment decisions with certified plant pathologists.
        </div>
        """,
        unsafe_allow_html=True,
    )
