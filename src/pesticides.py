from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import streamlit as st

from src.config import COLOR_BORDER, COLOR_FOREST_GREEN, COLOR_LIGHT_SAGE, COLOR_SECONDARY_GREEN

VERIFIED_PESTICIDES_PATH = Path("data/verified_pesticides.json")


@st.cache_data(show_spinner=False)
def load_verified_pesticides_catalog() -> Dict[str, Any]:
    """Load the verified commercial pesticide products catalog."""
    if not VERIFIED_PESTICIDES_PATH.is_file():
        return {}
    with open(VERIFIED_PESTICIDES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_verified_products(class_name: str) -> List[Dict[str, Any]]:
    """Retrieve verified commercial products for a given class name."""
    catalog = load_verified_pesticides_catalog()
    class_entry = catalog.get(class_name, {})
    return class_entry.get("products", [])


def get_pesticide_audit_summary() -> Dict[str, Any]:
    """Return verified commercial product count statistics across all 38 classes."""
    catalog = load_verified_pesticides_catalog()
    meta = catalog.get("_metadata", {})
    verified_classes = [k for k in catalog.keys() if not k.startswith("_")]
    
    total_products = sum(len(catalog[k].get("products", [])) for k in verified_classes)
    
    return {
        "classes_with_verified_products": len(verified_classes),
        "total_commercial_products": total_products,
        "classes_requiring_expert_confirmation": 38 - len(verified_classes) - 12,  # 16
        "healthy_classes_no_treatment": 12,
        "total_classes": 38,
        "source_authority": meta.get("source_authority", "PPQS / CIBRC, Ministry of Agriculture & Farmers Welfare, India"),
        "last_audit_date": meta.get("last_audit_date", "2024-10"),
    }


def render_commercial_product_cards(
    class_name: str,
    is_healthy: bool,
    is_ambiguous: bool = False,
    ambiguity_message: str = "",
) -> None:
    """
    Render verified commercial pesticide trade name product cards.
    Handles:
      - Healthy leaves: prominently blocks pesticide recommendations.
      - Ambiguous predictions: safely withholds recommendations.
      - Verified products: displays complete commercial trade names and CIBRC label details.
      - Unverified products: displays the mandatory KVK expert-confirmation notice.
    """
    if is_healthy or "healthy" in class_name.lower():
        st.markdown(
            f"""
            <div style="background: #F1F8E9; border: 1.5px solid #81C784; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1B5E20; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
                    <span>🌱</span> No Chemical Treatment Required
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #2E7D32; margin-bottom: 8px;">
                    Healthy leaf detected — no disease treatment or pesticide is required based on this prediction.
                </div>
                <p style="font-size: 0.88rem; color: #2E7D32; line-height: 1.5; margin-bottom: 0;">
                    Applying chemical fungicides, bactericides, or synthetic pesticides preventively to healthy plants without a verified economic threshold causes unnecessary input cost, risk of phytotoxicity, and environmental harm. Prophylactic pesticide application is strictly not recommended.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if is_ambiguous:
        st.markdown(
            f"""
            <div style="background: #FFF8E1; border: 1.5px solid #FFC107; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #7F5800; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
                    <span>⚠️</span> Commercial Pesticide Recommendations Withheld
                </div>
                <p style="font-size: 0.9rem; color: #5C4000; line-height: 1.5; margin-bottom: 10px;">
                    {ambiguity_message if ambiguity_message else "Because the predicted crop species identity or disease classification is ambiguous, commercial chemical pesticide trade names and dosages are withheld to prevent accidental crop injury or illegal off-label spraying."}
                </p>
                <div style="background: #FFFFFF; border-radius: 8px; padding: 10px 14px; border: 1px solid #FFE082; font-size: 0.85rem; color: #5C4000;">
                    👉 <strong>Action:</strong> Confirm your target crop in the <strong>Target Crop Context</strong> selector above, or consult a local Krishi Vigyan Kendra (KVK) officer for certified physical diagnosis before purchasing chemical treatments.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    products = get_verified_products(class_name)

    if not products:
        # Mandatory Requirement 5.2 statement
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border-left: 5px solid #D97706; border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                <div style="font-size: 1.05rem; font-weight: 700; color: #B45309; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
                    <span>⚠️</span> Verified Indian Trade Name Unavailable
                </div>
                <div style="font-size: 0.92rem; color: #3E544A; line-height: 1.55; margin-bottom: 10px;">
                    <strong>Verified Indian trade name unavailable for this exact crop and pathogen combination.</strong>
                    Commercial product recommendations cannot be made without verified CIBRC registration to prevent off-label pesticide usage or severe crop phytotoxicity.
                </div>
                <div style="background: #FDFBF7; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px 14px; font-size: 0.86rem; color: #78350F;">
                    <strong>Mandatory Agricultural Advisory:</strong> Consult the current registered product label or your local Krishi Vigyan Kendra (KVK) extension scientist before purchasing or applying any chemical treatment. Emphasize cultural sanitation, infected shoot pruning, and resistant varieties.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Render Verified Commercial Product Cards
    st.markdown(
        f"""
        <div style="margin-bottom: 12px; font-size: 0.9rem; color: #2E5A44;">
            <strong>Registered Commercial Formulations:</strong> Found <strong>{len(products)}</strong> CIBRC-registered commercial product(s) in India for this condition:
        </div>
        """,
        unsafe_allow_html=True,
    )

    for p in products:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1.5px solid #A7D3B9; border-left: 6px solid #1E7E34; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 3px 12px rgba(30,126,52,0.06);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 700; color: #1E7E34; text-transform: uppercase; letter-spacing: 0.5px;">
                            COMMERCIAL TRADE NAME (INDIA)
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 800; color: {COLOR_FOREST_GREEN};">
                            {p['trade_name']}
                        </div>
                        <div style="font-size: 0.86rem; color: #556E63;">
                            Manufacturer: <strong>{p['manufacturer']}</strong> • <code>{p['country_registration']}</code>
                        </div>
                    </div>
                    <span style="background: #E8F5E9; color: #1E7E34; border: 1px solid #C8E6C9; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">
                        VERIFIED CIBRC REGISTRATION
                    </span>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; padding: 14px; background: #F8FAF8; border-radius: 8px; margin-bottom: 14px; font-size: 0.88rem; color: #26332D;">
                    <div>
                        <span style="color: #6E887B; font-size: 0.8rem; text-transform: uppercase;">Active Ingredient(s)</span><br>
                        <strong>{p['active_ingredients']}</strong> ({p['concentrations']})
                    </div>
                    <div>
                        <span style="color: #6E887B; font-size: 0.8rem; text-transform: uppercase;">Formulation Type</span><br>
                        <strong>{p['formulation']}</strong>
                    </div>
                    <div>
                        <span style="color: #6E887B; font-size: 0.8rem; text-transform: uppercase;">Approved Crop & Target</span><br>
                        <strong>{p['approved_crop']}</strong> • {p['approved_target']}
                    </div>
                    <div>
                        <span style="color: #6E887B; font-size: 0.8rem; text-transform: uppercase;">Label Application Dosage</span><br>
                        <code style="color: #173F35; font-weight: 700; background: #E8F2E8;">{p['dosage']}</code>
                    </div>
                </div>

                <div style="font-size: 0.87rem; color: #3E544A; line-height: 1.5; margin-bottom: 12px;">
                    <strong>Application Protocol:</strong> {p['application_method']}
                </div>

                <div style="display: flex; gap: 20px; font-size: 0.84rem; color: #4A6357; margin-bottom: 12px; flex-wrap: wrap;">
                    <div>⏱️ <strong>Pre-Harvest Interval (PHI):</strong> {p['pre_harvest_interval']}</div>
                    <div>🚪 <strong>Re-Entry Interval (REI):</strong> {p['re_entry_interval']}</div>
                    <div>🛡️ <strong>Safety Gear (PPE):</strong> {p['safety_instructions']}</div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #E8ECE8; pt: 10px; padding-top: 10px; font-size: 0.8rem; color: #6E887B; flex-wrap: wrap; gap: 8px;">
                    <div>Verified Source: <a href="{p['official_label_url']}" target="_blank" style="color: #2F6B4F; text-decoration: underline;">CIBRC Official Portal</a> • Last Audit: {p['last_verified']}</div>
                    <div style="font-style: italic;">Always read the official printed container label before use.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
