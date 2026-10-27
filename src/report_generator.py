import datetime
from typing import Any, Dict, Optional

from src.guidance import get_guidance
from src.pesticides import get_verified_products


def generate_printable_html_report(scan: Dict[str, Any]) -> str:
    """
    Generate a standalone, professional, clean white A4-friendly HTML diagnostic report
    for the exact selected scan record.
    Includes print stylesheet (@media print) and automated print trigger.
    """
    if not scan:
        return "<p>Error: No scan data provided for report generation.</p>"

    # 1. Header & Identity
    scan_id = scan.get("id", "UNKNOWN-ID")
    scan_time = scan.get("timestamp", "N/A")
    report_gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = scan.get("filename", "leaf_scan.jpg")
    crop = scan.get("crop", "Unknown")
    condition = scan.get("condition", "Unknown")
    is_healthy = scan.get("is_healthy", False)
    predicted_class = scan.get("predicted_class", "Unknown")
    conf_pct = scan.get("confidence_percent", 0.0)
    top_preds = scan.get("top_predictions", [])
    notes = scan.get("notes", "")

    # Status styling
    status_text = "HEALTHY FOLIAGE" if is_healthy else "DISEASED / PATHOGEN DETECTED"
    status_bg = "#E8F5E9" if is_healthy else "#FFEBEE"
    status_color = "#1B5E20" if is_healthy else "#B71C1C"

    # 2. Guidance & Product Details
    guidance = scan.get("guidance_snapshot") or get_guidance(predicted_class) or {}
    products = get_verified_products(predicted_class) if not is_healthy else []

    # 3. Build Top Predictions Rows
    top_rows_html = ""
    for p in top_preds[:3]:
        p_name = p.get("condition") or p.get("class_name", "")
        p_pct = p.get("confidence_percent", 0.0)
        p_crop = p.get("crop", "")
        p_rank = p.get("rank", 1)
        top_rows_html += f"""
        <tr>
            <td style="padding: 6px 10px; border: 1px solid #D0D8D0; text-align: center;">#{p_rank}</td>
            <td style="padding: 6px 10px; border: 1px solid #D0D8D0;"><strong>{p_name}</strong> ({p_crop})</td>
            <td style="padding: 6px 10px; border: 1px solid #D0D8D0; text-align: right; font-weight: bold; color: #173F35;">{p_pct:.2f}%</td>
        </tr>
        """

    # 4. Images & Explainability Section
    img_b64 = scan.get("image_b64")
    overlay_b64 = scan.get("overlay_b64")

    if img_b64 and overlay_b64:
        images_html = f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 10px; margin-bottom: 8px;">
            <div style="flex: 1; text-align: center; border: 1px solid #D0D8D0; padding: 10px; border-radius: 6px; background: #FAFCFA;">
                <img src="{img_b64}" style="max-width: 100%; max-height: 220px; object-fit: contain; border-radius: 4px;" alt="Analyzed Leaf">
                <div style="font-size: 11px; color: #556E63; margin-top: 6px; font-weight: bold;">Original Leaf Photo ({filename})</div>
            </div>
            <div style="flex: 1; text-align: center; border: 1px solid #D0D8D0; padding: 10px; border-radius: 6px; background: #FAFCFA;">
                <img src="{overlay_b64}" style="max-width: 100%; max-height: 220px; object-fit: contain; border-radius: 4px;" alt="Grad-CAM Overlay">
                <div style="font-size: 11px; color: #556E63; margin-top: 6px; font-weight: bold;">Grad-CAM Layer Attention (conv2d_3)</div>
            </div>
        </div>
        """
    elif img_b64:
        images_html = f"""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 8px; border: 1px solid #D0D8D0; padding: 10px; border-radius: 6px; background: #FAFCFA;">
            <img src="{img_b64}" style="max-width: 250px; max-height: 220px; object-fit: contain; border-radius: 4px;" alt="Analyzed Leaf">
            <div style="font-size: 11px; color: #556E63; margin-top: 6px; font-weight: bold;">Original Leaf Photo ({filename})</div>
            <div style="font-size: 11px; color: #888; font-style: italic; margin-top: 4px;">Grad-CAM visualization was unavailable for this scan.</div>
        </div>
        """
    else:
        images_html = """
        <div style="border: 1px dashed #D0D8D0; padding: 16px; border-radius: 6px; text-align: center; background: #FAFCFA; color: #777; font-size: 12px; margin-top: 8px; margin-bottom: 8px;">
            Leaf photograph and Grad-CAM overlay are unavailable for this session entry. Diagnostic numerical metrics and treatment guidance are provided below.
        </div>
        """

    # 5. Disease Guidance & Commercial Pesticide Section
    if is_healthy:
        guidance_section_html = f"""
        <div style="background: #F1F8E9; border: 2px solid #81C784; border-radius: 8px; padding: 14px 18px; margin-top: 12px; margin-bottom: 14px;">
            <div style="font-size: 15px; font-weight: bold; color: #1B5E20; margin-bottom: 6px;">
                🌱 Healthy Leaf Detected — No Disease Treatment or Pesticide is Required Based on this Prediction
            </div>
            <p style="font-size: 12px; color: #2E7D32; line-height: 1.5; margin: 0;">
                The submitted foliage displays normal vigor without diagnostic necrotic or chlorotic lesions. 
                Applying chemical fungicides or insecticides preventively to healthy crops without a justified economic threshold 
                is economically wasteful, risks chemical phytotoxicity, and is environmentally detrimental. Prophylactic pesticide application is strictly not advised.
            </p>
        </div>

        <div style="border: 1px solid #D0D8D0; border-radius: 6px; padding: 12px 16px; margin-bottom: 14px; background: #FFFFFF;">
            <h4 style="margin: 0 0 8px 0; color: #173F35; font-size: 13px;">🛡️ Routine Maintenance & Plant Health Practices</h4>
            <ul style="font-size: 12px; color: #2E453A; line-height: 1.6; margin: 0; padding-left: 18px;">
                <li>Maintain balanced N-P-K soil fertility; avoid excessive nitrogen that generates succulent, disease-prone tissue.</li>
                <li>Utilize ground drip irrigation; avoid overhead sprinkler wetting during twilight to prevent overnight fungal sporulation.</li>
                <li>Conduct routine weekly field scouting across lower and inner canopies.</li>
                <li>Maintain clean orchard borders and eliminate alternate weed hosts.</li>
            </ul>
        </div>
        """
    else:
        # Diseased Foliage Guidance
        symptoms_text = guidance.get("plain_language_symptoms", "No specific symptoms reported in database.")
        prev_measures = guidance.get("preventive_measures", [])
        if isinstance(prev_measures, list):
            prev_html = "".join(f"<li>{m}</li>" for m in prev_measures)
        else:
            prev_html = f"<li>{prev_measures}</li>"

        mgmt_measures = guidance.get("immediate_non_chemical_management", [])
        if isinstance(mgmt_measures, list):
            mgmt_html = "".join(f"<li>{m}</li>" for m in mgmt_measures)
        else:
            mgmt_html = f"<li>{mgmt_measures}</li>"

        # Commercial Products Cards HTML
        if products:
            prod_cards_html = ""
            for p in products:
                prod_cards_html += f"""
                <div style="border: 1.5px solid #A7D3B9; border-left: 5px solid #1E7E34; border-radius: 6px; padding: 10px 14px; margin-top: 8px; margin-bottom: 10px; background: #FAFCFA;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 14px; font-weight: bold; color: #173F35;">Trade Name: {p['trade_name']}</span>
                        <span style="font-size: 11px; background: #E8F5E9; color: #1E7E34; padding: 2px 6px; border-radius: 4px; font-weight: bold;">CIBRC VERIFIED</span>
                    </div>
                    <div style="font-size: 11px; color: #556E63; margin-bottom: 6px;">
                        Active Ingredient: <strong>{p['active_ingredients']} ({p['concentrations']})</strong> • Formulation: <strong>{p['formulation']}</strong> • Manufacturer: {p['manufacturer']}
                    </div>
                    <div style="font-size: 11px; color: #173F35; margin-bottom: 4px; background: #EBF3ED; padding: 4px 8px; border-radius: 4px;">
                        <strong>Dosage:</strong> <code>{p['dosage']}</code> | <strong>Water Volume:</strong> {p['water_volume']}
                    </div>
                    <div style="font-size: 11px; color: #444; line-height: 1.4; margin-bottom: 4px;">
                        <strong>Application Method:</strong> {p['application_method']}
                    </div>
                    <div style="font-size: 10.5px; color: #555; display: flex; gap: 15px; margin-top: 4px;">
                        <span>⏱️ <strong>PHI:</strong> {p['pre_harvest_interval']}</span>
                        <span>🚪 <strong>REI:</strong> {p['re_entry_interval']}</span>
                        <span>🛡️ <strong>Safety Gear:</strong> {p['safety_instructions']}</span>
                    </div>
                </div>
                """
            pesticide_block_html = f"""
            <div style="margin-top: 12px; margin-bottom: 12px;">
                <h4 style="margin: 0 0 6px 0; color: #173F35; font-size: 13px;">🧪 Verified Commercial Pesticide Options (India CIBRC Approved)</h4>
                {prod_cards_html}
            </div>
            """
        else:
            pesticide_block_html = """
            <div style="background: #FFFDF7; border: 1.5px solid #FDE68A; border-left: 5px solid #D97706; border-radius: 6px; padding: 10px 14px; margin-top: 12px; margin-bottom: 12px;">
                <div style="font-size: 12px; font-weight: bold; color: #B45309; margin-bottom: 4px;">
                    ⚠️ Verified Indian Trade Name Unavailable / Requires Expert Confirmation
                </div>
                <p style="font-size: 11.5px; color: #78350F; margin: 0; line-height: 1.5;">
                    A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer. 
                    Commercial brand names and dosages are withheld to prevent off-label spraying or crop phytotoxicity.
                </p>
            </div>
            """

        guidance_section_html = f"""
        <div style="border: 1px solid #D0D8D0; border-radius: 6px; padding: 12px 16px; margin-top: 12px; margin-bottom: 12px; background: #FFFFFF;">
            <h4 style="margin: 0 0 6px 0; color: #173F35; font-size: 13px;">🔍 Visual Symptoms & Pathological Signs</h4>
            <p style="font-size: 12px; color: #2E453A; line-height: 1.5; margin: 0 0 10px 0;">{symptoms_text}</p>
            
            <h4 style="margin: 0 0 6px 0; color: #173F35; font-size: 13px;">⚡ Immediate Cultural & Sanitation Management</h4>
            <ul style="font-size: 12px; color: #2E453A; line-height: 1.5; margin: 0 0 10px 0; padding-left: 18px;">
                {mgmt_html}
            </ul>

            <h4 style="margin: 0 0 6px 0; color: #173F35; font-size: 13px;">🛡️ Preventive Agronomic Measures</h4>
            <ul style="font-size: 12px; color: #2E453A; line-height: 1.5; margin: 0; padding-left: 18px;">
                {prev_html}
            </ul>
        </div>
        {pesticide_block_html}
        """

    # 6. Assemble Full Standalone HTML with Print CSS
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plant Disease Assessment Report — {scan_id}</title>
<style>
    @page {{
        size: A4 portrait;
        margin: 15mm 15mm 15mm 15mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #FFFFFF;
        color: #26332D;
        line-height: 1.45;
        font-size: 12px;
        margin: 0;
        padding: 20px;
    }}
    .report-container {{
        max-width: 820px;
        margin: 0 auto;
        background: #FFFFFF;
    }}
    .print-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #173F35;
        color: #FFFFFF;
        padding: 10px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
    }}
    .print-btn {{
        background: #2F6B4F;
        color: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.4);
        padding: 8px 18px;
        font-size: 13px;
        font-weight: bold;
        border-radius: 6px;
        cursor: pointer;
    }}
    .print-btn:hover {{
        background: #1F4B37;
    }}
    .header-table {{
        width: 100%;
        border-collapse: collapse;
        border-bottom: 2px solid #173F35;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }}
    .header-logo {{
        font-size: 20px;
        font-weight: 800;
        color: #173F35;
        letter-spacing: -0.5px;
    }}
    .header-sub {{
        font-size: 11px;
        color: #556E63;
        margin-top: 2px;
    }}
    .meta-box {{
        background: #F8FAF8;
        border: 1px solid #D0D8D0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-size: 11.5px;
    }}
    .meta-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
    }}
    .assessment-card {{
        border: 1.5px solid #D0D8D0;
        border-radius: 6px;
        padding: 14px 18px;
        background: #FFFFFF;
        margin-bottom: 14px;
    }}
    .assessment-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }}
    .assessment-disease {{
        font-size: 20px;
        font-weight: 800;
        color: #173F35;
    }}
    .table-top {{
        width: 100%;
        border-collapse: collapse;
        font-size: 11.5px;
        margin-top: 8px;
    }}
    .table-top th {{
        background: #EBF2EC;
        padding: 6px 10px;
        border: 1px solid #D0D8D0;
        text-align: left;
        color: #173F35;
    }}
    .disclaimer-box {{
        background: #F8FAF8;
        border: 1px solid #D0D8D0;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 10.5px;
        color: #556E63;
        line-height: 1.45;
        margin-top: 14px;
    }}
    @media print {{
        body {{
            padding: 0;
            margin: 0;
        }}
        .print-bar {{
            display: none !important;
        }}
        .report-container {{
            max-width: 100%;
        }}
    }}
</style>
</head>
<body>
<div class="report-container">
    <div class="print-bar">
        <span><strong>SmartAgri AI Diagnostic System</strong> — Clinical Plant Pathology Assessment</span>
        <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
    </div>

    <table class="header-table">
        <tr>
            <td>
                <div class="header-logo">🌿 SmartAgri AI — Plant Disease Diagnostics</div>
                <div class="header-sub">Automated Leaf Pathology Screening & Explainable AI Verification</div>
            </td>
            <td style="text-align: right; vertical-align: bottom;">
                <div style="font-size: 13px; font-weight: bold; color: #173F35;">PLANT LEAF ASSESSMENT REPORT</div>
                <div style="font-size: 11px; color: #6E887B;">Document Ref: <code>{scan_id}</code></div>
            </td>
        </tr>
    </table>

    <div class="meta-box">
        <div class="meta-grid">
            <div><strong>Scan Record ID:</strong> {scan_id}</div>
            <div><strong>Scan Timestamp:</strong> {scan_time}</div>
            <div><strong>Report Generated:</strong> {report_gen_time}</div>
            <div><strong>Image Filename:</strong> {filename}</div>
            <div><strong>Target Crop:</strong> {crop}</div>
            <div><strong>Architecture:</strong> CNN (38 Classes / 14 Crops)</div>
        </div>
        {f'<div style="margin-top: 6px; padding-top: 6px; border-top: 1px dashed #D0D8D0;"><strong>Field Notes:</strong> {notes}</div>' if notes else ''}
    </div>

    <div class="assessment-card">
        <div class="assessment-header">
            <span style="background: {status_bg}; color: {status_color}; font-weight: bold; font-size: 11px; padding: 4px 10px; border-radius: 4px; border: 1px solid {status_color};">
                {status_text}
            </span>
            <span style="font-size: 13px; color: #556E63;">
                Confidence: <strong style="font-size: 16px; color: #173F35;">{conf_pct:.2f}%</strong>
            </span>
        </div>
        <div class="assessment-disease">{crop}: {condition}</div>
        <div style="font-size: 11.5px; color: #556E63; margin-top: 2px;">
            Dataset Class Identifier: <code>{predicted_class}</code>
        </div>
    </div>

    <!-- Candidate Probability Distribution Table -->
    <div style="margin-bottom: 14px;">
        <div style="font-size: 12px; font-weight: bold; color: #173F35; margin-bottom: 4px;">Top Candidate Predictions (Softmax Distribution):</div>
        <table class="table-top">
            <thead>
                <tr>
                    <th style="width: 10%;">Rank</th>
                    <th>Predicted Condition & Crop</th>
                    <th style="width: 25%; text-align: right;">Confidence</th>
                </tr>
            </thead>
            <tbody>
                {top_rows_html}
            </tbody>
        </table>
    </div>

    <!-- Visual Attention Section -->
    <div style="margin-top: 10px; margin-bottom: 14px;">
        <div style="font-size: 12px; font-weight: bold; color: #173F35;">Visual Attention & Gradient Heatmap (Layer conv2d_3):</div>
        {images_html}
        <div style="font-size: 10.5px; color: #556E63; font-style: italic; text-align: center; margin-top: 4px;">
            Note: Grad-CAM attention highlights spatial features in layer conv2d_3 that influenced network activation. Heatmap concentration indicates correlation with trained visual motifs and does not constitute independent clinical proof of pathogen infection.
        </div>
    </div>

    <!-- Detailed Agronomic Guidance Section -->
    {guidance_section_html}

    <!-- Regulatory Disclaimer -->
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Agricultural Disclaimer & Advisory Notice:</strong><br>
        1. <strong>Screening Support:</strong> This assessment is generated by machine vision trained on the PlantVillage laboratory dataset (38 classes across 14 crops). High statistical confidence reflects visual correlation with trained patterns and is not a biological guarantee.<br>
        2. <strong>Regulatory Compliance:</strong> Chemical recommendations cite Central Insecticides Board & Registration Committee (CIBRC) approved uses in India. Always read the current container label for statutory local regulations, state-specific restrictions, and pre-harvest intervals.<br>
        3. <strong>Expert Physical Verification:</strong> Before taking costly or irreversible chemical actions, verify diagnosis with a local certified Krishi Vigyan Kendra (KVK) scientist or state agricultural extension officer.
    </div>

    <div style="text-align: center; margin-top: 16px; font-size: 10px; color: #888; border-top: 1px solid #E0E0E0; padding-top: 8px;">
        SmartAgri AI System • Official Agricultural Diagnostics Framework • Page 1 of 1
    </div>
</div>
</body>
</html>
"""
    return html
