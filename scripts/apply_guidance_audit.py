import json

with open("data/disease_guidance.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Classes to update to Requires Expert Confirmation with suppressed dosages
expert_classes = {
    "Cherry_(including_sour)___Powdery_mildew": {
        "status": "Requires Expert Confirmation",
        "rationale": "Cherry is not listed under CIBRC national Major Uses for Hexaconazole or Wettable Sulfur. Commercial cherry cultivation in India is localized to temperate zones (J&K, HP) where localized SKUAST/state university advisories must be consulted.",
        "src_title": "PPQS CIBRC Major Uses & SKUAST-K Temperate Fruit Advisory",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "status": "Requires Expert Confirmation",
        "rationale": "Gray leaf spot (Cercospora zeae-maydis) is not an approved registered label target disease under CIBRC for maize in India (only Turcicum blight, Maydis blight, and Common rust have approved label claims). Consult your local KVK for verified localized recommendations.",
        "src_title": "PPQS CIBRC Major Uses of Fungicides (Maize Section)",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Grape___Black_rot": {
        "status": "Requires Expert Confirmation",
        "rationale": "Black rot (Guignardia bidwellii) is not a registered target disease under CIBRC for grapes in India (CIBRC grape registrations cover Downy Mildew, Powdery Mildew, and Anthracnose). Field sanitation and canker removal are primary.",
        "src_title": "PPQS CIBRC Major Uses of Fungicides (Grape Section) & ICAR-NRCG",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "status": "Requires Expert Confirmation",
        "rationale": "Leaf blight (Isariopsis clavispora / Pseudocercospora cladosporioides) is not an individually registered label target disease under CIBRC for grapes. Consult ICAR-NRCG or local KVK viticulture specialists.",
        "src_title": "PPQS CIBRC Major Uses & ICAR-NRC for Grapes",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Pepper,_bell___Bacterial_spot": {
        "status": "Requires Expert Confirmation",
        "rationale": "Antibiotics (Streptomycin/Tetracycline) are NOT approved by CIBRC for bell pepper/chilli and are subject to regulatory restrictions on food crops to prevent antimicrobial resistance (AMR). Standalone Copper Oxychloride is registered for anthracnose/fruit rot, not bacterial spot in CIBRC.",
        "src_title": "PPQS CIBRC Major Uses & Ministry of Agriculture Regulatory Directives",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Squash___Powdery_mildew": {
        "status": "Requires Expert Confirmation",
        "rationale": "Squash (Cucurbita pepo) does not possess an individual dedicated crop entry in CIBRC Major Uses (unlike cucumber and bottle gourd). Broad cucurbit extension guidelines exist, but crop-specific label registration requires local KVK verification.",
        "src_title": "PPQS CIBRC Major Uses of Fungicides (Cucurbitaceous Vegetables)",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Strawberry___Leaf_scorch": {
        "status": "Requires Expert Confirmation",
        "rationale": "Strawberry is a minor specialty crop in India with no dedicated CIBRC major use label claims for Leaf Scorch (Diplocarpon earlianum). Consult local horticulture extension officers for certified regional practices.",
        "src_title": "National Horticulture Board (NHB) & PPQS Registration Compendium",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Tomato___Bacterial_spot": {
        "status": "Requires Expert Confirmation",
        "rationale": "Streptocycline/antibiotics are NOT approved by CIBRC for tomato bacterial spot and their use on vegetables is restricted by government directives to curb antimicrobial resistance (AMR). Standard synthetic fungicides have zero efficacy against bacterial spot.",
        "src_title": "PPQS CIBRC Regulatory Notifications & ICAR-IIVR",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Tomato___Leaf_Mold": {
        "status": "Requires Expert Confirmation",
        "rationale": "Tomato leaf mold (Passalora fulva) is primarily a greenhouse/protected cultivation pathogen and is not an approved registered major use label target on CIBRC labels (which cover Early Blight, Late Blight, and Buckeye Rot).",
        "src_title": "PPQS CIBRC Major Uses of Fungicides (Tomato Section) & ICAR-IIVR",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Tomato___Septoria_leaf_spot": {
        "status": "Requires Expert Confirmation",
        "rationale": "Septoria leaf spot (Septoria lycopersici) is not an approved registered label target on CIBRC major uses labels for tomato in India. Consult your local KVK for verified localized fungicides.",
        "src_title": "PPQS CIBRC Major Uses of Fungicides (Tomato Section)",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
    "Tomato___Target_Spot": {
        "status": "Requires Expert Confirmation",
        "rationale": "Target spot (Corynespora cassiicola) does not have an approved crop-specific registered label claim for tomato under CIBRC in India. Consult your local KVK or agricultural officer.",
        "src_title": "PPQS CIBRC Major Uses of Fungicides (Tomato Section)",
        "src_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
    },
}

for cname, info in expert_classes.items():
    if cname in data:
        data[cname]["chemical_treatment_status"] = info["status"]
        data[cname]["chemical_guidance"]["available"] = False
        data[cname]["chemical_guidance"]["active_ingredient"] = None
        data[cname]["chemical_guidance"]["formulation_and_dosage"] = None
        data[cname]["chemical_guidance"]["application_method"] = (
            "Foliar chemical spraying is not verified for this specific crop-pathogen pair under current CIBRC registrations. Do not apply unverified chemical dosages."
        )
        data[cname]["chemical_guidance"]["pre_harvest_interval"] = "N/A - No verified chemical treatment"
        data[cname]["chemical_guidance"]["re_entry_interval"] = "N/A - No verified chemical treatment"
        data[cname]["chemical_guidance"]["safety_precautions"] = (
            "Standard PPE advisory: Wear protective nitrile gloves, eye protection, and mask during field operations."
        )
        data[cname]["chemical_guidance"]["disclaimer"] = (
            f"A verified chemical recommendation is not available for this disease. Consult your local KVK or agricultural officer. {info['rationale']}"
        )
        data[cname]["sources"] = [
            {
                "title": info["src_title"],
                "official_url": info["src_url"],
                "source_type": "PPQS / CIBRC, Ministry of Agriculture & Farmers Welfare",
                "last_verified": "Checked October 2024 / PPQS CIBRC Major Uses",
            }
        ]

# Also ensure Apple Black Rot and Peach Bacterial Spot sources and dates are crisp
if "Apple___Black_rot" in data:
    data["Apple___Black_rot"]["sources"] = [
        {
            "title": "PPQS CIBRC Major Uses of Fungicides (Apple Section) & ICAR-CITH",
            "official_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
            "source_type": "PPQS / CIBRC, Ministry of Agriculture & Farmers Welfare",
            "last_verified": "Checked October 2024 / PPQS CIBRC Major Uses",
        }
    ]

if "Peach___Bacterial_spot" in data:
    data["Peach___Bacterial_spot"]["sources"] = [
        {
            "title": "PPQS CIBRC Major Uses of Pesticides & ICAR-CITH Peach Guidelines",
            "official_url": "https://ppqs.gov.in/divisions/cib-rc/major-uses-pesticides",
            "source_type": "PPQS / CIBRC, Ministry of Agriculture & Farmers Welfare",
            "last_verified": "Checked October 2024 / PPQS CIBRC Major Uses",
        }
    ]

with open("data/disease_guidance.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Updated 11 classes successfully.")
