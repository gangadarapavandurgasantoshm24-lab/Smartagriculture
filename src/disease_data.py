from typing import Any, Dict, List, Optional
import pandas as pd
from src.config import CLASS_ANALYSIS_PATH


# Factual Pathological and Agronomic Extension Data for all 38 Classes
# Sourced from University Agricultural Extensions (Cornell, UC Davis IPM, USDA-ARS, Purdue)
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "crop": "Apple",
        "common_name": "Apple Scab",
        "category": "Diseased",
        "pathogen": "Venturia inaequalis (Ascomycete fungus)",
        "symptoms": "Olive-green to dull brown velvety spots on upper leaf surfaces that darken and develop raised, scabby margins. Distorted leaf growth and premature leaf drop.",
        "favorable_conditions": "Cool (55°F to 75°F / 13°C to 24°C), rainy spring conditions with prolonged leaf wetness (6+ hours).",
        "ipm_practices": "Collect and shred or compost fallen autumn leaves to eliminate overwintering ascospores. Prune canopy trees to improve sun penetration and air drying. Plant resistant cultivars (e.g., Liberty, Freedom).",
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "common_name": "Black Rot (Frogeye Leaf Spot)",
        "category": "Diseased",
        "pathogen": "Botryosphaeria obtusa (Fungal pathogen)",
        "symptoms": "Small purple specks on leaves expanding into circular tan lesions with distinct purple or dark margins ('frogeye' appearance). Concentric ring patterns can form.",
        "favorable_conditions": "Warm (68°F to 80°F / 20°C to 27°C) humid weather, especially following mechanical fruit injury or fire blight strikes.",
        "ipm_practices": "Prune out dead wood, mummified fruits, and cankers during dormant periods. Disinfect pruning tools between cuts. Maintain balanced orchard tree nutrition.",
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "common_name": "Cedar Apple Rust",
        "category": "Diseased",
        "pathogen": "Gymnosporangium juniperi-virginianae (Heteroecious rust fungus)",
        "symptoms": "Bright yellow-orange or reddish circular spots on the upper leaf surface, later producing tiny black fruiting bodies (pycnia). Undersides develop tube-like aecia.",
        "favorable_conditions": "Spring rains dispersing basidiospores from nearby eastern red cedar or juniper galls.",
        "ipm_practices": "Eliminate alternate cedar/juniper hosts within 1-2 miles of commercial orchards where practical. Select rust-immune cultivars (e.g., Enterprise, Redfree).",
    },
    "Apple___healthy": {
        "crop": "Apple",
        "common_name": "Healthy Apple Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Uniform deep green leaf blade, intact serrated leaf margins, prominent pale green veins, and absence of chlorosis, necrosis, or lesions.",
        "favorable_conditions": "Well-drained soil, 6-8 hours full sunlight, adequate spacing, and balanced N-P-K fertilization.",
        "ipm_practices": "Continue routine scout monitoring, drip irrigation management, and annual dormant pruning to maintain healthy airflow.",
    },
    "Blueberry___healthy": {
        "crop": "Blueberry",
        "common_name": "Healthy Blueberry Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Glossy, smooth, elliptical leaves with consistent green pigment and no signs of leaf scorch, spotting, or viral mottling.",
        "favorable_conditions": "Acidic soils (pH 4.5–5.2), high organic matter, pine bark mulching, and steady soil moisture without waterlogging.",
        "ipm_practices": "Maintain soil acidification via elemental sulfur, ensure drip irrigation, and avoid overhead watering.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry (including sour)",
        "common_name": "Powdery Mildew",
        "category": "Diseased",
        "pathogen": "Podosphaera clandestina (Erysiphales fungus)",
        "symptoms": "White to light gray powdery fungal patches on young terminal leaves. Infected leaves become curled, twisted, and stunted, with reduced photosynthesis.",
        "favorable_conditions": "Moderate temperatures (60°F to 80°F / 15°C to 27°C) and high relative humidity; does not require free standing water.",
        "ipm_practices": "Prune dense watersprouts to lower relative canopy humidity. Avoid excess nitrogen fertilization which stimulates vulnerable succulent tissue.",
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry (including sour)",
        "common_name": "Healthy Cherry Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Vibrant green, pointed ovate leaves with serrated margins, clean petiole glands, and unblemished surface cuticle.",
        "favorable_conditions": "Full sun, loamy soil with neutral to slightly acidic pH, good air drainage to prevent spring frost damage.",
        "ipm_practices": "Maintain standard seasonal sanitation, dormant copper sprays if bacterial canker is endemic in the district.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (maize)",
        "common_name": "Gray Leaf Spot (Cercospora Leaf Spot)",
        "category": "Diseased",
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "symptoms": "Small tan spots with yellow halos expanding into long, narrow, rectangular lesions restricted by parallel leaf veins. Lesions turn grayish with fungal sporulation.",
        "favorable_conditions": "Prolonged warm (75°F to 85°F / 24°C to 29°C) conditions combined with high humidity (90%+) and morning dews.",
        "ipm_practices": "Rotate crops with non-hosts like soybean or alfalfa for at least one year. Till infected residue in fields with severe history. Plant resistant corn hybrids.",
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (maize)",
        "common_name": "Common Rust",
        "category": "Diseased",
        "pathogen": "Puccinia sorghi (Macrocyclic rust fungus)",
        "symptoms": "Small, oval to elongate cinnamon-brown powdery pustules scattered across both upper and lower leaf surfaces. Pustules rupture the epidermis.",
        "favorable_conditions": "Cool to moderate temperatures (60°F to 75°F / 15°C to 24°C) with high humidity (>95%) and night dew.",
        "ipm_practices": "Plant corn hybrids carrying the Rp single-gene or multi-gene general resistance. Early planting helps avoid peak mid-summer airborne spore flights.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (maize)",
        "common_name": "Northern Corn Leaf Blight (NCLB)",
        "category": "Diseased",
        "pathogen": "Exserohilum turcicum (syn. Setosphaeria turcica)",
        "symptoms": "Distinctive large, elongated cigar-shaped or elliptical grayish-green to tan lesions (1 to 6 inches long) that develop on lower leaves and progress upward.",
        "favorable_conditions": "Moderate temperatures (65°F to 80°F / 18°C to 27°C) accompanied by extended leaf wetness (6-18 hours).",
        "ipm_practices": "Select hybrids with Ht resistance genes. Practice crop rotation and bury residue in high-risk conservation tillage acreage.",
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (maize)",
        "common_name": "Healthy Corn Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Broad, erect, deep green arching leaves with intact midrib, parallel venation, and clean leaf margins free of lesions.",
        "favorable_conditions": "Adequate nitrogen, phosphorus, and potassium fertility, warm soil during emergence, and 20-25 inches of seasonal water.",
        "ipm_practices": "Regular scouting for armyworm and rootworm damage; soil moisture monitoring.",
    },
    "Grape___Black_rot": {
        "crop": "Grape",
        "common_name": "Grape Black Rot",
        "category": "Diseased",
        "pathogen": "Guignardia bidwellii (Fungus)",
        "symptoms": "Circular reddish-brown leaf spots with dark borders. Inside the spots, tiny black pimple-like pycnidia fruiting bodies form in a ring.",
        "favorable_conditions": "Warm (70°F to 85°F / 21°C to 29°C) wet weather with continuous canopy moisture.",
        "ipm_practices": "Canopy shoot positioning, leaf removal around grape clusters, sanitation of mummified berries from vine cordons during winter pruning.",
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "common_name": "Esca (Black Measles / Grape Trunk Disease)",
        "category": "Diseased",
        "pathogen": "Fomitiporia mediterranea, Phaeomoniella chlamydospora complex",
        "symptoms": "'Tiger-stripe' chlorotic and necrotic patterns between leaf veins. Berries develop small brown-purple puncture spots resembling measles.",
        "favorable_conditions": "Older established vineyards; stressed vines under high summer evaporative demand.",
        "ipm_practices": "Protect pruning wounds with sealant or Trichoderma biocontrol agents. Avoid large pruning cuts during wet rainy spells.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "common_name": "Leaf Blight (Phaeoisariopsis / Isariopsis Leaf Spot)",
        "category": "Diseased",
        "pathogen": "Pseudocercospora vitis (syn. Phaeoisariopsis vitis)",
        "symptoms": "Irregular dark brown to black necrotic spots with chlorotic haloes on older leaves, causing localized blighting and premature defoliation.",
        "favorable_conditions": "High humidity, overhead irrigation, dense canopies in late summer to autumn.",
        "ipm_practices": "Improve trellis canopy ventilation, prune dense outer shoots, and remove severely affected leaf litter.",
    },
    "Grape___healthy": {
        "crop": "Grape",
        "common_name": "Healthy Grape Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Palmate lobed leaves with vibrant green coloration, clean sinus lobes, prominent palmate venation, and no marginal necrosis.",
        "favorable_conditions": "Well-drained gravelly or sandy loam soils, good sun exposure on trellises, deficit irrigation management.",
        "ipm_practices": "Routine canopy shoot thinning, wire lifting, and pest monitoring for leafhoppers and powdery mildew.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange",
        "common_name": "Huanglongbing (HLB / Citrus Greening)",
        "category": "Diseased",
        "pathogen": "Candidatus Liberibacter asiaticus (Fastidious bacterium spread by Asian citrus psyllid)",
        "symptoms": "Asymmetrical blotchy mottle chlorosis on leaves (veins do not cross symmetrically). Leaves are small, upright, and leathery. Yellowing shoots ('yellow dragon').",
        "favorable_conditions": "Active Asian citrus psyllid (Diaphorina citri) vector populations feeding on new flush foliage.",
        "ipm_practices": "Vector control using systemic insecticides during new flush periods. Remove confirmed infected trees to reduce inoculum reservoir. Use certified disease-free nursery stock.",
    },
    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "common_name": "Bacterial Spot (Shot Hole)",
        "category": "Diseased",
        "pathogen": "Xanthomonas arboricola pv. pruni (Bacterium)",
        "symptoms": "Water-soaked polygonal angular purple spots along leaf midrib and tips. Centers often drop out, leaving a perforated 'shot hole' appearance.",
        "favorable_conditions": "Warm, windy, rainy periods causing sand-blasting or mechanical wounding of tender leaves.",
        "ipm_practices": "Avoid overhead sprinkler irrigation. Maintain windbreaks in sandy soils. Plant tolerant peach varieties (e.g., Candor, Biscoe, Belle of Georgia).",
    },
    "Peach___healthy": {
        "crop": "Peach",
        "common_name": "Healthy Peach Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Slender, lanceolate leaves with gently undulating margins, uniform bright green sheen, and no perforations or gumming.",
        "favorable_conditions": "Chilling hours met in winter, full sun, deep well-drained soil (pH 6.0–6.5), weed-free root zones.",
        "ipm_practices": "Maintain regular orchard scouting, dormant fungicide sprays for leaf curl, and balanced fertilization.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Pepper, bell",
        "common_name": "Bacterial Leaf Spot",
        "category": "Diseased",
        "pathogen": "Xanthomonas euvesicatoria species complex",
        "symptoms": "Small, water-soaked, circular to irregular lesions that become dark brown with yellow halos. Severe infection causes severe leaf drop and scalded fruit.",
        "favorable_conditions": "High temperatures (75°F to 86°F / 24°C to 30°C) with persistent rains, sprinkler splash, or heavy dews.",
        "ipm_practices": "Use certified pathogen-free pepper seeds. Avoid working in fields when foliage is wet. Rotate fields out of Solanaceous crops for 2-3 years.",
    },
    "Pepper,_bell___healthy": {
        "crop": "Pepper, bell",
        "common_name": "Healthy Bell Pepper Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Broad, smooth ovate leaves, rich dark green color, distinct pinnate veins, and no necrotic margins or curling.",
        "favorable_conditions": "Warm soil, consistent moisture (drip irrigation), high light intensity, and balanced calcium-to-nitrogen ratios.",
        "ipm_practices": "Mulch soil with plastic or straw, scout for aphids and thrips, use drip irrigation.",
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "common_name": "Early Blight",
        "category": "Diseased",
        "pathogen": "Alternaria solani (Fungus)",
        "symptoms": "Small brown to black spots on older lower leaves, developing characteristic concentric ring ridges resembling a target board with yellow surrounding tissue.",
        "favorable_conditions": "Alternating wet and dry periods; warm temperatures (75°F to 85°F / 24°C to 29°C) on plants under stress.",
        "ipm_practices": "Avoid nitrogen deficiency which accelerates leaf senescence. Irrigate early in the morning so foliage dries quickly. Rotate crops with non-Solanaceous crops.",
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "common_name": "Late Blight",
        "category": "Diseased",
        "pathogen": "Phytophthora infestans (Oomycete water mold)",
        "symptoms": "Large, irregular, water-soaked dark brown to purplish-black lesions that spread rapidly across leaves. White downy cottony mold appears on leaf undersides in humid air.",
        "favorable_conditions": "Cool, moist weather (60°F to 70°F / 15°C to 21°C) with relative humidity above 90%.",
        "ipm_practices": "Plant certified disease-free seed tubers. Eliminate cull piles which serve as primary spring spore sources. Apply preventive protective fungicides under high forecast pressure.",
    },
    "Potato___healthy": {
        "crop": "Potato",
        "common_name": "Healthy Potato Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Compound pinnate leaves with bright green leaflets, firm petioles, crisp margins, and no wilt or spotting.",
        "favorable_conditions": "Well-hilled soil, cool night temperatures, consistent soil moisture (avoiding saturated tubers), and full sun.",
        "ipm_practices": "Maintain adequate hilling to protect tubers from light and spore wash; scout for Colorado potato beetle.",
    },
    "Raspberry___healthy": {
        "crop": "Raspberry",
        "common_name": "Healthy Raspberry Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Pinnately compound leaves (3-5 leaflets) with sharply serrated margins, whitish-pubescent undersides, and vigorous cane shoots.",
        "favorable_conditions": "Raised beds for drainage, organic mulch, trellis wire support, and adequate drip irrigation.",
        "ipm_practices": "Annual floricane pruning after harvest, good row spacing to avoid spur blight and cane botrytis.",
    },
    "Soybean___healthy": {
        "crop": "Soybean",
        "common_name": "Healthy Soybean Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Trifoliate leaves with vibrant green, smooth leaflets, clean veins, and no chlorotic mottling, puckering, or rust pustules.",
        "favorable_conditions": "Warm seedbed, active Bradyrhizobium nodulation, neutral soil pH (6.0–6.8), and timely summer rains.",
        "ipm_practices": "Scout for soybean aphid and defoliating caterpillars; practice no-till crop rotation with corn.",
    },
    "Squash___Powdery_mildew": {
        "crop": "Squash",
        "common_name": "Squash Powdery Mildew",
        "category": "Diseased",
        "pathogen": "Podosphaera xanthii (syn. Sphaerotheca fuliginea)",
        "symptoms": "Talcum-powder-like white fungal colonies on upper and lower leaf surfaces and petioles. Leaves gradually turn yellow, brown, and brittle.",
        "favorable_conditions": "Dense shade, poor air movement, moderate temperatures (68°F to 80°F / 20°C to 27°C), and high relative humidity.",
        "ipm_practices": "Plant PM-tolerant squash varieties. Ensure wide plant spacing (4-6 feet between rows). Use bio-fungicides (e.g., potassium bicarbonate, neem oil) at first sign.",
    },
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "common_name": "Strawberry Leaf Scorch",
        "category": "Diseased",
        "pathogen": "Diplocarpon earlianum (Fungus)",
        "symptoms": "Numerous small, irregular purple to dark brown blotches. Unlike common leaf spot, scorch spots remain dark throughout without white centers. Heavily infected leaves look burned.",
        "favorable_conditions": "Prolonged leaf wetness during warm spring and autumn periods.",
        "ipm_practices": "Renovate strawberry beds after harvest by mowing old foliage. Avoid overhead sprinkler irrigation; remove dead plant debris.",
    },
    "Strawberry___healthy": {
        "crop": "Strawberry",
        "common_name": "Healthy Strawberry Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Trifoliate, coarsely serrated leaflets with deep green color, sturdy runners, and no necrotic margins or fungal spots.",
        "favorable_conditions": "Raised plasticulture beds, drip fertigation, high organic matter, and adequate winter mulch protection.",
        "ipm_practices": "Scout weekly for two-spotted spider mites; replace strawberry crowns every 3-4 years.",
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "common_name": "Bacterial Spot",
        "category": "Diseased",
        "pathogen": "Xanthomonas perforans / Xanthomonas gardneri complex",
        "symptoms": "Small, dark, greasy, water-soaked spots that become angular and brown with yellow halos. Severe infections cause extensive defoliation and sunscalded fruit.",
        "favorable_conditions": "Warm temperatures (75°F to 86°F / 24°C to 30°C) with persistent splashing rain or overhead irrigation.",
        "ipm_practices": "Use hot-water treated or certified disease-free seeds. Never work in wet tomato rows. Apply copper-based bactericides combined with mancozeb where resistance is not established.",
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "common_name": "Early Blight",
        "category": "Diseased",
        "pathogen": "Alternaria linariae (syn. Alternaria solani)",
        "symptoms": "Circular brown-black spots with distinct concentric rings ('bullseye' or target board pattern) surrounded by chlorotic yellow halos, starting on lowest leaves.",
        "favorable_conditions": "Warm temperatures (75°F to 85°F / 24°C to 29°C) with heavy dew, frequent rains, or overhead sprinkling.",
        "ipm_practices": "Stake and prune tomato vines to keep foliage off the ground. Mulch soil around plant bases to prevent spore splash from soil. Rotate out of Solanaceae for 3 years.",
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "common_name": "Late Blight",
        "category": "Diseased",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Large, irregular water-soaked pale to dark olive-brown patches on leaves and stems. Under high humidity, a delicate white downy fungal-like bloom appears on leaf undersides.",
        "favorable_conditions": "Cool, wet, cloudy weather (60°F to 72°F / 15°C to 22°C) with relative humidity above 90%.",
        "ipm_practices": "Destroy volunteer tomato/potato plants. Select late blight resistant cultivars (e.g., Defiant, Mountain Merit, Iron Lady). Immediately destroy infected plants to protect regional crops.",
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "common_name": "Leaf Mold",
        "category": "Diseased",
        "pathogen": "Passalora fulva (syn. Fulvia fulva / Cladosporium fulvum)",
        "symptoms": "Pale green to yellowish indistinct spots on the upper leaf surface, corresponding to olive-green to velvety brown mold growth on the lower surface.",
        "favorable_conditions": "High relative humidity (85%+) and warm temperatures (70°F to 80°F / 21°C to 27°C), common in greenhouses and high tunnels.",
        "ipm_practices": "Ventilate greenhouses and use circulating fans to drop relative humidity below 85%. Increase plant spacing and prune lower leaves.",
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "common_name": "Septoria Leaf Spot",
        "category": "Diseased",
        "pathogen": "Septoria lycopersici (Fungus)",
        "symptoms": "Numerous small (1/8 inch), circular spots with dark brown margins and sunken light gray or tan centers. Tiny black specks (pycnidia) are visible in the center.",
        "favorable_conditions": "Moderate temperatures (68°F to 77°F / 20°C to 25°C) with persistent wet foliage.",
        "ipm_practices": "Strip lower leaves as plants mature to reduce soil-splash infection. Apply thick straw or plastic mulch. Practice 2 to 3 year crop rotation.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "common_name": "Two-Spotted Spider Mite Damage",
        "category": "Diseased",
        "pathogen": "Tetranychus urticae (Arachnid mite pest)",
        "symptoms": "Fine yellow or white stippling (speckling) on upper leaf surfaces. As colonies grow, fine webbing covers the leaves and shoot tips, causing leaves to bronze and dry up.",
        "favorable_conditions": "Hot (>80°F / 27°C), dry, dusty conditions and drought stress.",
        "ipm_practices": "Avoid broad-spectrum pyrethroids that destroy natural predatory mites. Release beneficial phytoseiid predators (e.g., Phytoseiulus persimilis). Wash foliage with water or insecticidal soap.",
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "common_name": "Target Spot",
        "category": "Diseased",
        "pathogen": "Corynespora cassiicola (Fungus)",
        "symptoms": "Small, pinpoint water-soaked lesions that enlarge into circular brown lesions with light brown centers and dark concentric rings. Unlike early blight, target spot attacks middle/upper canopy too.",
        "favorable_conditions": "Warm (68°F to 82°F / 20°C to 28°C) temperatures combined with long periods of high humidity.",
        "ipm_practices": "Improve air movement within plant rows. Avoid overhead irrigation. Remove weed hosts like wild poinsettia and nightshade.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "common_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "category": "Diseased",
        "pathogen": "Tomato yellow leaf curl virus (Begomovirus transmitted by Whiteflies)",
        "symptoms": "Severe stunting, erect bushy growth, pronounced upward cupping of leaflet margins, and chlorotic (yellow) leaf margins with reduction in leaf blade size.",
        "favorable_conditions": "High populations of the silverleaf whitefly (Bemisia tabaci) in warm summer/autumn seasons.",
        "ipm_practices": "Install fine 50-mesh insect screening on greenhouses. Apply yellow sticky traps for monitoring. Use reflective silver mulches to repel whitefly vectors. Plant TYLCV-resistant hybrids.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "common_name": "Tomato Mosaic Virus (ToMV)",
        "category": "Diseased",
        "pathogen": "Tomato mosaic virus (Tobamovirus)",
        "symptoms": "Mottled light and dark green mosaic patterns on leaves, leaf distortion (fern-like or shoestring appearance), stunted vegetative growth, and delayed maturity.",
        "favorable_conditions": "Mechanically transmitted very easily via contaminated hands, tools, clothes, and sap contact.",
        "ipm_practices": "Disinfect pruning shears in 20% nonfat dry milk or 10% trisodium phosphate. Prohibit smoking/tobacco near greenhouse tomatoes. Use certified virus-free seed.",
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "common_name": "Healthy Tomato Foliage",
        "category": "Healthy",
        "pathogen": "None (Healthy tissue)",
        "symptoms": "Vigorous, dark green compound leaves with glandular trichomes (hairs), crisp margins, sturdy stems, and no chlorosis or necrosis.",
        "favorable_conditions": "Deep fertile soil with organic matter, 6-8 hours of direct sun, consistent drip irrigation, and balanced N-P-K plus calcium.",
        "ipm_practices": "Stake and trellis plants, prune suckers for optimal light penetration, maintain regular pest scouting.",
    },
}


def get_disease_info(class_name: str) -> Dict[str, Any]:
    """Retrieve detailed agricultural extension info for a class."""
    if class_name in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[class_name]

    # Fallback parser if custom class passed
    crop_part, cond_part = (
        class_name.split("___", 1) if "___" in class_name else ("Crop", class_name)
    )
    is_healthy = "healthy" in cond_part.lower()
    return {
        "crop": crop_part.replace("_", " "),
        "common_name": cond_part.replace("_", " "),
        "category": "Healthy" if is_healthy else "Diseased",
        "pathogen": "N/A" if is_healthy else "Unspecified plant pathogen",
        "symptoms": "No visible symptoms"
        if is_healthy
        else "Lesions or discoloration on foliage.",
        "favorable_conditions": "Standard agricultural conditions.",
        "ipm_practices": "Consult local agricultural extension service.",
    }


def get_all_crops() -> List[str]:
    """Return sorted unique list of all crops represented in the model."""
    crops = sorted(list({data["crop"] for data in DISEASE_KNOWLEDGE_BASE.values()}))
    return crops


def filter_diseases(
    crop: Optional[str] = None,
    category: Optional[str] = None,
    search_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter disease library records by crop, status category, or search keywords.
    """
    results = []
    query = search_query.strip().lower() if search_query else ""

    for class_key, data in DISEASE_KNOWLEDGE_BASE.items():
        if crop and crop != "All" and data["crop"] != crop:
            continue
        if category and category != "All" and data["category"] != category:
            continue

        if query:
            searchable = (
                f"{class_key} {data['crop']} {data['common_name']} {data['pathogen']} {data['symptoms']}"
            ).lower()
            if query not in searchable:
                continue

        item = dict(data)
        item["class_key"] = class_key
        results.append(item)

    return results
