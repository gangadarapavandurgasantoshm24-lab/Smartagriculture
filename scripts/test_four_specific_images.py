import sys
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import predict_leaf
from src.guidance import get_guidance

valid_base = Path("data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid")

test_cases = [
    ("Healthy Maize", "Corn_(maize)___healthy"),
    ("Diseased Maize", "Corn_(maize)___Common_rust_"),
    ("Bell Pepper", "Pepper,_bell___Bacterial_spot"),
    ("Tomato", "Tomato___healthy"),
]

print("=" * 70)
print("TESTING SPECIFIC TARGET IMAGES ACROSS CORN, PEPPER, AND TOMATO")
print("=" * 70)

for label, target_class in test_cases:
    folder = valid_base / target_class
    if not folder.exists():
        print(f"[!] Directory not found for {target_class}")
        continue
    
    images = list(folder.glob("*.*"))
    if not images:
        print(f"[!] No images found in {folder}")
        continue
    
    sample_img = images[0]
    res = predict_leaf(sample_img, top_k=3)
    
    pred_class = res["predicted_class"]
    conf = res["confidence_percent"]
    is_healthy = res["is_healthy"]
    guidance = get_guidance(pred_class)
    
    print(f"\nTarget Category: {label}")
    print(f"  Image file:      {sample_img.name}")
    print(f"  Predicted class: {pred_class} (Confidence: {conf:.2f}%)")
    print(f"  Is Healthy:      {is_healthy}")
    print(f"  Status Match:    {'MATCH' if pred_class == target_class else 'MISMATCH'}")
    
    print("  Top 3 Candidates:")
    for cand in res["top_predictions"]:
        print(f"    #{cand['rank']}: {cand['class_name']} ({cand['confidence_percent']:.2f}%)")
        
    if guidance:
        chem = guidance.get("chemical_guidance", {})
        chem_avail = chem.get("available", False)
        print(f"  Guidance Disease:  {guidance.get('disease_name')}")
        print(f"  Guidance Etiology: {guidance.get('etiology')}")
        print(f"  Chemical Available:{chem_avail}")
        if is_healthy:
            assert chem_avail is False, f"ERROR: Healthy class {pred_class} must have chemical available=False!"
            print("  Guidance Action:   Healthy foliage detected - No chemical treatment or pesticide required.")
        else:
            if chem_avail:
                print(f"  Verified Active:   {chem.get('active_ingredient')}")
                print(f"  Dosage:            {chem.get('formulation_and_dosage')}")
            else:
                print(f"  Notice:            {chem.get('disclaimer')[:80]}...")
    else:
        print("  [!] Guidance record not found!")

print("\n" + "=" * 70)
print("SPECIFIC TARGET IMAGES TEST COMPLETE")
print("=" * 70)
