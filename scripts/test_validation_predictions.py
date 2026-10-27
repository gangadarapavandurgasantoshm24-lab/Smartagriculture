import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import predict_leaf

dataset_base = Path("data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid")

test_classes = [
    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Tomato___healthy"
]

total_correct = 0
total_tested = 0

for cls in test_classes:
    folder = dataset_base / cls
    if folder.exists():
        imgs = list(folder.glob("*.*"))[:4]
        print(f"\n=== Testing class: {cls} ({len(imgs)} samples) ===")
        for img in imgs:
            res = predict_leaf(img)
            pred = res["predicted_class"]
            conf = res["confidence_percent"]
            match = pred == cls
            total_tested += 1
            if match:
                total_correct += 1
                status = "MATCH"
            else:
                status = f"MISMATCH (Got: {pred})"
            print(f"  {img.name[:30]:30} -> {pred:42} ({conf:5.1f}%) [{status}]")
            if not match or conf < 80.0:
                print("     Top 3 candidates:")
                for p in res["top_predictions"]:
                    print(f"       #{p['rank']}: {p['class_name']} ({p['confidence_percent']:.2f}%)")

print(f"\nSummary: {total_correct}/{total_tested} correct ({total_correct/total_tested*100:.1f}%)")
