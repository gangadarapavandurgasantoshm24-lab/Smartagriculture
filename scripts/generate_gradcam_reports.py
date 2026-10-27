import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inference import predict_leaf, load_class_names
from src.gradcam import generate_gradcam_heatmap, apply_gradcam_overlay

out_dir = Path("reports/gradcam_examples")
out_dir.mkdir(parents=True, exist_ok=True)

test_cases = [
    {
        "id": "case1_corn_rust",
        "path": Path("data/raw/test/test/CornCommonRust1.JPG"),
        "actual": "Corn_(maize)___Common_rust_",
        "type": "Correct Benchmark (Corn Common Rust)",
        "explanation": "High focal attention concentrated over orange-brown elongated rust pustules erupting through the upper leaf epidermis.",
    },
    {
        "id": "case2_tomato_early_blight",
        "path": Path("data/raw/test/test/TomatoEarlyBlight1.JPG"),
        "actual": "Tomato___Early_blight",
        "type": "Correct Benchmark (Tomato Early Blight)",
        "explanation": "Attention maps localize directly onto dark brown concentric target-like necrotic rings surrounded by chlorotic yellow halos on lower leaflets.",
    },
    {
        "id": "case3_potato_healthy",
        "path": Path("data/raw/test/test/PotatoHealthy1.JPG"),
        "actual": "Potato___healthy",
        "type": "Correct Benchmark (Potato Healthy)",
        "explanation": "Diffuse, broadly distributed feature activation across the uniform green leaf lamina without focal concentration on necrotic spots.",
    },
    {
        "id": "case4_apple_cedar_rust",
        "path": Path("data/raw/test/test/AppleCedarRust1.JPG"),
        "actual": "Apple___Cedar_apple_rust",
        "type": "Correct Benchmark (Apple Cedar Rust)",
        "explanation": "Strong localized activation on bright yellow-orange circular lesions (aecia) characteristic of Gymnosporangium juniperi-virginianae infection.",
    },
    {
        "id": "case5_field_apple_failure",
        "path": Path("data/user_failure_cases/images_apple.jpg"),
        "actual": "Apple (with leaf lesions, field context)",
        "type": "Field Misclassification (Apple -> Peach Bacterial Spot)",
        "explanation": "The CNN was misled by outdoor natural lighting and irregular lesion margins, focusing strongly on dark edges and confusing apple foliage with peach bacterial necrosis.",
    },
    {
        "id": "case6_field_corn_health_failure",
        "path": Path("data/user_failure_cases/images_corn health.jpg"),
        "actual": "Corn (Maize) Healthy (with border artifact)",
        "type": "Field Misclassification (Maize -> Pepper Bell Healthy)",
        "explanation": "The white letterbox border and aspect ratio stretching misled spatial filters at conv2d_3, dispersing attention along margins rather than linear parallel maize venation.",
    },
]

print(f"Executing Grad-CAM generator for {len(test_cases)} case studies...")

for tc in test_cases:
    p = tc["path"]
    if not p.is_file():
        print(f"Skipping missing image: {p}")
        continue

    with open(p, "rb") as f:
        img_bytes = f.read()

    res = predict_leaf(img_bytes, top_k=3)
    heatmap = generate_gradcam_heatmap(
        res["input_tensor"], target_class_index=res["class_index"]
    )
    colored_heatmap, overlay = apply_gradcam_overlay(
        res["display_image"], heatmap, alpha=0.45
    )

    # Save individual components
    orig_path = out_dir / f"{tc['id']}_orig.png"
    heat_path = out_dir / f"{tc['id']}_heatmap.png"
    over_path = out_dir / f"{tc['id']}_overlay.png"
    panel_path = out_dir / f"{tc['id']}_panel.png"

    cv2.imwrite(str(orig_path), cv2.cvtColor(res["display_image"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heat_path), cv2.cvtColor(colored_heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(over_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Create publication-quality 3-panel composite figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=200)

    # 1. Original Image
    axes[0].imshow(res["display_image"])
    axes[0].set_title(
        f"(a) Input Leaf: {p.name}\nActual: {tc['actual']}",
        fontsize=9,
        fontweight="bold",
    )
    axes[0].axis("off")

    # 2. Grad-CAM Heatmap
    im_heat = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        "(b) Grad-CAM Heatmap (conv2d_3)\n[60x60 -> 256x256]",
        fontsize=9,
        fontweight="bold",
    )
    axes[1].axis("off")
    fig.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Alpha Blended Overlay
    axes[2].imshow(overlay)
    pred_text = f"Pred: {res['predicted_class']}\nConfidence: {res['confidence_percent']:.2f}%"
    axes[2].set_title(
        f"(c) Overlay (alpha=0.45)\n{pred_text}",
        fontsize=9,
        fontweight="bold",
    )
    axes[2].axis("off")

    fig.suptitle(
        f"{tc['type']} | CNN Layer conv2d_3 Attention",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(str(panel_path), dpi=200, bbox_inches="tight")
    plt.close()

    print(
        f"Generated {tc['id']}: Pred={res['predicted_class']} ({res['confidence_percent']:.2f}%) vs Actual={tc['actual']}"
    )

print("All Grad-CAM case studies generated successfully in reports/gradcam_examples/.")
