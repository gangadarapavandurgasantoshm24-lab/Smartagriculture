import json
import os
import sys
from pathlib import Path

# Suppress excessive TensorFlow C++ logs and prevent OpenMP collision
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES_PATH, MODEL_PATH
from src.inference import load_class_names, preprocess_image

VALID_DIR = Path("data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid")
SAMPLES_PER_CLASS = 50

def run_evaluation_for_model(model_path: Path, model_label: str, class_names: list[str]):
    print("\n" + "=" * 80)
    print(f"EVALUATING CHECKPOINT: {model_label}")
    print(f"Path: {model_path}")
    print("=" * 80)
    
    if not model_path.is_file():
        print(f"[!] Checkpoint not found: {model_path}")
        return None
        
    model = tf.keras.models.load_model(str(model_path))
    
    y_true_indices = []
    y_pred_indices = []
    y_pred_confs = []
    is_correct_list = []
    file_records = []
    
    # Collect 50 images per class
    for class_idx, class_name in enumerate(class_names):
        class_folder = VALID_DIR / class_name
        if not class_folder.is_dir():
            print(f"[!] Warning: Folder {class_folder} not found!")
            continue
            
        all_imgs = sorted(list(class_folder.glob("*.*")))
        selected_imgs = all_imgs[:SAMPLES_PER_CLASS]
        
        # Batch load and preprocess
        tensors = []
        for img_p in selected_imgs:
            tensor, _ = preprocess_image(img_p)
            tensors.append(tensor[0])
            
        batch_tensors = np.array(tensors, dtype=np.float32)
        preds = model.predict(batch_tensors, verbose=0)
        
        for idx_in_batch, img_p in enumerate(selected_imgs):
            probs = preds[idx_in_batch]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            correct = (top_idx == class_idx)
            
            y_true_indices.append(class_idx)
            y_pred_indices.append(top_idx)
            y_pred_confs.append(top_conf)
            is_correct_list.append(correct)
            
            file_records.append({
                "file": img_p.name,
                "true_class": class_name,
                "pred_class": class_names[top_idx],
                "confidence": top_conf,
                "correct": correct
            })
            
    y_true = np.array(y_true_indices)
    y_pred = np.array(y_pred_indices)
    confs = np.array(y_pred_confs)
    corrects = np.array(is_correct_list)
    
    total = len(y_true)
    overall_acc = float(np.mean(corrects)) * 100.0
    
    print(f"\n--- Overall Summary for {model_label} ---")
    print(f"Total Evaluated Images:  {total} (across {len(class_names)} classes, {SAMPLES_PER_CLASS} per class)")
    print(f"Overall Accuracy:        {overall_acc:.2f}% ({np.sum(corrects)} / {total})")
    
    # Confidence distributions
    corr_confs = confs[corrects]
    incorr_confs = confs[~corrects]
    
    print(f"\n--- Confidence Distributions ---")
    if len(corr_confs) > 0:
        print(f"Correct Predictions (n={len(corr_confs)}):")
        print(f"  Mean:   {corr_confs.mean()*100:.2f}%")
        print(f"  Median: {np.median(corr_confs)*100:.2f}%")
        print(f"  Min:    {corr_confs.min()*100:.2f}%")
        print(f"  Max:    {corr_confs.max()*100:.2f}%")
        print(f"  Std:    {corr_confs.std()*100:.2f}%")
    if len(incorr_confs) > 0:
        print(f"Incorrect Predictions (n={len(incorr_confs)}):")
        print(f"  Mean:   {incorr_confs.mean()*100:.2f}%")
        print(f"  Median: {np.median(incorr_confs)*100:.2f}%")
        print(f"  Min:    {incorr_confs.min()*100:.2f}%")
        print(f"  Max:    {incorr_confs.max()*100:.2f}%")
        print(f"  Std:    {incorr_confs.std()*100:.2f}%")
        
    # Specific Cross-Crop Confusions
    print(f"\n--- Cross-Crop Species Confusions ---")
    df_rec = pd.DataFrame(file_records)
    
    # 1. Apple predicted as Peach
    apple_as_peach = df_rec[df_rec["true_class"].str.startswith("Apple") & df_rec["pred_class"].str.startswith("Peach")]
    print(f"1. Apple images predicted as Peach: {len(apple_as_peach)} cases")
    for _, r in apple_as_peach.head(5).iterrows():
        print(f"   - {r['true_class']} -> {r['pred_class']} ({r['confidence']*100:.1f}%) [{r['file']}]")
        
    # 2. Maize predicted as Pepper or Peach
    maize_as_pepp_peach = df_rec[df_rec["true_class"].str.startswith("Corn") & (df_rec["pred_class"].str.startswith("Pepper") | df_rec["pred_class"].str.startswith("Peach"))]
    print(f"2. Maize images predicted as Pepper or Peach: {len(maize_as_pepp_peach)} cases")
    for _, r in maize_as_pepp_peach.head(5).iterrows():
        print(f"   - {r['true_class']} -> {r['pred_class']} ({r['confidence']*100:.1f}%) [{r['file']}]")
        
    # 3. Pepper predicted as Maize or Peach
    pepp_as_maize_peach = df_rec[df_rec["true_class"].str.startswith("Pepper") & (df_rec["pred_class"].str.startswith("Corn") | df_rec["pred_class"].str.startswith("Peach"))]
    print(f"3. Pepper images predicted as Maize or Peach: {len(pepp_as_maize_peach)} cases")
    for _, r in pepp_as_maize_peach.head(5).iterrows():
        print(f"   - {r['true_class']} -> {r['pred_class']} ({r['confidence']*100:.1f}%) [{r['file']}]")

    # Macro & Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\nMacro Metrics:    Precision={p_macro*100:.2f}%, Recall={r_macro*100:.2f}%, F1={f1_macro*100:.2f}%")
    print(f"Weighted Metrics: Precision={p_weight*100:.2f}%, Recall={r_weight*100:.2f}%, F1={f1_weight*100:.2f}%")

    # Per-class metrics
    p_cls, r_cls, f1_cls, s_cls = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class_df = pd.DataFrame({
        "class_name": class_names,
        "precision": p_cls,
        "recall": r_cls,
        "f1_score": f1_cls,
        "support": s_cls
    })
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    return {
        "model_label": model_label,
        "accuracy": overall_acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "corr_confs": corr_confs,
        "incorr_confs": incorr_confs,
        "apple_as_peach_count": len(apple_as_peach),
        "maize_as_pepper_peach_count": len(maize_as_pepp_peach),
        "pepper_as_maize_peach_count": len(pepp_as_maize_peach),
        "per_class_df": per_class_df,
        "confusion_matrix": cm,
        "records_df": df_rec
    }


def main():
    class_names = load_class_names()
    
    # 1. Evaluate models/best_model.keras
    res_app = run_evaluation_for_model(MODEL_PATH, "models/best_model.keras (Application Model)", class_names)
    
    # 2. Evaluate cnn/best_model.keras
    cnn_path = Path("cnn/best_model.keras")
    res_cnn = run_evaluation_for_model(cnn_path, "cnn/best_model.keras (Training Directory Checkpoint)", class_names)
    
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON ON EXACT SAME 1,900 VALIDATION IMAGES")
    print("=" * 80)
    if res_app and res_cnn:
        print(f"{'Metric':<35} {'models/best_model.keras':<25} {'cnn/best_model.keras':<25}")
        print("-" * 85)
        print(f"{'Overall Validation Accuracy':<35} {res_app['accuracy']:.2f}%{'':<19} {res_cnn['accuracy']:.2f}%")
        print(f"{'Macro Precision':<35} {res_app['macro_precision']*100:.2f}%{'':<19} {res_cnn['macro_precision']*100:.2f}%")
        print(f"{'Macro Recall':<35} {res_app['macro_recall']*100:.2f}%{'':<19} {res_cnn['macro_recall']*100:.2f}%")
        print(f"{'Macro F1-Score':<35} {res_app['macro_f1']*100:.2f}%{'':<19} {res_cnn['macro_f1']*100:.2f}%")
        print(f"{'Apple predicted as Peach':<35} {res_app['apple_as_peach_count']} images{'':<17} {res_cnn['apple_as_peach_count']} images")
        print(f"{'Maize predicted as Pepper/Peach':<35} {res_app['maize_as_pepper_peach_count']} images{'':<17} {res_cnn['maize_as_pepper_peach_count']} images")
        print(f"{'Pepper predicted as Maize/Peach':<35} {res_app['pepper_as_maize_peach_count']} images{'':<17} {res_cnn['pepper_as_maize_peach_count']} images")
        
        # Save detailed comparison report to reports/
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        res_app["per_class_df"].to_csv(reports_dir / "eval_per_class_app_model.csv", index=False)
        res_cnn["per_class_df"].to_csv(reports_dir / "eval_per_class_cnn_model.csv", index=False)
        pd.DataFrame(res_app["confusion_matrix"], index=class_names, columns=class_names).to_csv(reports_dir / "eval_confusion_matrix_app_model.csv")
        pd.DataFrame(res_cnn["confusion_matrix"], index=class_names, columns=class_names).to_csv(reports_dir / "eval_confusion_matrix_cnn_model.csv")
        print(f"\nDetailed artifacts saved to {reports_dir}/")


if __name__ == "__main__":
    main()
