"""
evaluate.py - Standalone evaluation script for the Screw Defect Detector.

This script ONLY reads the already-trained model and reports metrics on a
test set. It never modifies, retrains, or fine-tunes the model in any way —
completely safe to run without affecting the deployed weights.

Usage:
    python evaluate.py --test_dir path/to/dataset/test

Expects `test_dir` to contain two subfolders matching CLASS_NAMES in
predict.py:
    test_dir/
        good/
        defective/

Outputs:
    - Printed accuracy, precision, recall, F1, ROC-AUC, PR-AUC
    - results.csv        (per-image predictions)
    - confusion_matrix.png
    - roc_pr_curves.png
"""

import argparse
import os
import csv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from predict import load_model, predict_image


def collect_image_paths(test_dir):
    """
    Walks test_dir/good and test_dir/defective, returning a list of
    (filepath, true_label) tuples. true_label is 0 for defective, 1 for
    good, matching predict.py's class convention.
    """
    samples = []
    label_map = {"defective": 0, "good": 1}

    for class_name, label in label_map.items():
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: expected subfolder not found: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(class_dir, fname), label))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate the screw defect model (read-only).")
    parser.add_argument("--test_dir", type=str, required=True,
                         help="Path to test dir containing good/ and defective/ subfolders.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold for the defective class (default: 0.5).")
    parser.add_argument("--output_dir", type=str, default=".",
                         help="Where to write results.csv and plot images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model (read-only — no training will occur)...")
    model = load_model()

    samples = collect_image_paths(args.test_dir)
    if not samples:
        print("No images found. Check --test_dir structure (expects good/ and defective/ subfolders).")
        return

    print(f"Evaluating on {len(samples)} images...")

    y_true = []           # 1 = defective, 0 = good  (for sklearn "positive = defective")
    y_score = []           # P(defective)
    y_pred = []            # 1 = defective, 0 = good, at the given threshold
    per_image_rows = []

    for filepath, true_label_goodis1 in samples:
        image = Image.open(filepath)
        label_str, confidence = predict_image(model, image, threshold=args.threshold)

        is_defective_true = 1 if true_label_goodis1 == 0 else 0
        is_defective_pred = 1 if label_str == "defective" else 0

        # predict_image returns confidence relative to the predicted class;
        # recover P(defective) directly for a clean score to use in ROC/PR.
        prob_defective = confidence / 100.0 if label_str == "defective" else 1 - (confidence / 100.0)

        y_true.append(is_defective_true)
        y_score.append(prob_defective)
        y_pred.append(is_defective_pred)

        per_image_rows.append({
            "filepath": filepath,
            "true_label": "defective" if is_defective_true else "good",
            "predicted_label": label_str,
            "confidence_percent": f"{confidence:.1f}",
        })

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== EVALUATION RESULTS (read-only, model unchanged) ===")
    print(f"Threshold used:      {args.threshold}")
    print(f"Accuracy:            {accuracy:.3f}")
    print(f"Precision (defect):  {precision:.3f}")
    print(f"Recall (defect):     {recall:.3f}")
    print(f"F1 (defect):         {f1:.3f}")
    print(f"ROC-AUC:             {roc_auc:.3f}")
    print(f"PR-AUC (Avg Prec.):  {pr_auc:.3f}")
    print("Confusion matrix (rows=true, cols=pred, order=[defective, good]):")
    print(cm)

    # --- write per-image CSV ---
    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "true_label", "predicted_label", "confidence_percent"])
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"\nPer-image results written to: {csv_path}")

    # --- confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["defective", "good"])
    ax.set_yticklabels(["defective", "good"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot written to: {cm_path}")

    # --- ROC + PR curves ---
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(recall_curve, precision_curve, label=f"PR (AP = {pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    fig.tight_layout()
    curves_path = os.path.join(args.output_dir, "roc_pr_curves.png")
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"ROC/PR curve plot written to: {curves_path}")

    print(f"\nNote: metrics computed on {len(samples)} images — treat as noisy if this is a small test set.")


if __name__ == "__main__":
    main()
