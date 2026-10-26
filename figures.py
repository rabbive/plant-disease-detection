"""
Redraw the prediction figures in outputs/ from the committed model.

    python figures.py

Uses the same resize (train.prepare_image) and heatmap (gradcam.py) as
the app, so the figures match what the demo shows. The dataset figures
(class_distribution, dataset_samples) and training curves come from
training itself.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "outputs"


def short(label):
    return (label.replace("Tomato___", "").replace("Tomato_", "")
            .replace("Spider_mites_Two-spotted_spider_mite", "Spider mites")
            .replace("_", " "))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrong", default="spider_mites_1",
                        help="Misclassified sample for gradcam_wrong.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from app import decide
    from gradcam import cam_for, forward, overlay
    from train import load_trained_model, prepare_image

    names = json.loads((OUT / "class_names.json").read_text())
    model = load_trained_model(OUT / "plant_disease_subset.keras", len(names))
    order = ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot",
             "spider_mites", "target_spot", "yellow_leaf_curl_virus", "mosaic_virus", "healthy"]
    truth = dict(zip(order, ["Tomato___Bacterial_spot", "Tomato___Early_blight",
                             "Tomato___Late_blight", "Tomato___Leaf_Mold",
                             "Tomato___Septoria_leaf_spot",
                             "Tomato___Spider_mites_Two-spotted_spider_mite",
                             "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                             "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]))

    def run(path):
        arr, img = prepare_image(path)
        features, _, probs = forward(model, arr)
        return img, features, probs

    # 1. Predictions on one test leaf per class
    fig, axes = plt.subplots(2, 5, figsize=(20, 9.6))
    for ax, kind in zip(axes.flat, order):
        img, _, probs = run(REPO / "samples" / f"{kind}_1.jpg")
        k = int(np.argmax(probs))
        ok = names[k] == truth[kind]
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"True: {short(truth[kind])}\nPred: {short(names[k])} ({probs[k] * 100:.1f}%)",
                     color="darkgreen" if ok else "firebrick", fontsize=13)
    fig.suptitle("Predictions on test images the model never saw (green = correct)", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "sample_predictions.png", dpi=100)
    plt.close(fig)

    # 2. Grad-CAM for the first five classes
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.25))
    for col, kind in enumerate(order[:5]):
        img, features, probs = run(REPO / "samples" / f"{kind}_1.jpg")
        k = int(np.argmax(probs))
        axes[0, col].imshow(img); axes[0, col].set_title(short(truth[kind]), fontsize=13)
        axes[1, col].imshow(overlay(img, cam_for(model, features, k)))
        axes[1, col].set_title(f"Pred: {short(names[k])} {probs[k] * 100:.0f}%", fontsize=12)
        axes[0, col].axis("off"); axes[1, col].axis("off")
    fig.suptitle("Grad-CAM: where the model looks (red = most important)", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "gradcam_grid.png", dpi=120)
    plt.close(fig)

    # 3. A wrong prediction, and what the app shows for it
    img, features, probs = run(REPO / "samples" / f"{args.wrong}.jpg")
    k = int(np.argmax(probs))
    status, shown = decide(probs, names)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    axes[0].imshow(img)
    axes[0].set_title(f"True: {short(truth[args.wrong.rsplit('_', 1)[0]])}", fontsize=13)
    axes[1].imshow(overlay(img, cam_for(model, features, k)))
    axes[1].set_title(f"Model: {short(names[k])} ({probs[k] * 100:.1f}%)", color="firebrick", fontsize=13)
    axes[2].imshow(overlay(img, cam_for(model, features, shown)))
    title = (f"App: not clearly healthy,\ncheck for {short(names[shown])}"
             if status == "not_healthy" else f"App: {short(names[shown])} ({status})")
    axes[2].set_title(title, color="darkorange", fontsize=13)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("A wrong prediction, explained: raw model vs. what the app shows", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "gradcam_wrong.png", dpi=100)
    plt.close(fig)
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
