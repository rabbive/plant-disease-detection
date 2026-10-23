"""
Plant Disease Detection - Demo App (DA-3 showpiece)
=====================================================
Upload a tomato leaf photo -> disease name + confidence (top-3),
a Grad-CAM heatmap of where the model looked, and an info card
(cause, symptoms, prevention) for the top prediction.

Three safety rules sit on top of the raw model output (thresholds
were picked on the validation split, see decide() below):
  - photos that don't look like a tomato leaf are rejected
  - "Healthy" is only shown when the model is >= 90% sure, otherwise
    the most likely disease is shown instead
  - answers under 60% confidence are flagged as unsure

This is the thing you SHOW in a review / viva - not the notebook.

Run locally:
    pip install -r requirements.txt gradio
    python train.py --data-root /path/to/plantvillage   # first, once
    python app.py

Run in Colab:
    !pip -q install gradio
    # after training, in a new cell:  %run app.py
    # (Gradio prints a public share link you can open on your phone)

Deploy free on Hugging Face Spaces:
    python deploy/deploy_to_hf.py --space YOUR_HF_USERNAME/tomato-leaf-doctor

Model lookup order:
    1. outputs/plant_disease_subset.keras
    2. ./plant_disease_subset.keras
If no model is found the app still opens, and every prediction
returns a clear "train the model first" message instead of crashing.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Decision thresholds, tuned on the validation split (never the test
# split) - see "Safety rules" in the README for the numbers behind them.
HEALTHY_MIN_PROB = 0.90  # below this, show the top disease instead of "Healthy"
UNSURE_BELOW = 0.60      # top-1 under this -> "not sure, check by eye"
# The "is it a leaf?" cut-off lives in outputs/leaf_check.json.

DEFAULT_CLASSES = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

# Short field-guide cards. General guidance summarised from standard
# plant-pathology / agriculture-extension references - cite those in
# the report, and present this as information, not a prescription.
DISEASE_INFO = {
    "Tomato___healthy": {
        "name": "Healthy",
        "cause": "No disease detected.",
        "symptoms": "Leaf looks normal - even green colour, no spots or curling.",
        "prevention": "Keep doing what works: spaced plants, watering at the base, regular inspection.",
    },
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "cause": "Bacteria (Xanthomonas species), spread by splashing water and infected seed.",
        "symptoms": "Small dark water-soaked spots on leaves and fruit, turning brown/black.",
        "prevention": "Avoid overhead watering, rotate crops, remove infected leaves, copper-based sprays are commonly used.",
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "cause": "Fungus Alternaria solani, common in warm, humid weather.",
        "symptoms": "Brown spots with concentric 'target' rings, usually on older lower leaves first, with yellowing around the spot.",
        "prevention": "Mulch to stop soil splash, space plants for airflow, remove infected lower leaves, fungicide if it spreads.",
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "cause": "Oomycete Phytophthora infestans (the Irish potato famine organism), spreads very fast in cool wet weather.",
        "symptoms": "Large water-soaked pale-green to brown patches, white mould under the leaf, whole plant can collapse in days.",
        "prevention": "Act fast: remove and destroy infected plants (do not compost), improve airflow, preventive fungicide in wet spells.",
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "cause": "Fungus Passalora fulva, thrives in high humidity, common in greenhouses.",
        "symptoms": "Pale yellow spots on top of the leaf, olive-green velvety mould on the underside.",
        "prevention": "Ventilate / reduce humidity, space plants, water at the base, remove affected leaves.",
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "cause": "Fungus Septoria lycopersici, spread by rain and water splash from soil.",
        "symptoms": "Many small round spots with grey centres and dark borders, mostly on lower leaves; leaves yellow and drop.",
        "prevention": "Mulch, avoid wetting leaves, remove infected foliage, rotate crops, fungicide if severe.",
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "name": "Spider Mites (Two-spotted)",
        "cause": "Pest mite Tetranychus urticae, worst in hot dry weather - not a fungus or bacteria.",
        "symptoms": "Tiny yellow/white stippling on leaves, fine webbing under leaves, leaves bronze and dry out.",
        "prevention": "Spray undersides with water, encourage predatory mites, neem or miticide if the infestation is heavy.",
    },
    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "cause": "Fungus Corynespora cassiicola, favours warm humid conditions.",
        "symptoms": "Brown spots with concentric rings and a light centre, on leaves, stems and fruit.",
        "prevention": "Improve airflow, avoid leaf wetness, remove infected material, fungicide if it spreads.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "cause": "Virus (ToMV), spread by hands, tools, and infected seed. No cure once infected.",
        "symptoms": "Mottled light/dark green mosaic pattern, leaves curl and distort, plant growth is stunted.",
        "prevention": "Remove and destroy infected plants, disinfect tools and wash hands before handling plants, use resistant varieties.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "cause": "Virus (TYLCV), carried by whiteflies. No cure once infected.",
        "symptoms": "Leaves curl upwards, turn yellow at the edges, plant is stunted with a bushy look, heavy flower drop.",
        "prevention": "Control whiteflies (nets, traps, sprays), remove infected plants, use resistant varieties.",
    },
}


def _find_model():
    for p in [HERE / "outputs/plant_disease_subset.keras",
              HERE / "plant_disease_subset.keras"]:
        if p.exists():
            return p
    return None


def _load_class_names(model_path):
    if model_path is not None:
        for p in [model_path.parent / "class_names.json",
                  HERE / "outputs/class_names.json", HERE / "class_names.json"]:
            if p.exists():
                return json.loads(p.read_text())
    return DEFAULT_CLASSES


def _load_leaf_check(model_path):
    """Per-class mean feature vectors + similarity cut-off, or None.

    Built from the training split by deploy/build_leaf_check.py. Without
    it the app still works, it just can't reject non-leaf photos.
    """
    import numpy as np
    if model_path is None:
        return None
    p = model_path.parent / "leaf_check.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    c = np.asarray(d["centroids"], np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True), float(d["min_similarity"])


def leaf_similarity(pooled, centroids):
    """Cosine similarity of a photo's features to the closest class average.

    Tomato leaves land close to one of the 10 class averages; a face, a
    car or a blank wall lands far from all of them. Softmax confidence
    can't tell these apart - the model is often 90%+ "sure" on a selfie.
    """
    import numpy as np
    v = pooled / (np.linalg.norm(pooled) + 1e-8)
    return float((centroids @ v).max())


def decide(probs, class_names, similarity=None, min_similarity=None):
    """Turn raw model output into what the app shows.

    Returns (status, index): status is "not_leaf", "not_healthy"
    (leans healthy, but below the bar), "unsure" or "ok"; index is the
    class to report (None for "not_leaf").
    """
    import numpy as np
    if (similarity is not None and min_similarity is not None
            and similarity < min_similarity):
        return "not_leaf", None
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    healthy = class_names.index("Tomato___healthy") if "Tomato___healthy" in class_names else -1
    # A sick leaf called healthy is the costliest mistake (the farmer does
    # nothing), so "Healthy" needs a high bar; otherwise report the most
    # likely disease. Val split: sick->healthy errors 54 -> 12.
    if best == healthy and probs[best] < HEALTHY_MIN_PROB:
        return "not_healthy", int(order[1])
    if probs[best] < UNSURE_BELOW:
        return "unsure", best
    return "ok", best


def _pretty(label):
    return DISEASE_INFO.get(label, {}).get(
        "name", label.replace("Tomato___", "").replace("_", " "))


def info_card(label, prob, status, healthy_prob=None):
    info = DISEASE_INFO.get(label, {})
    name = info.get("name", _pretty(label))
    if status == "not_healthy":
        card = (f"## Not clearly healthy — check for {name}\n\n"
                f"> ⚠️ The model leans *Healthy* ({healthy_prob * 100:.0f}%), "
                f"but that is below the {HEALTHY_MIN_PROB:.0%} bar for calling a "
                f"leaf healthy. Missing a disease costs more than a second look, "
                f"so here is the most likely disease ({prob * 100:.0f}%). "
                f"Compare the leaf with the symptoms below.\n\n")
    else:
        card = f"## {name} — {prob * 100:.1f}% confidence\n\n"
    if status == "unsure":
        card += ("> ⚠️ **Not sure.** The model isn't confident about this "
                 "leaf. Treat this as a best guess and check the symptoms "
                 "below by eye, or try a sharper photo of one leaf.\n\n")
    card += (f"**Cause:** {info.get('cause', '-')}\n\n"
             f"**Symptoms to check:** {info.get('symptoms', '-')}\n\n"
             f"**Prevention / next step:** {info.get('prevention', '-')}\n\n"
             f"*Model output is a guide, not a diagnosis.*")
    return card


NOT_LEAF_CARD = (
    "## This doesn't look like a tomato leaf\n\n"
    "The model was trained on photos of **one tomato leaf on a plain "
    "background**, and this photo looks too different from those for its "
    "answer to mean anything. Try:\n\n"
    "- one leaf filling most of the frame\n"
    "- a plain background (paper, table, hand)\n"
    "- even light, in focus")

ABOUT_MD = """
### How it works
```
leaf photo -> 224x224 -> MobileNetV2 (ImageNet pre-trained, last 30 layers fine-tuned)
           -> global average pooling -> dense softmax (10 classes)
           -> safety rules -> disease + info card + heatmap
```
- **Data:** PlantVillage, all 10 tomato classes, 18,160 images, split
  70 / 15 / 15 into train / validation / test.
- **Training:** 10 epochs with the base frozen, then 5 epochs fine-tuning
  the last 30 layers at a 100x lower learning rate.
- **Heatmap (Grad-CAM):** the head is pooling -> dense, so the class
  activation map is exact: the last conv feature maps weighted by the
  dense weights of the shown class. Red = the area that drove the answer.

### Results on 2,734 held-out test images
| Model | Parameters | Test accuracy |
|---|---:|---:|
| **MobileNetV2 (transfer learning)** | 2.27 M | **91.33%** |
| Small CNN from scratch | 0.09 M | 85.37% |

### Safety rules (tuned on the validation split, checked on test)
| Rule | Why | Effect on test set |
|---|---|---|
| Show "Healthy" only at ≥ 90% confidence | a sick leaf called healthy means no treatment | sick-called-healthy 56 → 7, accuracy 91.4% → 92.8% |
| Flag answers below 60% as "not sure" | the model is right only ~45% of the time there | with rule 1: 6.8% flagged; 95.2% accuracy on the rest |
| Reject photos far from every class | softmax is confidently wrong on non-leaves | accepts 99.5% of test leaves, rejected 33/33 non-leaf images |
"""

LIMITS_MD = """
### Limitations
- Trained on PlantVillage: single leaves, plain backgrounds, lab light.
  Field photos (soil, many leaves, shadows) will score lower.
- Tomato only, 10 classes. A disease outside this list gets mapped to
  the closest one it knows.
- Weakest class: Early blight (recall 0.62), confused with other
  brown-spot diseases (target spot, Septoria, late blight).
- The info cards are general guidance summarised from agriculture
  extension sources, not a diagnosis or a treatment prescription.

Source code, notebooks and full results:
[github.com/rabbive/plant-disease-detection](https://github.com/rabbive/plant-disease-detection)
"""


def main():
    import numpy as np
    import gradio as gr
    from gradcam import cam_for, forward, overlay

    model_path = _find_model()
    class_names = _load_class_names(model_path)
    from train import load_trained_model, prepare_image
    model = (load_trained_model(model_path, len(class_names))
             if model_path else None)
    if model is None:
        print("WARNING: no trained model found - train first "
              "(plant_disease.ipynb or train.py). App will open in notice mode.")
    leaf_check = _load_leaf_check(model_path)

    def predict(img):
        """img: PIL image -> (top-3 dict, heatmap image, info card markdown)."""
        if model is None:
            return {}, None, ("## No trained model yet\n"
                              "Run the training notebook first, then restart this app. "
                              "It looks for `outputs/plant_disease_subset.keras`.")
        if img is None:
            return {}, None, "Upload a tomato leaf photo first."
        arr, img = prepare_image(img)  # same resize as training
        # No preprocess_input here: the model applies it internally (train.py
        # build_mobilenet), so it expects raw 0-255 pixels like the training data.
        features, pooled, probs = forward(model, arr)

        sim, min_sim = None, None
        if leaf_check is not None:
            sim, min_sim = leaf_similarity(pooled, leaf_check[0]), leaf_check[1]
        status, best = decide(probs, class_names, sim, min_sim)
        if status == "not_leaf":
            return {}, None, NOT_LEAF_CARD

        top3 = np.argsort(probs)[::-1][:3]
        scores = {_pretty(class_names[i]): float(probs[i]) for i in top3}
        heat = overlay(img, cam_for(model, features, best))
        healthy_prob = (probs[class_names.index("Tomato___healthy")]
                        if status == "not_healthy" else None)
        return scores, heat, info_card(class_names[best], probs[best], status,
                                       healthy_prob)

    # Clickable examples: all test-split leaves in samples/ (never trained on)
    examples = sorted(str(p) for p in (HERE / "samples").glob("*.jpg"))

    with gr.Blocks(title="Tomato Leaf Disease Detector") as demo:
        gr.Markdown("# 🍅 Tomato Leaf Disease Detector\n"
                    "Upload a photo of **one tomato leaf**. The model "
                    "(MobileNetV2, trained on PlantVillage) names the disease, "
                    "shows where it looked, and gives a short info card.")
        with gr.Tab("Diagnose"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Tomato leaf photo",
                                     height=320)
                    button = gr.Button("Diagnose", variant="primary")
                with gr.Column():
                    # No big heading: the card below says what the app concludes, which
                    # can differ from the raw top-1 (the Healthy rule).
                    label = gr.Label(num_top_classes=3, label="Top-3 predictions (raw model output)",
                                     show_heading=False)
                    heatmap = gr.Image(label="Where the model looked (red = most important)",
                                       height=320, interactive=False)
                    card = gr.Markdown()
            outputs = [label, heatmap, card]
            if examples:
                gr.Examples(examples, inputs=image, outputs=outputs, fn=predict,
                            run_on_click=True, cache_examples=False,
                            label="Or click a test leaf (never seen in training)",
                            examples_per_page=20)
            button.click(predict, inputs=image, outputs=outputs)
            image.upload(predict, inputs=image, outputs=outputs)
            image.clear(lambda: ({}, None, ""), outputs=outputs)
        with gr.Tab("How it works"):
            gr.Markdown(ABOUT_MD)
            figures = [(HERE / "outputs" / f, cap) for f, cap in [
                ("confusion_matrix_test.png", "Confusion matrix (test set)"),
                ("training_curves.png", "Training curves"),
                ("gradcam_grid.png", "Grad-CAM on test leaves"),
            ]]
            figures = [(str(p), cap) for p, cap in figures if p.exists()]
            if figures:
                gr.Gallery(figures, columns=3, height=420, object_fit="contain",
                           label="Figures (click to enlarge)", allow_preview=True)
        with gr.Tab("Limitations"):
            gr.Markdown(LIMITS_MD)
    demo.launch()  # in Colab, add share=True for a public link


if __name__ == "__main__":
    main()
