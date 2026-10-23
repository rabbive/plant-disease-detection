"""
Plant Disease Detection - Demo App (DA-3 showpiece)
=====================================================
Upload a tomato leaf photo -> disease name + confidence (top-3)
+ an info card (cause, symptoms, prevention) for the top prediction.

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
    Upload app.py + requirements.txt + the trained model file
    (plant_disease_subset.keras) + class_names.json to a Gradio Space.

Model lookup order:
    1. outputs/plant_disease_subset.keras
    2. ./plant_disease_subset.keras
If no model is found the app still opens, and every prediction
returns a clear "train the model first" message instead of crashing.
"""

import json
from pathlib import Path

IMG_SIZE = (224, 224)

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
    for p in [Path("outputs/plant_disease_subset.keras"),
              Path("plant_disease_subset.keras")]:
        if p.exists():
            return p
    return None


def _load_class_names(model_path):
    if model_path is not None:
        for p in [model_path.parent / "class_names.json",
                  Path("outputs/class_names.json"), Path("class_names.json")]:
            if p.exists():
                return json.loads(p.read_text())
    return DEFAULT_CLASSES


def main():
    import numpy as np
    import gradio as gr
    import tensorflow as tf

    model_path = _find_model()
    class_names = _load_class_names(model_path)
    from train import load_trained_model
    model = (load_trained_model(model_path, len(class_names))
             if model_path else None)
    if model is None:
        print("WARNING: no trained model found - train first "
              "(plant_disease.ipynb or train.py). App will open in notice mode.")

    def predict(img):
        """img: PIL image from Gradio. Returns (top-3 dict, info card markdown)."""
        if model is None:
            return {}, ("## No trained model yet\n"
                        "Run the training notebook first, then restart this app. "
                        "It looks for `outputs/plant_disease_subset.keras`.")
        if img is None:
            return {}, "Upload a tomato leaf photo first."
        img = img.convert("RGB").resize(IMG_SIZE)  # PNGs may be RGBA
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        # No preprocess_input here: the model applies it internally (train.py
        # build_mobilenet), so it expects raw 0-255 pixels like the training data.
        probs = model.predict(arr, verbose=0)[0]

        top3 = np.argsort(probs)[::-1][:3]
        scores = {}
        for i in top3:
            label = class_names[i]
            pretty = DISEASE_INFO.get(label, {}).get(
                "name", label.replace("Tomato___", "").replace("_", " "))
            scores[pretty] = float(probs[i])

        best = class_names[top3[0]]
        info = DISEASE_INFO.get(best, {})
        name = info.get("name", best)
        conf = probs[top3[0]] * 100
        card = (f"## {name} — {conf:.1f}% confidence\n\n"
                f"**Cause:** {info.get('cause', '-')}\n\n"
                f"**Symptoms to check:** {info.get('symptoms', '-')}\n\n"
                f"**Prevention / next step:** {info.get('prevention', '-')}\n\n"
                f"*Model output is a guide, not a diagnosis — confirm "
                f"visually, especially if confidence is below ~70%.*")
        return scores, card

    # One clickable example per class (test-split leaves, see samples/)
    samples = Path(__file__).resolve().parent / "samples"
    examples = [[str(p)] for p in sorted(samples.glob("*_1.jpg"))] or None

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a tomato leaf photo"),
        outputs=[gr.Label(num_top_classes=3, label="Top-3 predictions"),
                 gr.Markdown(label="Disease info")],
        title="Plant Disease Detection - Tomato",
        description="Upload a single tomato leaf photo. The model "
                    "(MobileNetV2, trained on PlantVillage) predicts the "
                    "disease and shows a short info card.",
        examples=examples,
        cache_examples=False,
    )
    demo.launch()  # in Colab, add share=True for a public link


if __name__ == "__main__":
    main()
