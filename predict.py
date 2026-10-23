"""
Predict the disease of a single leaf photo with the trained DA-2 model.

Usage:
    python predict.py leaf.jpg
    python predict.py leaf.jpg --model outputs/plant_disease_subset.keras

Prints the top-3 predictions - screenshot this output for your report's
"Sample Predictions" section, alongside the leaf photo itself.
"""

import argparse
import json
from pathlib import Path

IMG_SIZE = (224, 224)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a leaf image (.jpg/.png)")
    parser.add_argument("--model", default="outputs/plant_disease_subset.keras")
    parser.add_argument("--classes", default="outputs/class_names.json")
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf

    class_names = json.loads(Path(args.classes).read_text())
    from train import load_trained_model
    model = load_trained_model(args.model, len(class_names))

    img = tf.keras.utils.load_img(args.image, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # No preprocess_input here: the model applies it internally (train.py
    # build_mobilenet), so it expects raw 0-255 pixels like the training data.

    probs = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]

    print(f"\nImage: {args.image}")
    for i in top3:
        label = class_names[i].replace("Tomato___", "Tomato - ").replace("_", " ")
        print(f"  {label:30s} {probs[i] * 100:6.2f}%")


if __name__ == "__main__":
    main()
