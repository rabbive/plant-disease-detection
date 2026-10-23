"""
Grad-CAM / CAM heatmap for the trained Plant Disease model.

Because the model head is GlobalAveragePooling -> Dense, the class
activation map is exact and simple: weight the last convolutional
feature maps (MobileNetV2 output, 7x7x1280) by the trained Dense
layer's weights for the predicted class, then overlay on the photo.
A good heatmap lights up the disease spots, not the background -
screenshot it for the report's explainability section, including
for one WRONG prediction (find one via predict.py on test images).

Usage:
    python gradcam.py leaf.jpg
    -> prints the prediction, saves gradcam_leaf.png next to it
"""

import argparse
import json
from pathlib import Path

IMG_SIZE = (224, 224)


def make_cam(model, img_array):
    """Return (predicted_index, probabilities, 224x224 heatmap 0..1)."""
    import numpy as np
    import tensorflow as tf

    # Locate the MobileNetV2 base (a nested Model) and the Dense head.
    base = next(l for l in model.layers
                if isinstance(l, tf.keras.Model) and "mobilenet" in l.name)
    dense = next(l for l in model.layers
                 if isinstance(l, tf.keras.layers.Dense))

    # Same path as the model at inference (augmentation is off then):
    # preprocess_input -> MobileNetV2 base. Calling the base directly
    # avoids tf.keras.Model(model.input, base.output), which Keras 3
    # rejects for a nested model ("not connected to inputs").
    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.convert_to_tensor(img_array, tf.float32))
    features = base(x, training=False)[0]                      # (7, 7, 1280)
    pooled = tf.reduce_mean(features, axis=(0, 1))             # GAP, (1280,)
    logits = (pooled[tf.newaxis] @ dense.kernel)[0] + dense.bias  # (num_classes,)
    probs = tf.nn.softmax(logits).numpy()
    pred = int(np.argmax(probs))

    weights = dense.kernel[:, pred].numpy()                    # (1280,)
    cam = np.maximum(features.numpy() @ weights, 0)            # ReLU, (7, 7)
    cam = cam / (cam.max() + 1e-8)
    return pred, probs, cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a leaf image (.jpg/.png)")
    parser.add_argument("--model", default="outputs/plant_disease_subset.keras")
    parser.add_argument("--classes", default="outputs/class_names.json")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import tensorflow as tf

    class_names = json.loads(Path(args.classes).read_text())
    model = tf.keras.models.load_model(args.model)

    img = tf.keras.utils.load_img(args.image, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    # No preprocess_input here: the model applies it internally (train.py
    # build_mobilenet), so it expects raw 0-255 pixels like the training data.

    pred, probs, cam = make_cam(model, arr)
    label = class_names[pred].replace("Tomato___", "Tomato - ").replace("_", " ")
    print(f"Prediction: {label} ({probs[pred] * 100:.2f}%)")

    heat = cv2.resize(cam, IMG_SIZE)
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original, 0.6, heat, 0.4, 0)

    out_path = args.out or str(Path(args.image).with_name(
        f"gradcam_{Path(args.image).stem}.png"))
    cv2.imwrite(out_path, overlay)
    print(f"Saved heatmap overlay -> {out_path}")


if __name__ == "__main__":
    main()
