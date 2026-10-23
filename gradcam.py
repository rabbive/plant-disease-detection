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


def forward(model, img_array):
    """One pass through the model, keeping what CAM needs.

    Returns (features, pooled, probs): the last conv feature maps
    (7x7x1280), their global average (1280,) and the softmax output.
    """
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
    return features.numpy(), pooled.numpy(), probs


def cam_for(model, features, class_index):
    """7x7 class activation map (0..1) for one class."""
    import numpy as np
    import tensorflow as tf

    dense = next(l for l in model.layers
                 if isinstance(l, tf.keras.layers.Dense))
    weights = np.asarray(dense.kernel)[:, class_index]         # (1280,)
    cam = np.maximum(features @ weights, 0)                    # ReLU, (7, 7)
    return cam / (cam.max() + 1e-8)


def make_cam(model, img_array):
    """Return (predicted_index, probabilities, 7x7 heatmap 0..1)."""
    import numpy as np

    features, _, probs = forward(model, img_array)
    pred = int(np.argmax(probs))
    return pred, probs, cam_for(model, features, pred)


def overlay(img, cam, alpha=0.4):
    """Blend a CAM onto a PIL image as a blue->red heatmap (PIL image out)."""
    import numpy as np
    from PIL import Image

    size = img.size
    heat = np.asarray(Image.fromarray(np.uint8(255 * cam)).resize(
        size, Image.BILINEAR), np.float32) / 255.0
    # Jet-style colormap: blue (cold) -> green -> red (hot)
    rgb = np.clip(np.stack([1.5 - np.abs(4 * heat - 3),
                            1.5 - np.abs(4 * heat - 2),
                            1.5 - np.abs(4 * heat - 1)], axis=-1), 0, 1)
    base = np.asarray(img.convert("RGB"), np.float32) / 255.0
    out = (1 - alpha) * base + alpha * rgb
    return Image.fromarray(np.uint8(255 * out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a leaf image (.jpg/.png)")
    parser.add_argument("--model", default="outputs/plant_disease_subset.keras")
    parser.add_argument("--classes", default="outputs/class_names.json")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    class_names = json.loads(Path(args.classes).read_text())
    from train import load_trained_model, prepare_image
    model = load_trained_model(args.model, len(class_names))

    arr, img = prepare_image(args.image)  # same resize as training
    # No preprocess_input here: the model applies it internally (train.py
    # build_mobilenet), so it expects raw 0-255 pixels like the training data.

    pred, probs, cam = make_cam(model, arr)
    label = class_names[pred].replace("Tomato___", "Tomato - ").replace("_", " ")
    print(f"Prediction: {label} ({probs[pred] * 100:.2f}%)")

    out_path = args.out or str(Path(args.image).with_name(
        f"gradcam_{Path(args.image).stem}.png"))
    overlay(img, cam).save(out_path)
    print(f"Saved heatmap overlay -> {out_path}")


if __name__ == "__main__":
    main()
