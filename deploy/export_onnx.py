"""
Export the trained model to ONNX for the static (in-browser) Space.

    pip install "tensorflow>=2.16" tf2onnx onnx onnxruntime pillow
    python deploy/export_onnx.py

Writes to deploy/static/:
  model.onnx    image (raw 0-255 RGB, 224x224) -> three outputs:
                  probs  (1, 10)         softmax, as the Keras model
                  pooled (1, 1280)       features for the leaf check
                  cams   (1, 7, 7, 10)   class activation maps (Grad-CAM)
  classes.json  labels in output order + app.py's info cards
  rules.json    app.py's safety thresholds + outputs/leaf_check.json

It checks the ONNX outputs against Keras / gradcam.py on every image in
samples/ before writing the JSON files.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "deploy/static/model.onnx"
sys.path.insert(0, str(REPO))


def inference_model():
    """Trained network minus the augmentation layers (no-ops at inference)."""
    import tensorflow as tf
    from train import load_trained_model
    names = json.loads((REPO / "outputs/class_names.json").read_text())
    trained = load_trained_model(REPO / "outputs/plant_disease_subset.keras", len(names))
    base = next(l for l in trained.layers if l.name.startswith("mobilenetv2"))
    dense = trained.layers[-1]  # Dense softmax
    inputs = tf.keras.Input((224, 224, 3), name="image")  # raw 0-255 RGB
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)  # = preprocess_input
    features = base(x, training=False)                            # (7, 7, 1280)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="pooled")(features)
    probs = tf.keras.layers.Activation("linear", name="probs")(dense(pooled))
    # CAM: the dense weights applied at every 7x7 position (no bias, no softmax)
    cam_layer = tf.keras.layers.Dense(len(names), use_bias=False, name="cams")
    cams = cam_layer(features)
    cam_layer.set_weights([dense.get_weights()[0]])
    return tf.keras.Model(inputs, [probs, pooled, cams]), trained


def main():
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx
    from gradcam import forward
    from train import prepare_image

    model, trained = inference_model()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, output_path=str(OUT))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(str(OUT))
    out_names = [o.name for o in sess.get_outputs()]
    print("ONNX outputs:", out_names)
    for img in sorted((REPO / "samples").glob("*.jpg")):
        arr, _ = prepare_image(img)
        got = dict(zip(out_names, sess.run(None, {sess.get_inputs()[0].name: arr})))
        features, pooled, probs = forward(trained, arr)
        want_cams = features @ trained.layers[-1].get_weights()[0]
        for key, want in [("probs", probs), ("pooled", pooled), ("cams", want_cams)]:
            diff = np.abs(got[key][0] - want).max()
            assert diff < 1e-3 * max(1, np.abs(want).max()), (img.name, key, diff)
        assert got["probs"][0].argmax() == probs.argmax(), img.name
    print("ONNX matches Keras on all samples (probs, pooled features, CAMs).")

    # Labels in model output order + the app's info cards, for index.html
    import app
    names = json.loads((REPO / "outputs/class_names.json").read_text())
    classes = [{"label": n, **app.DISEASE_INFO[n]} for n in names]
    (OUT.parent / "classes.json").write_text(json.dumps(classes, indent=1, ensure_ascii=False))
    print(f"Wrote {OUT.parent / 'classes.json'}")

    # Same safety rules as app.py, so the browser and the Gradio app agree
    leaf = json.loads((REPO / "outputs/leaf_check.json").read_text())
    rules = {"healthy_min_prob": app.HEALTHY_MIN_PROB,
             "unsure_below": app.UNSURE_BELOW,
             "healthy_index": names.index("Tomato___healthy"),
             "min_similarity": leaf["min_similarity"],
             "centroids": leaf["centroids"]}
    (OUT.parent / "rules.json").write_text(json.dumps(rules))
    print(f"Wrote {OUT.parent / 'rules.json'}")


if __name__ == "__main__":
    main()
