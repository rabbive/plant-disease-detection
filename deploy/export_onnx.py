"""
Export the trained model to ONNX for the static (in-browser) Space.

    pip install "tensorflow>=2.16" tf2onnx onnx onnxruntime pillow
    python deploy/export_onnx.py

Writes deploy/static/model.onnx, checks it against Keras on every
image in samples/ (same top-1, probabilities within 1e-4), and writes
deploy/static/classes.json (labels + app.py's info cards).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "deploy/static/model.onnx"


def inference_model():
    """Trained network minus the augmentation layers (no-ops at inference)."""
    import sys
    import tensorflow as tf
    sys.path.insert(0, str(REPO))
    from train import load_trained_model
    names = json.loads((REPO / "outputs/class_names.json").read_text())
    trained = load_trained_model(REPO / "outputs/plant_disease_subset.keras", len(names))
    base = next(l for l in trained.layers if l.name.startswith("mobilenetv2"))
    inputs = tf.keras.Input((224, 224, 3), name="image")  # raw 0-255 RGB
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)  # = preprocess_input
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = trained.layers[-1](x)  # Dense softmax
    return tf.keras.Model(inputs, outputs), trained


def main():
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    model, trained = inference_model()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, output_path=str(OUT))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(str(OUT))
    for img in sorted((REPO / "samples").glob("*.jpg")):
        arr = np.asarray(Image.open(img).convert("RGB").resize((224, 224)),
                         dtype=np.float32)[None]
        want = trained.predict(arr, verbose=0)[0]
        got = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
        assert got.argmax() == want.argmax(), img.name
        assert np.abs(got - want).max() < 1e-4, (img.name, np.abs(got - want).max())
    print("ONNX matches Keras on all samples.")

    # Labels in model output order + the app's info cards, for index.html
    from app import DISEASE_INFO
    names = json.loads((REPO / "outputs/class_names.json").read_text())
    classes = [{"label": n, **DISEASE_INFO[n]} for n in names]
    (OUT.parent / "classes.json").write_text(json.dumps(classes, indent=1, ensure_ascii=False))
    print(f"Wrote {OUT.parent / 'classes.json'}")


if __name__ == "__main__":
    main()
