"""
Plant Disease Detection - training script
===========================================
Trains a MobileNetV2 transfer-learning classifier on a 10-class
subset of PlantVillage (Tomato only) instead of all 38 classes.

Why a subset?
- Full PlantVillage = 54,305 images / 38 classes -> hours of training.
- 10 Tomato classes = ~18,000 images -> trains in ~25-30 min on a
  free Google Colab GPU, still one clean crop-specific system.

Split: 70% train / 15% validation / 15% test (per class, seeded).
The headline accuracy is the TEST accuracy - data the model never
saw during training or tuning.

Usage:
    python train.py --data-root /path/to/plantvillage
    python train.py --data-root /path/to/plantvillage --baseline
        (--baseline also trains a small CNN from scratch and prints
         a comparison table - the DA-3 experiment)

--data-root must contain the class folders (either directly, or
inside a train/ subfolder, as in the Kaggle "New Plant Diseases
Dataset"). Only the folders in SELECTED_CLASSES are used.

Outputs (in ./outputs):
    plant_disease_subset.keras  - trained model
    plant_disease_subset.weights.h5 - same weights, loadable on older Keras
    class_names.json            - label order used by predict.py / app.py
    training_curves.png         - accuracy / loss graph for the report
    confusion_matrix_test.png   - confusion matrix (test set)
    classification_report_test.txt - precision / recall / F1 per class (test set)
    results_summary.json        - val + test accuracy, per-model comparison
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

# 10 Tomato classes from PlantVillage - one crop, 9 diseases + healthy.
# Folder names must match the Kaggle / PlantVillage naming exactly.
SELECTED_CLASSES = [
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

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10          # frozen-base training
FINETUNE_EPOCHS = 5  # unfreeze step, set 0 to skip
BASELINE_EPOCHS = 10
SEED = 123
SPLIT = (0.70, 0.15, 0.15)  # train / val / test


def find_class_dirs(data_root: Path) -> dict:
    """Locate each selected class folder anywhere under data_root.

    Handles both layouts:
      data_root/Tomato___healthy/...                (PlantVillage raw)
      data_root/.../train/Tomato___healthy/...    (Kaggle augmented)
    Prefers a folder under a 'train' directory when both exist.
    Spaces count as underscores: raw PlantVillage names the spider-mite
    folder "Tomato___Spider_mites Two-spotted_spider_mite".
    """
    all_dirs = [p for p in data_root.rglob("*") if p.is_dir()]
    found = {}
    for class_name in SELECTED_CLASSES:
        candidates = [p for p in all_dirs
                      if p.name.replace(" ", "_") == class_name]
        if not candidates:
            raise FileNotFoundError(
                f"Class folder '{class_name}' not found under {data_root}. "
                "Check --data-root points at the downloaded PlantVillage dataset."
            )
        candidates.sort(key=lambda p: "train" not in p.parts)  # train/ first
        found[class_name] = candidates[0]
    return found


def build_splits(class_dirs: dict, split_root: Path) -> Path:
    """Copy/link images into split_root/{train,val,test}/<class>/
    with a per-class seeded 70/15/15 split."""
    if split_root.exists():
        shutil.rmtree(split_root)
    rng = random.Random(SEED)
    counts = {}
    for class_name, src in class_dirs.items():
        files = sorted(p for p in src.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        parts = {"train": files[:n_train],
                 "val": files[n_train:n_train + n_val],
                 "test": files[n_train + n_val:]}
        counts[class_name] = {k: len(v) for k, v in parts.items()}
        for split_name, split_files in parts.items():
            dst = split_root / split_name / class_name
            dst.mkdir(parents=True, exist_ok=True)
            for img in split_files:
                try:
                    os.link(img, dst / img.name)  # hardlink: no extra disk
                except OSError:
                    shutil.copy2(img, dst / img.name)
        print(f"  {class_name}: {counts[class_name]}")
    return split_root


def load_split(split_root: Path, name: str, shuffle=False):
    import tensorflow as tf
    ds = tf.keras.utils.image_dataset_from_directory(
        split_root / name, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        shuffle=shuffle, seed=SEED)
    return ds


def cache_compact(ds):
    """Cache images in RAM as uint8 instead of float32 (4x smaller).

    ~18k images at 224x224 is ~11 GB as float32 - about all of free Colab's
    RAM - but under 3 GB as uint8. Batches are cast back to float32
    (still 0-255) on the way out, so the model sees the same values.
    """
    import tensorflow as tf
    to_uint8 = lambda x, y: (tf.cast(tf.round(x), tf.uint8), y)
    to_float = lambda x, y: (tf.cast(x, tf.float32), y)
    return ds.map(to_uint8).cache().map(to_float)


def build_mobilenet(num_classes: int, weights="imagenet"):
    import tensorflow as tf
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights=weights)
    base.trainable = False
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def load_trained_model(model_path, num_classes: int):
    """Load the trained model, on any Keras 3 version.

    A .keras file saved by a newer Keras can fail to load on an older
    one (e.g. Colab or Hugging Face). The weights file saved next to
    it does load, so on failure rebuild the same architecture (no
    ImageNet download) and load the weights into it.
    """
    import tensorflow as tf
    model_path = Path(model_path)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as err:
        weights = model_path.with_name(model_path.stem + ".weights.h5")
        if not weights.exists():
            raise
        print(f"Could not load {model_path.name} with this Keras "
              f"({type(err).__name__}); rebuilding from {weights.name}")
        model, _ = build_mobilenet(num_classes, weights=None)
        model.load_weights(weights)
        return model


def build_baseline(num_classes: int):
    """Small CNN trained from scratch - the comparison baseline."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=IMG_SIZE + (3,)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="baseline_cnn")
    return model


def evaluate(model, ds, class_names, out_dir: Path, tag: str):
    """Confusion matrix + classification report on the given dataset."""
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_true, y_pred = [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

    report = classification_report(y_true, y_pred, target_names=class_names)
    (out_dir / f"classification_report_{tag}.txt").write_text(report)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    short = [c.replace("Tomato___", "").replace("_", " ") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=short, yticklabels=short)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({tag} set)")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{tag}.png", dpi=150)
    plt.close()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="Folder containing the PlantVillage dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--baseline", action="store_true",
                        help="Also train the from-scratch baseline CNN "
                             "and write a comparison into results_summary.json")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    # Imported here so --help works even without TensorFlow installed.
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---- 1. Prepare the 10-class 70/15/15 splits -----------------------
    print("\n[1/6] Building splits...")
    class_dirs = find_class_dirs(Path(args.data_root))
    split_root = build_splits(class_dirs, Path("data_split"))

    # ---- 2. Load datasets ----------------------------------------------
    print("\n[2/6] Loading datasets...")
    train_ds = load_split(split_root, "train", shuffle=True)
    val_ds = load_split(split_root, "val")
    test_ds = load_split(split_root, "test")
    class_names = train_ds.class_names
    print("Classes:", class_names)
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

    autotune = tf.data.AUTOTUNE
    train_ds = cache_compact(train_ds).shuffle(1000).prefetch(autotune)
    val_ds = cache_compact(val_ds).prefetch(autotune)
    test_ds = cache_compact(test_ds).prefetch(autotune)

    # ---- 3. MobileNetV2 transfer learning (base frozen) -----------------
    print("\n[3/6] Training MobileNetV2 (base frozen)...")
    model, base = build_mobilenet(len(class_names))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # ---- 4. Fine-tuning: unfreeze the top of the base -------------------
    if FINETUNE_EPOCHS > 0:
        print("\n[4/6] Fine-tuning top layers...")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history_ft = model.fit(train_ds, validation_data=val_ds,
                               epochs=FINETUNE_EPOCHS)
        for key in history.history:
            history.history[key] += history_ft.history[key]
    else:
        print("\n[4/6] Fine-tuning skipped")

    model.save(out_dir / "plant_disease_subset.keras")
    # Weights-only copy: loads on older Keras versions (see load_trained_model)
    model.save_weights(out_dir / "plant_disease_subset.weights.h5")
    print(f"Saved model -> {out_dir / 'plant_disease_subset.keras'} (+ .weights.h5)")
    summary["mobilenetv2"] = {
        "params": int(model.count_params()),
        "val_accuracy_last_epoch": float(history.history["val_accuracy"][-1]),
    }

    # ---- 5. Training curves + test-set evaluation -----------------------
    print("\n[5/6] Evaluating on the test set (never seen in training)...")
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, history.history["accuracy"], label="Train")
    ax1.plot(epochs_range, history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs_range, history.history["loss"], label="Train")
    ax2.plot(epochs_range, history.history["val_loss"], label="Validation")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    test_acc = evaluate(model, test_ds, class_names, out_dir, "test")
    summary["mobilenetv2"]["test_accuracy"] = test_acc
    print(f"MobileNetV2 test accuracy: {test_acc * 100:.2f}%")

    # ---- 6. Optional baseline comparison --------------------------------
    if args.baseline:
        print("\n[6/6] Training baseline CNN from scratch...")
        baseline = build_baseline(len(class_names))
        baseline.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])
        baseline.fit(train_ds, validation_data=val_ds, epochs=BASELINE_EPOCHS)
        b_test = evaluate(baseline, test_ds, class_names, out_dir, "baseline_test")
        summary["baseline_cnn"] = {
            "params": int(baseline.count_params()),
            "test_accuracy": b_test,
        }
        print(f"Baseline CNN test accuracy: {b_test * 100:.2f}%")
        print(f"Transfer learning gain: {(test_acc - b_test) * 100:+.2f} points")
    else:
        print("\n[6/6] Baseline skipped (pass --baseline to run it)")

    (out_dir / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs in {out_dir}/ - see results_summary.json")


if __name__ == "__main__":
    random.seed(SEED)
    main()
