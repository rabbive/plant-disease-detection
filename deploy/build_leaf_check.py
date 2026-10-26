"""
Build outputs/leaf_check.json - the "is this a tomato leaf?" check in app.py.

For each class, average the trained model's pooled MobileNetV2 features
(1280 numbers per image) over the TRAINING split. At prediction time the
app measures the cosine similarity between a photo's features and the
closest class average; non-leaf photos (faces, objects, blank walls)
land far from all ten.

The cut-off is chosen on the VALIDATION split: the similarity that
--keep (default 99%) of real validation leaves score above. The test
split is only used to report the result.

Usage (same data and split as train.py, so run it after training):
    python deploy/build_leaf_check.py --data-root /path/to/plantvillage
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def pooled_features(model, ds):
    """Pooled MobileNetV2 features, batched - same path as gradcam.forward."""
    import numpy as np
    import tensorflow as tf
    base = next(l for l in model.layers
                if isinstance(l, tf.keras.Model) and "mobilenet" in l.name)
    feats, labels = [], []
    for images, y in ds:
        x = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        feats.append(tf.reduce_mean(base(x, training=False), axis=(1, 2)).numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def similarity(feats, centroids):
    import numpy as np
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    return (f @ centroids.T).max(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["image", "leaf"], default="image",
                        help="Same --split the model was trained with")
    parser.add_argument("--outputs", default=str(REPO / "outputs"))
    parser.add_argument("--keep", type=float, default=0.99,
                        help="Share of validation leaves the check must accept")
    args = parser.parse_args()

    import numpy as np
    import train

    out = Path(args.outputs)
    class_names = json.loads((out / "class_names.json").read_text())
    model = train.load_trained_model(out / "plant_disease_subset.keras", len(class_names))

    split_root = Path("data_split_leafcheck")
    leaf_map = (json.loads(train.LEAF_MAP.read_text())
                if args.split == "leaf" else None)
    train.build_splits(train.find_class_dirs(Path(args.data_root)), split_root,
                       by_leaf=args.split == "leaf", leaf_map=leaf_map)

    f_train, y_train = pooled_features(model, train.load_split(split_root, "train"))
    centroids = np.stack([f_train[y_train == k].mean(axis=0)
                          for k in range(len(class_names))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    s_val = similarity(pooled_features(model, train.load_split(split_root, "val"))[0], centroids)
    s_test = similarity(pooled_features(model, train.load_split(split_root, "test"))[0], centroids)
    cut = float(np.quantile(s_val, 1 - args.keep))
    print(f"cut-off {cut:.4f}: accepts {np.mean(s_val >= cut):.2%} of val, "
          f"{np.mean(s_test >= cut):.2%} of test leaves")

    (out / "leaf_check.json").write_text(json.dumps({
        "min_similarity": round(cut, 4),
        "accepted_val": round(float(np.mean(s_val >= cut)), 4),
        "accepted_test": round(float(np.mean(s_test >= cut)), 4),
        "centroids": np.round(centroids, 5).tolist(),
    }))
    print(f"Saved {out / 'leaf_check.json'}")


if __name__ == "__main__":
    main()
