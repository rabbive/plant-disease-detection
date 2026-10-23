# Plant Disease Detection — Tomato (MobileNetV2 Transfer Learning)

A computer-vision project that classifies **tomato leaf diseases from a
single photo** using a CNN. Built on the PlantVillage dataset with
MobileNetV2 transfer learning, a proper train/val/test split, a
from-scratch baseline comparison, Grad-CAM explainability, and a
Gradio demo app.

> **Status:** trained. A reference run (CPU, raw PlantVillage tomato
> images) is committed in `outputs/`, so the demo app works straight
> after cloning. Every number below comes from that run — nothing in
> this README is a copied result.

## Why Tomato, why 10 classes

Full PlantVillage has 54,305 images across 38 classes (14 crops) —
training all of it takes hours. This project deliberately builds a
**crop-specific system**: all 10 Tomato classes (~18,000 images),
which trains in about 25–30 minutes on a free Colab GPU and still
covers the full pipeline end to end.

| # | Class | Type |
|---|-------|------|
| 1 | Healthy | — |
| 2 | Bacterial spot | Bacteria |
| 3 | Early blight | Fungus |
| 4 | Late blight | Oomycete |
| 5 | Leaf mold | Fungus |
| 6 | Septoria leaf spot | Fungus |
| 7 | Spider mites | Pest |
| 8 | Target spot | Fungus |
| 9 | Tomato mosaic virus | Virus |
| 10 | Tomato yellow leaf curl virus | Virus |

## How it works

```
leaf photo -> resize 224x224 -> MobileNetV2 preprocessing
           -> MobileNetV2 base (pre-trained on ImageNet)
           -> GlobalAveragePooling -> Dropout (0.2)
           -> Dense + softmax (10 outputs)
           -> disease name + confidence
```

Training is two-stage:
1. **Frozen base, 10 epochs, Adam lr 1e-3** — only the new head learns.
2. **Fine-tune, 5 epochs, Adam lr 1e-5** — the last 30 layers of
   MobileNetV2 unfreeze and adapt to leaf images.

Data split is **70% train / 15% validation / 15% test** per class.
The headline metric is **test accuracy** — images the model never
saw in training or tuning.

## Results (reference run)

18,160 images → 12,707 train / 2,719 val / 2,734 test.

| Model | Parameters | Test accuracy |
|-------|-----------:|--------------:|
| **MobileNetV2 (transfer learning)** | 2,270,794 | **91.33%** |
| Small CNN from scratch (baseline) | 94,538 | 85.37% |

- Best classes: Yellow Leaf Curl Virus (F1 0.98), Late blight and
  mosaic virus (0.94). Weakest: Early blight (recall 0.62), mostly
  confused with other brown-spot diseases.
- Full per-class report: `outputs/classification_report_test.txt`;
  figures: `outputs/*.png`; demo walkthrough: `docs/DEMO_SCRIPT.md`.
- Caveat: PlantVillage has several photos per leaf, and a random
  image-level split puts ~97% of test images on a leaf that also
  appears in training, so this accuracy is optimistic.

Sample leaves for the demo (all from the test split) are in `samples/`.

## Quickstart

### See the results — no training (2 minutes)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rabbive/plant-disease-detection/blob/main/plant_disease_results.ipynb)

`plant_disease_results.ipynb` loads the trained model from `outputs/`
and shows the charts, live predictions on `samples/`, Grad-CAM, an
upload-your-own-photo cell and the web app with a public link. Runs on
a normal Colab CPU; no Kaggle token needed.

### Train it yourself — Google Colab (free GPU)
1. Open `plant_disease.ipynb` in Colab
2. Runtime → Change runtime type → **GPU**
3. Run all cells (downloads the dataset via `kagglehub` — a free
   Kaggle account/API token is needed once)

### Train it yourself — local
```bash
pip install -r requirements.txt
# download PlantVillage (e.g. Kaggle "New Plant Diseases Dataset")
python train.py --data-root /path/to/plantvillage            # main model
python train.py --data-root /path/to/plantvillage --baseline # + baseline CNN comparison
python predict.py some_leaf.jpg                              # one prediction
python gradcam.py some_leaf.jpg                              # explainability heatmap
python app.py                                                # Gradio demo
```

## The demo app

`app.py` is the showpiece: upload a leaf photo and get the top-3
predictions plus an **info card** for the top disease — cause,
symptoms to check, and prevention/next steps for all 10 classes.
Runs locally (`python app.py` → http://127.0.0.1:7860), in Colab
(`%run app.py` prints a public share link), or as a free public
website on Hugging Face Spaces (below). It already includes the
trained model, and one example leaf per class is clickable under the
upload box.

### Put it online — Hugging Face Spaces (free, run on your computer)
1. Create a free account at https://huggingface.co/join
2. Create a token with **Write** access:
   https://huggingface.co/settings/tokens → *New token* → *Write*
3. In this repo folder:
   ```bash
   git pull                      # make sure outputs/ and samples/ are present
   pip install huggingface_hub
   python deploy/deploy_to_hf.py --space YOUR_HF_USERNAME/tomato-leaf-doctor
   ```
   Paste the token when asked. The script uploads only what the app
   needs (~20 MB: `app.py`, `train.py`, the model, sample leaves) and
   prints the link, e.g. `https://huggingface.co/spaces/YOUR_HF_USERNAME/tomato-leaf-doctor`.
4. The first build takes ~5 minutes (watch the *Logs* tab). After that
   the link works for anyone, on phone or laptop. Run step 3 again to
   push updates. Add `--dry-run` to see the file list without uploading.

If the model file fails to load on a different Keras version, the app
rebuilds the network and loads `plant_disease_subset.weights.h5`
instead — same predictions.

## Outputs (generated by training)

| File | What it is |
|------|------------|
| `outputs/plant_disease_subset.keras` | Trained model |
| `outputs/plant_disease_subset.weights.h5` | Same weights, loadable on older Keras |
| `outputs/class_names.json` | Label order for predict/app |
| `outputs/training_curves.png` | Accuracy / loss graphs |
| `outputs/confusion_matrix_test.png` | Confusion matrix on the test set |
| `outputs/classification_report_test.txt` | Per-class precision / recall / F1 |
| `outputs/results_summary.json` | Val + test accuracy, model comparison |

The committed reference run also adds presentation figures made from
the trained model: `class_distribution.png`, `dataset_samples.png`,
`sample_predictions.png`, `gradcam_grid.png`, `gradcam_wrong.png`,
`app_demo_*.png` (app screenshots), plus the baseline's report and
confusion matrix and the cleaned `train_log.txt`.

## Experiments included

- **Baseline comparison** — a small CNN trained from scratch on the
  same split, compared against transfer learning (params, test
  accuracy). The gap between them is the point: pre-training wins.
- **Grad-CAM (CAM)** — because the head is GAP → Dense, the class
  activation map is exact: last conv feature maps weighted by the
  Dense weights for the predicted class. Good heatmaps light up the
  disease spots, not the background.

## Limitations

- PlantVillage photos are single leaves on plain backgrounds taken
  in controlled conditions. Real field photos (soil, hands, many
  leaves, mixed light) will score lower — this is stated, not hidden.
- Tomato only. Extending to all 38 classes is a class-list change
  plus a much longer training run.
- The info cards in the app are general guidance, not a diagnosis
  or a treatment prescription.

## Repo structure

```
.
├── plant_disease.ipynb   # Colab notebook: full pipeline + experiments
├── plant_disease_results.ipynb  # Colab: results + live demo, no training
├── train.py              # Training script (MobileNetV2, optional --baseline)
├── predict.py            # Single-image prediction (top-3 + confidence)
├── gradcam.py            # Grad-CAM heatmap for one image
├── app.py                # Gradio demo with disease info cards
├── requirements.txt
├── outputs/              # reference run: model, charts, reports, logs
├── samples/              # 20 test-split leaves for the demo
├── deploy/               # Hugging Face Space: deploy script + config
└── docs/                 # demo script, DA stage plan, upgrades, report outline
```

## Credits

- Dataset: PlantVillage — Mohanty, Hughes & Salathé (2016),
  *Using Deep Learning for Image-Based Plant Disease Detection*.
  Kaggle copy used for training: "New Plant Diseases Dataset
  (Augmented)" by vipooool.
- Starting reference: the student project
  [RayanAIX/Plant-Disease-Detection](https://github.com/RayanAIX/Plant-Disease-Detection),
  rebuilt here with a real trained model, test split, baseline
  comparison, Grad-CAM and a working demo.
- Framework: TensorFlow / Keras, MobileNetV2 (ImageNet weights).

## License

MIT — see [LICENSE](LICENSE).
