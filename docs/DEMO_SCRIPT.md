# Demo Script — what to show, in what order (~7 minutes)

Every number here comes from the reference run committed in
`outputs/` (raw PlantVillage tomato images, CPU, `train.py --baseline`,
seed 123). If you re-train, re-read `outputs/results_summary.json` and
`outputs/classification_report_test.txt` and update the numbers.

## Before the professor arrives
1. Pick where the app runs (the trained model is already in `outputs/`,
   no training needed for any of these):
   - **Best:** your Hugging Face Space link (see README → "Put it
     online") — works on the projector and on the professor's phone.
   - **Laptop:** `pip install -r requirements.txt`, `python app.py`,
     open http://127.0.0.1:7860.
   - **Colab:** open `plant_disease_results.ipynb` (README badge), run
     all; the last cell prints a public link. The same notebook walks
     through every chart below — a good backup if anything fails.
2. Do one test click on an example leaf so the model is warm (first
   prediction is slow; a sleeping Space takes ~1 minute to wake up).
3. Have these open in tabs: `outputs/class_distribution.png`,
   `outputs/training_curves.png`, `outputs/confusion_matrix_test.png`,
   `outputs/gradcam_grid.png`, `outputs/gradcam_wrong.png`.

## 1. The problem (30 s)
Farmers lose crops to leaf diseases; spotting them early needs an
expert. Goal: one leaf photo → disease name + what to do, in a second.

## 2. The data (1 min) — `class_distribution.png`, `dataset_samples.png`
- PlantVillage, all 10 tomato classes (9 diseases + healthy), 18,160
  images, split 70/15/15 per class → 12,707 train / 2,719 val / 2,734 test.
- Point at the imbalance: Yellow Leaf Curl Virus has 5,357 images,
  mosaic virus only 373. That's why we report per-class F1, not just
  accuracy.

## 3. The model (1 min) — README diagram
- MobileNetV2 pre-trained on ImageNet: it already knows edges and
  textures; we only teach it leaves.
- Stage 1: base frozen, train a new 10-class head (10 epochs).
- Stage 2: unfreeze the last 30 layers, fine-tune at lr 1e-5 (5 epochs).
- 2.27 M parameters, only 12,810 trained in stage 1.

## 4. Training (45 s) — `training_curves.png`
- Validation accuracy climbs 84.7% → 89.7% (frozen) → 91.4% (fine-tuned),
  peaking at 92.1%.
- Train and validation stay close → little overfitting.
- If asked about the dip at epoch 11: that's exactly where fine-tuning
  starts (top 30 layers unfrozen); training accuracy drops for one
  epoch and recovers, while validation accuracy keeps rising.

## 5. Results (1.5 min) — `confusion_matrix_test.png`
- **Test accuracy: 91.33%** on 2,734 images never used in training.
- Best: Yellow Leaf Curl Virus (F1 0.98), Late blight & mosaic virus (0.94).
- Weakest: **Early blight — recall 0.62**. Its errors spread over
  target spot (12), Septoria (11), late blight (10) — all brown-spot
  diseases that look alike, and early blight is one of the smaller classes.
- Healthy leaves are never misclassified (240/240), but 19 spider-mite
  and 18 target-spot leaves were called healthy — early-stage
  symptoms are subtle.

## 6. Transfer learning vs from scratch (45 s)

| Model | Parameters | Test accuracy |
|-------|-----------:|--------------:|
| MobileNetV2 (transfer learning) | 2,270,794 | 91.33% |
| Small CNN from scratch | 94,538 | 85.37% |

Same data, same split (baseline: 10 epochs; MobileNetV2: 10 + 5
fine-tuning) — the +6-point gap is what ImageNet pre-training buys.
Be fair about it: the baseline is 24× smaller and was still improving
at epoch 10 (validation 87.2%), so with more epochs the gap would shrink.

## 7. Explainability (1 min) — `gradcam_grid.png`, `gradcam_wrong.png`
- Grad-CAM shows where the model looks: on late blight and Septoria
  the heat sits on the lesions, not the background.
- Show the wrong one: a target-spot leaf with barely visible lesions,
  called healthy at only 56% confidence — the model was unsure, and
  the app tells the user to double-check anything below ~70%.

## 8. Live demo (1.5 min) — the app
Upload from `samples/` (all are test images the model never saw):

| Upload | Expect |
|--------|--------|
| `late_blight_1.jpg` | Late Blight, ~98%, info card |
| `leaf_mold_1.jpg` | Leaf Mold, ~97% |
| `septoria_leaf_spot_2.jpg` | Septoria leaf spot, ~100% |
| `healthy_1.jpg` | Healthy, ~100% |
| `spider_mites_1.jpg` | **Wrong on purpose:** says Healthy (~86%) — be upfront about it |

Screenshots of the app working are in `outputs/app_demo_*.png` in
case the live demo fails.

## 9. Limitations — say these before you're asked
- **Same-leaf overlap, and the check:** PlantVillage has ~4 photos of
  each leaf, so the random image split put ~97% of test images on a
  leaf that also has a photo in training. Retrained with a leaf-grouped
  split (no leaf on both sides): 91.96% vs 91.33%, the same within
  noise. Say this before you're asked: it shows the number holds up.
- Plain lab backgrounds; real field photos will score lower.
- Tomato only; class imbalance (373 vs 5,357 images).

## Likely viva questions
- *Why MobileNetV2?* Small (2.3 M params), fast on CPU/phones,
  strong ImageNet features.
- *Why freeze first, then fine-tune?* A random new head would send
  large, noisy gradients into the pre-trained layers; train the head
  first, then adapt the top layers gently with a 100× smaller lr.
- *Why test accuracy, not validation?* Validation guided our choices
  (epochs, fine-tuning); test was touched once, at the end.
- *Isn't the test set leaking?* Checked: a leaf-grouped split gives
  91.96%, the same as 91.33% within noise (`train.py --split leaf`).
- *What would you do next?* Class weights or oversampling for small
  classes, field photos for testing, confidence calibration.
