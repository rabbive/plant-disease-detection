# Upgrades & Extra Stuff — Making One Project Worthy of DA-1, DA-2 and DA-3

Added 2026-10-21. The base project (6-class MobileNetV2) is a good
DA-2. For three DAs it needs an experiment, an explanation, and a
demo. Tiers below are in priority order — do Tier 1 fully before
touching Tier 2. Time estimates are estimates, for Colab GPU work.

## Tier 1 — Must-do (these are what make it worthy)

### U1. Baseline comparison: small CNN from scratch vs MobileNetV2
- Train a simple 3-conv CNN (no pre-training) on the *same* subset,
  same split, same epochs budget.
- One table: Model | Trainable params | Training time | Val accuracy.
- Expected story: transfer learning wins by a wide margin with less
  training — that gap *is* your result, and your DA-3 experiment.
- ~30–45 min. Notebook: 2 extra cells (build + train) + 1 table cell.

### U2. Grad-CAM explainability
- For 3–4 test leaves, overlay a heatmap of where the model looked.
- If the heat lights up the disease spots (not the background),
  that's your proof the model learned disease features.
- Include one *wrong* prediction's Grad-CAM too — analysing a
  failure honestly scores better than hiding it.
- ~20 lines using the last conv layer of MobileNetV2
  (`Conv_1` / `expanded_conv_project_BN` output) + OpenCV overlay.
- ~30 min. Goes in DA-3 report + viva talking point #1.

### U3. Expand 6 → 10 classes (all Tomato)
Add: `Tomato___Septoria_leaf_spot`, `Tomato___Spider_mites_Two-spotted_spider_mite`,
`Tomato___Tomato_mosaic_virus`, `Tomato___Tomato_Yellow_Leaf_Curl_Virus`.
- One-line change to the class list in the notebook / train.py.
- Kills the "only 6 classes is too small" objection while keeping
  the one-crop story clean. Training grows to ~25–30 min (estimate).
- Do this *before* the final DA-2/DA-3 training runs so every
  number in every report comes from the same 10-class model.

### U4. Proper test split + full metrics
- Change split from 80/20 to **70 train / 15 validation / 15 test**.
- Headline accuracy in reports = **test** accuracy (data the model
  never saw in any form). Validation is only for watching training.
- Report per-class precision / recall / F1 (already generated) and
  name the worst class + most-confused pair with a reason.
- ~15 min of code change; runs inside the normal training.

### U5. Working demo app with remedy info (Gradio)
- Upload leaf photo → disease name + confidence (top-3) + a short
  info card per disease: cause, visible symptoms, prevention /
  treatment pointers (from standard agriculture extension sources,
  cited in the report).
- This is the DA-3 demo: it turns a notebook into a product a
  farmer could actually use, and it's what you show live.
- Pattern exists in the upstream repo's app.py — rebuild it around
  our real trained model instead of its 2-class placeholder.
- ~45–60 min including writing the 10 disease info cards.

## Tier 2 — Worth it if time allows (pick 1–2)

### E1. Second backbone comparison
Add EfficientNetB0 (or ResNet50) trained the same way.
Table becomes 3 models — the strongest single table in the report.
~30 min extra training per model.

### E2. Augmentation ablation
Train MobileNetV2 twice: with augmentation vs without.
Shows augmentation's effect on the train/val gap (overfitting).
Two short runs, one small table, very "research method" for DA-3.

### E3. Field-photo robustness test
Take 10–15 real leaf photos (phone photos / web images with soil,
hands, multiple leaves) and test the model on them.
Accuracy will drop — report it honestly in Limitations with
example failures. This pre-empts the faculty's hardest question.

### E4. TFLite conversion + efficiency numbers
Convert the final model to TFLite; report model size (MB) and
average prediction time per image on CPU.
One short table; supports the "runs on a farmer's phone" claim.

### E5. Error gallery
A figure of 6 misclassified test images with true vs predicted
labels. Pairs with U2's failure Grad-CAM in the discussion section.

## Tier 3 — Presentation / DA-1 material (cheap, high polish)

### P1. Dataset visuals for DA-1
Class-distribution bar chart, a 6×2 sample-image grid, and
preprocessing figures (original → resized → HSV → edges) for one
diseased leaf. Pure syllabus CV, fills DA-1 with real content.

### P2. Clean repo + README results section
Final numbers, graphs and demo screenshot committed to the repo
README, so the GitHub link itself tells the whole story if a
faculty member opens it.

### P3. 2-minute demo video
Screen-record the Gradio app classifying 3 leaves. Insurance if
the live demo fails on the day, and reusable in every DA review.

## What NOT to add
- **All 38 classes as the main model** — hours of training, worse
  story, and the comparison/ablation work matters more.
- **Object detection (YOLO) on top** — different task, doubles the
  work, muddles a classification project's marks.
- **A custom mobile app** — the Gradio demo already proves the
  use case; an app is weeks for the same demo value.
