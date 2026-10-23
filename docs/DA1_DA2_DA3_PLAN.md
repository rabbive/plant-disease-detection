# One Project, Three DAs — Plant Disease Detection Plan

Ashwanth confirmed on 2026-10-21 that DA-1, DA-2 and DA-3 all depend
on this one project. This maps the single project into three stages
so each DA has its own deliverable and nothing is repeated.

> Assumption to verify: this mapping works if the three DAs are
> stages of one course project (proposal → implementation → final),
> which is the usual VIT project-DA pattern. If any DA is instead a
> set of theory questions, that DA needs its question paper checked
> separately — the project alone won't cover it.

## DA-1 — Proposal, dataset and preprocessing
**Story:** what problem, what data, how images are prepared.
- Aim, real-world importance, literature survey (PlantVillage paper
  + 3–4 related student/research works)
- Dataset study: PlantVillage — 54,305 images, 38 classes, 14 crops;
  our scope = 6 Tomato classes and why (focused, trainable)
- Preprocessing demos (great DA-1 material, pure CV syllabus):
  resize to 224×224, pixel scaling, augmentation (flip/rotation/zoom),
  sample images per class, class-count table/graph
- Methodology block diagram: photo → preprocess → CNN → label
**Deliverable:** proposal report + dataset/preprocessing notebook cells

## DA-2 — Model building and training (the current setup)
**Story:** build the classifier and train it.
- Transfer learning explained: MobileNetV2 pre-trained on ImageNet,
  base frozen, new head (GAP → Dropout → Dense softmax, 6 outputs)
- Training: 80/20 split, sparse categorical cross-entropy, Adam,
  10 epochs @ 1e-3; accuracy/loss curves
- First results: validation accuracy, sample predictions
**Deliverable:** `plant_disease_da2.ipynb` (already built) + interim report

## DA-3 — Fine-tuning, evaluation, comparison and demo
**Story:** improve it, prove it, show it.
- Fine-tuning: unfreeze last 30 layers, 5 epochs @ 1e-5; before/after table
- Full evaluation: confusion matrix, per-class precision/recall/F1,
  which diseases get confused (early vs late blight) and why
- **Upgrade 1 — baseline comparison:** a small CNN trained from
  scratch on the same subset vs MobileNetV2 transfer learning.
  One comparison table turns this from "ran a model" into an experiment.
- **Upgrade 2 — Grad-CAM:** heatmap showing which part of the leaf
  the model looked at. Very "computer vision", ~20 lines, strong viva point.
- **Demo:** Gradio app — upload a leaf photo, get disease + confidence
  (pattern from the upstream repo's app.py, but with our real 6-class model)
- Limitations: plain-background dataset, tomato only, field photos harder
**Deliverable:** final report + demo screenshots / live demo

## Upgrades locked into the plan (2026-10-21, details in UPGRADES.md)
- **Tier 1 (must-do):** U1 baseline CNN-from-scratch comparison table ·
  U2 Grad-CAM heatmaps (incl. one failure case) · U3 expand 6 → 10
  Tomato classes before the final runs · U4 proper 70/15/15 split with
  test accuracy as the headline · U5 Gradio demo with per-disease
  remedy info cards
- **Tier 2 (pick 1–2):** second backbone (EfficientNetB0) · augmentation
  ablation · field-photo robustness test · TFLite size/latency table ·
  error gallery
- **Tier 3 (polish):** DA-1 dataset visuals · README results section ·
  2-minute demo video
- **Explicitly not doing:** all-38-classes as the main model, YOLO
  detection bolted on, a custom mobile app

## Is it worthy enough for all three? — honest gaps
- **Common topic:** many students pick plant disease. The comparison
  table + Grad-CAM are what separate this from the crowd.
- **Subset size:** frame 6 classes as a deliberate crop-specific
  system. If DA-3 time allows, expanding to 10–15 classes is a
  one-line change in the notebook (class list) + longer training.
- **High accuracy is expected** on this clean dataset — say so in
  the report and discuss it as a limitation instead of hiding it.
  Faculty respect that more than an unexplained 99%.
