# DA-2 Report Outline — Plant Disease Detection using CNN

Use simple English, short sentences. Fill every [bracket] from *your*
actual run — do not copy numbers from other projects.

## 1. Aim
To classify tomato leaf diseases from a leaf image using a
convolutional neural network with transfer learning.

## 2. Introduction (half page)
- Plant diseases reduce crop yield; manual inspection is slow.
- Computer vision can classify a leaf photo in under a second.
- This project classifies 10 tomato leaf conditions.

## 3. Dataset
- PlantVillage dataset: 54,305 images, 38 classes, 14 crop species.
- This work uses all 10 Tomato classes:
  healthy, bacterial spot, early blight, late blight, leaf mold,
  Septoria leaf spot, spider mites, target spot, mosaic virus,
  yellow leaf curl virus — about [your count] images.
- Split: 70% training, 15% validation, 15% test (per class, seeded). Images resized to 224×224.
- Add: one sample image from 3–4 classes (screenshot from the dataset).

## 4. Methodology
1. **Preprocessing** — resize to 224×224, MobileNetV2 preprocessing.
2. **Augmentation** — random flip, rotation, zoom (training only).
3. **Model** — MobileNetV2 pre-trained on ImageNet, top removed;
   GlobalAveragePooling → Dropout (0.2) → Dense softmax (10 outputs).
   Paste the `model.summary()` output here.
4. **Training** — Stage 1: base frozen, 10 epochs, Adam lr 1e-3.
   Stage 2: last 30 layers unfrozen, 5 epochs, Adam lr 1e-5.
   Loss: sparse categorical cross-entropy.

## 5. Results
- Final test accuracy (headline): [your %]; validation accuracy: [your %] (from notebook step 11 output; validation from steps 8–9).
- Insert `training_curves.png` — one line explaining both graphs
  (accuracy rises, loss falls, small train/val gap = little overfitting).
- Insert `confusion_matrix.png` — name the pair your model confused
  most, e.g. early blight vs late blight, and why (similar spots).
- Insert the classification report table (precision / recall / F1).

## 6. Sample Predictions
- 3 leaf photos + the top-3 output from `predict.py` / notebook step 15.
- Note the confidence for each correct prediction.

## 7. Conclusion
The model classifies tomato leaf diseases with [your %] test
accuracy using transfer learning, without training a CNN from scratch.

## 8. Limitations & Future Work
- Dataset photos have plain backgrounds; field photos (soil, hands,
  many leaves) will score lower.
- Only tomato, only 10 conditions — can extend to all 38 classes.
- Future: mobile app for farmers, real-time camera input.

## References
- Mohanty, Hughes & Salathé (2016), *Using Deep Learning for
  Image-Based Plant Disease Detection* — the PlantVillage paper.
- PlantVillage dataset (Kaggle: New Plant Diseases Dataset, augmented).
- Base project reference: github.com/RayanAIX/Plant-Disease-Detection
- TensorFlow / MobileNetV2 documentation.
