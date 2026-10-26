---
title: Tomato Leaf Doctor
emoji: 🍅
colorFrom: green
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Tomato leaf photo -> disease, heatmap and advice
---

# Tomato Leaf Doctor

Upload a photo of a single tomato leaf. A MobileNetV2 model (transfer
learning, trained on 10 PlantVillage tomato classes) returns the top-3
predictions, a Grad-CAM heatmap of where it looked, and an info card
with the cause, symptoms and prevention for the most likely disease.
The model runs in your browser (ONNX Runtime Web), so photos never
leave your device.

- Test accuracy: 91.33% (2,734 held-out images); a CNN trained from
  scratch on the same data gets 85.37%.
- Safety rules (tuned on the validation split): "Healthy" only at 90%+
  confidence (sick leaves called healthy on test: 56 -> 7), "not sure"
  below 60%, and photos that don't look like a tomato leaf are
  rejected instead of getting a made-up diagnosis.
- Works best on one leaf against a plain background, like the training
  photos. Field photos will score lower.
- The info cards are general guidance, not a diagnosis.

Source code, training notebook and full results:
https://github.com/rabbive/plant-disease-detection
