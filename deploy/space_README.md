---
title: Tomato Leaf Disease Detector
emoji: 🍅
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Upload a tomato leaf photo, get the disease + advice
---

# Tomato Leaf Disease Detector

Upload a photo of a single tomato leaf. A MobileNetV2 model (transfer
learning, trained on 10 PlantVillage tomato classes) returns the top-3
predictions and an info card with the cause, symptoms and prevention
for the most likely disease. Example leaves below the upload box are
test images the model never saw during training.

- Test accuracy: 91.33% (2,734 held-out images); a CNN trained from
  scratch on the same data gets 85.37%.
- Works best on one leaf against a plain background, like the training
  photos. Field photos will score lower.
- The info cards are general guidance, not a diagnosis.

Source code, training notebook and full results:
https://github.com/rabbive/plant-disease-detection
