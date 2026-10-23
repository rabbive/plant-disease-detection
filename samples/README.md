# Sample leaves for the demo

20 tomato leaf photos, 2 per class, picked at random (seed 7) from the
**test split** of the reference run, so the committed model never saw
them during training. Expected predictions for each are in
`../outputs/predictions.json`: the raw top-3, plus what the app shows
after its safety rules (`app_shows`, `app_status`). 18/20 are correct;
the two misses are useful to show, see `docs/DEMO_SCRIPT.md`.

Source: PlantVillage dataset (Hughes & Salathé; Mohanty, Hughes &
Salathé 2016), original images from
https://github.com/spMohanty/PlantVillage-Dataset (`raw/color`).
