# CNN validation error analysis

Model: `cnn\best_model.keras`; validation split: `data\raw\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid`.

## Verified setup

- Input/output shapes: `(None, 256, 256, 3)` / `(None, 38)`.
- Preprocessing: RGB, resize to 256×256, float32 divided by 255.
- Training and validation directories: `data\raw\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train` and `data\raw\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid`.
- Validation images: 17,572; classes: 38; labels are alphabetical, as in `cnn/data_loader.py`.

## Metrics

- Accuracy: 0.932165 (93.22%)
- Macro precision/recall/F1: 0.933333 / 0.931836 / 0.931868
- Weighted precision/recall/F1: 0.933200 / 0.932165 / 0.931973

## Five weakest classes by recall

1. `Peach___Bacterial_spot`: 77.3420% (support 459)
2. `Tomato___Septoria_leaf_spot`: 78.2110% (support 436)
3. `Tomato___Early_blight`: 78.3333% (support 480)
4. `Tomato___Target_Spot`: 81.6193% (support 457)
5. `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`: 82.9268% (support 410)

## Frequent directed confusions

1. `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` → `Corn_(maize)___Northern_Leaf_Blight`: 67
2. `Tomato___Early_blight` → `Tomato___Late_blight`: 37
3. `Grape___Black_rot` → `Grape___Esca_(Black_Measles)`: 31
4. `Tomato___Spider_mites Two-spotted_spider_mite` → `Tomato___Target_Spot`: 29
5. `Tomato___Target_Spot` → `Tomato___Spider_mites Two-spotted_spider_mite`: 28
6. `Tomato___Late_blight` → `Potato___Late_blight`: 23
7. `Peach___Bacterial_spot` → `Tomato___Late_blight`: 22
8. `Tomato___Late_blight` → `Tomato___Early_blight`: 22
9. `Peach___Bacterial_spot` → `Tomato___Bacterial_spot`: 21
10. `Potato___Late_blight` → `Tomato___Late_blight`: 21

## Error observations

The confusion pairs above are verified counts on this validation split. A plausible hypothesis is overlapping visible lesion or texture patterns across some diseases, but these metrics alone cannot establish a biological cause.

## Index-to-label mapping

- 0: `Apple___Apple_scab`
- 1: `Apple___Black_rot`
- 2: `Apple___Cedar_apple_rust`
- 3: `Apple___healthy`
- 4: `Blueberry___healthy`
- 5: `Cherry_(including_sour)___Powdery_mildew`
- 6: `Cherry_(including_sour)___healthy`
- 7: `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
- 8: `Corn_(maize)___Common_rust_`
- 9: `Corn_(maize)___Northern_Leaf_Blight`
- 10: `Corn_(maize)___healthy`
- 11: `Grape___Black_rot`
- 12: `Grape___Esca_(Black_Measles)`
- 13: `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
- 14: `Grape___healthy`
- 15: `Orange___Haunglongbing_(Citrus_greening)`
- 16: `Peach___Bacterial_spot`
- 17: `Peach___healthy`
- 18: `Pepper,_bell___Bacterial_spot`
- 19: `Pepper,_bell___healthy`
- 20: `Potato___Early_blight`
- 21: `Potato___Late_blight`
- 22: `Potato___healthy`
- 23: `Raspberry___healthy`
- 24: `Soybean___healthy`
- 25: `Squash___Powdery_mildew`
- 26: `Strawberry___Leaf_scorch`
- 27: `Strawberry___healthy`
- 28: `Tomato___Bacterial_spot`
- 29: `Tomato___Early_blight`
- 30: `Tomato___Late_blight`
- 31: `Tomato___Leaf_Mold`
- 32: `Tomato___Septoria_leaf_spot`
- 33: `Tomato___Spider_mites Two-spotted_spider_mite`
- 34: `Tomato___Target_Spot`
- 35: `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
- 36: `Tomato___Tomato_mosaic_virus`
- 37: `Tomato___healthy`

## Comparison with existing report

The existing `cnn\classification_report.txt` reports 17,572 validation images, 93.22% accuracy, macro F1 0.9319, and weighted F1 0.9320. This run independently evaluates the saved model; compare the results above for consistency.

CSV confidence is the softmax score for the predicted class. The gallery shows up to 24 misclassified examples.
