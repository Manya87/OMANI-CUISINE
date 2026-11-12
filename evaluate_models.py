
import tensorflow as tf
import numpy as np
import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

DATA_DIR = Path("dataset_split")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load class names
with open(MODEL_DIR / "class_names.json") as f:
    class_names = json.load(f)

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

# Load test images
images = []
labels = []

for class_name in class_names:

    folder = DATA_DIR / "test" / class_name

    for file in folder.iterdir():

        if file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        try:
            image = tf.keras.utils.load_img(
                file,
                target_size=IMG_SIZE
            )

            image = tf.keras.utils.img_to_array(image)

            images.append(image)
            labels.append(class_to_index[class_name])

        except Exception as e:
            print("Skipped:", file, e)

X_test = np.asarray(images, dtype=np.float32)
y_test = np.asarray(labels, dtype=np.int32)

print("Test images:", len(X_test))
print("Test classes:", len(np.unique(y_test)))

test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
).batch(BATCH_SIZE)

# Models to compare
model_paths = {
    "Baseline": MODEL_DIR / "best_model.keras",
    "Improved": MODEL_DIR / "omani_cuisine_model_improved.keras"
}

comparison = []

for model_name, model_path in model_paths.items():

    print("\nEvaluating:", model_name)

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_ds, verbose=1)

    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(
        y_test,
        predicted_labels
    )

    print("Test accuracy:", accuracy)

    report = classification_report(
        y_test,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULT_DIR / f"{model_name.lower()}_classification_report.csv"
    )

    cm = confusion_matrix(
        y_test,
        predicted_labels,
        labels=list(range(len(class_names)))
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(
        RESULT_DIR / f"{model_name.lower()}_confusion_matrix.csv"
    )

    comparison.append({
        "Model": model_name,
        "Test Accuracy": accuracy,
        "Precision Macro": report["macro avg"]["precision"],
        "Recall Macro": report["macro avg"]["recall"],
        "F1 Macro": report["macro avg"]["f1-score"]
    })

# Save comparison
comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison_df.to_string(index=False))

print("\nReports saved in results folder.")
