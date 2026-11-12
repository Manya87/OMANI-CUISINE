
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_DIR = Path("results")

# Load baseline results
cm = pd.read_csv(
    RESULT_DIR / "baseline_confusion_matrix.csv",
    index_col=0
)

report = pd.read_csv(
    RESULT_DIR / "baseline_classification_report.csv",
    index_col=0
)

# Clean report index
report.index = report.index.astype(str)

# Remove summary rows to retain dish classes
class_report = report[
    ~report.index.isin([
        "accuracy",
        "macro avg",
        "weighted avg"
    ])
].copy()

# Keep only actual class rows
class_report = class_report[
    ["precision", "recall", "f1-score", "support"]
].dropna()

# =====================================
# 1. BEST AND WEAKEST DISHES
# =====================================

best = class_report.sort_values(
    "f1-score", ascending=False
).head(15)

worst = class_report.sort_values(
    "f1-score", ascending=True
).head(15)

print("\nTOP 15 BEST RECOGNISED DISHES")
print("=" * 75)
print(best.round(3).to_string())

print("\nTOP 15 WEAKEST RECOGNISED DISHES")
print("=" * 75)
print(worst.round(3).to_string())

best.to_csv(RESULT_DIR / "baseline_best_classes.csv")
worst.to_csv(RESULT_DIR / "baseline_weakest_classes.csv")

# =====================================
# 2. MOST COMMON MISCLASSIFICATIONS
# =====================================

values = cm.to_numpy()

errors = []

for i in range(len(cm.index)):
    for j in range(len(cm.columns)):
        if i != j and values[i, j] > 0:
            errors.append({
                "Actual Dish": cm.index[i],
                "Predicted Dish": cm.columns[j],
                "Incorrect Predictions": int(values[i, j])
            })

error_df = pd.DataFrame(errors)

if not error_df.empty:
    error_df = error_df.sort_values(
        "Incorrect Predictions",
        ascending=False
    )

    print("\nTOP 20 MISCLASSIFICATIONS")
    print("=" * 75)
    print(error_df.head(20).to_string(index=False))

    error_df.to_csv(
        RESULT_DIR / "baseline_misclassifications.csv",
        index=False
    )

# =====================================
# 3. CONFUSION MATRIX HEATMAP
# =====================================

fig, ax = plt.subplots(figsize=(22, 19))

im = ax.imshow(values, interpolation="nearest", cmap="Blues")

plt.colorbar(im, ax=ax, label="Number of images")

ax.set_xticks(np.arange(len(cm.columns)))
ax.set_yticks(np.arange(len(cm.index)))

ax.set_xticklabels(cm.columns, rotation=90, fontsize=7)
ax.set_yticklabels(cm.index, fontsize=7)

ax.set_xlabel("Predicted Dish")
ax.set_ylabel("Actual Dish")
ax.set_title("Baseline Model - Omani Cuisine Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "baseline_confusion_matrix_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n=====================================")
print("BASELINE ANALYSIS COMPLETED")
print("=====================================")
print("Saved:")
print("- baseline_best_classes.csv")
print("- baseline_weakest_classes.csv")
print("- baseline_misclassifications.csv")
print("- baseline_confusion_matrix_heatmap.png")
