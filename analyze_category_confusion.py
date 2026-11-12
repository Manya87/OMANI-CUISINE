
import pandas as pd
import numpy as np

cm = pd.read_csv(
    "results/baseline_confusion_matrix.csv",
    index_col=0
)

matrix = cm.to_numpy()

labels = cm.index.astype(str)

actual_veg = np.array([
    x.startswith("vegetarian/") for x in labels
])

actual_nonveg = ~actual_veg

# Rows = actual, columns = predicted

vv = matrix[np.ix_(actual_veg, actual_veg)].sum()
vn = matrix[np.ix_(actual_veg, actual_nonveg)].sum()

nn = matrix[np.ix_(actual_nonveg, actual_nonveg)].sum()
nv = matrix[np.ix_(actual_nonveg, actual_veg)].sum()

print("\nCATEGORY-LEVEL CONFUSION")
print("=" * 50)

print("Actual vegetarian predicted vegetarian:", vv)
print("Actual vegetarian predicted non-vegetarian:", vn)

print("Actual non-vegetarian predicted non-vegetarian:", nn)
print("Actual non-vegetarian predicted vegetarian:", nv)

print("\nCATEGORY SUMMARY")
print("=" * 50)

print("Vegetarian correct:", vv)
print("Vegetarian incorrect:", vn)
print("Non-vegetarian correct:", nn)
print("Non-vegetarian incorrect:", nv)

print("\nTotal correct:", vv + nn)
print("Total incorrect:", vn + nv)

print("\nCategory classification accuracy:")
print("Vegetarian:", round(vv / (vv + vn), 4))
print("Non-vegetarian:", round(nn / (nn + nv), 4))

print("\nAnalysis completed.")
