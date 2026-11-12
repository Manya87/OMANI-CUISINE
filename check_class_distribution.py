
import pandas as pd

df = pd.read_csv("dataset_split/split_counts.csv")

print("\nLOWEST 15 CLASSES BY TRAINING IMAGES")
print("=" * 65)

print(
    df.sort_values("Train")[
        ["Category", "Dish", "Total", "Train", "Validation", "Test"]
    ].head(15).to_string(index=False)
)

print("\nHIGHEST 15 CLASSES BY TRAINING IMAGES")
print("=" * 65)

print(
    df.sort_values("Train", ascending=False)[
        ["Category", "Dish", "Total", "Train", "Validation", "Test"]
    ].head(15).to_string(index=False)
)

print("\nCATEGORY SUMMARY")
print("=" * 65)

print(
    df.groupby("Category")[["Total", "Train", "Validation", "Test"]]
    .sum()
)
