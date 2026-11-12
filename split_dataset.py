
from pathlib import Path
import random
import shutil
import csv

# Paths
SOURCE = Path("collected_images")
DESTINATION = Path("dataset_split")

# Reproducible random split
random.seed(42)

CATEGORIES = ["vegetarian", "non_vegetarian"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if not SOURCE.exists():
    raise SystemExit("ERROR: collected_images folder not found.")

if DESTINATION.exists():
    raise SystemExit(
        "ERROR: dataset_split already exists. "
        "Rename or inspect it before running this script again."
    )

DESTINATION.mkdir(parents=True)

split_counts = []

for category in CATEGORIES:
    category_path = SOURCE / category

    if not category_path.exists():
        raise SystemExit(f"ERROR: Missing folder: {category_path}")

    dish_folders = sorted(
        folder for folder in category_path.iterdir()
        if folder.is_dir()
    )

    print(f"\nProcessing {category}: {len(dish_folders)} classes")

    for dish_folder in dish_folders:
        images = sorted(
            file for file in dish_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        random.shuffle(images)
        total = len(images)

        if total < 3:
            raise SystemExit(
                f"ERROR: {dish_folder} has only {total} images. "
                "At least 3 are needed for all three splits."
            )

        train_count = max(1, int(total * 0.70))
        val_count = max(1, int(total * 0.15))
        test_count = total - train_count - val_count

        if test_count < 1:
            test_count = 1
            train_count -= 1

        splits = {
            "train": images[:train_count],
            "validation": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:]
        }

        for split_name, split_images in splits.items():
            target_folder = (
                DESTINATION / split_name / category / dish_folder.name
            )
            target_folder.mkdir(parents=True, exist_ok=True)

            for image in split_images:
                shutil.copy2(image, target_folder / image.name)

        split_counts.append([
            category,
            dish_folder.name,
            total,
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"])
        ])

        print(
            f"{dish_folder.name}: "
            f"Total={total}, "
            f"Train={len(splits['train'])}, "
            f"Validation={len(splits['validation'])}, "
            f"Test={len(splits['test'])}"
        )

# Save split report
report_path = DESTINATION / "split_counts.csv"

with report_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Category", "Dish", "Total",
        "Train", "Validation", "Test"
    ])
    writer.writerows(split_counts)

print("\n===================================")
print("DATASET SPLIT COMPLETED!")
print("===================================")
print("Total classes:", len(split_counts))
print("Training images:", sum(row[3] for row in split_counts))
print("Validation images:", sum(row[4] for row in split_counts))
print("Testing images:", sum(row[5] for row in split_counts))
print("Report:", report_path)
print("Original collected_images folder was not modified.")
