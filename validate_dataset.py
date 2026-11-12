
from pathlib import Path
from PIL import Image
from collections import defaultdict
import hashlib
import csv

BASE = Path("collected_images")
REPORT = Path("results")
REPORT.mkdir(exist_ok=True)

valid_images = []
invalid_images = []
hash_groups = defaultdict(list)

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print("\nSTARTING DATASET VALIDATION\n")

for category in ["vegetarian", "non_vegetarian"]:

    category_path = BASE / category

    for dish_folder in sorted(category_path.iterdir()):

        if not dish_folder.is_dir():
            continue

        for file in dish_folder.iterdir():

            if file.suffix.lower() not in extensions:
                continue

            try:
                with Image.open(file) as img:
                    img.verify()

                with Image.open(file) as img:
                    width, height = img.size
                    img_format = img.format

                if width < 100 or height < 100:
                    raise ValueError("Image resolution too small")

                file_hash = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()

                valid_images.append({
                    "category": category,
                    "dish": dish_folder.name,
                    "filename": file.name,
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "hash": file_hash
                })

                hash_groups[file_hash].append(
                    f"{category}/{dish_folder.name}/{file.name}"
                )

            except Exception as e:
                invalid_images.append({
                    "path": str(file),
                    "error": str(e)
                })

# Save validation report
with open(REPORT / "validation_report.csv",
          "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category", "dish", "filename",
            "width", "height", "format", "hash"
        ]
    )

    writer.writeheader()
    writer.writerows(valid_images)

# Identify exact duplicates
duplicates = {
    h: paths for h, paths in hash_groups.items()
    if len(paths) > 1
}

with open(REPORT / "duplicates_report.txt",
          "w", encoding="utf-8") as f:

    for paths in duplicates.values():
        f.write("\n".join(paths) + "\n\n")

# Save corrupted image report
with open(REPORT / "invalid_images.txt",
          "w", encoding="utf-8") as f:

    for item in invalid_images:
        f.write(item["path"] + " | " + item["error"] + "\n")

print("VALIDATION COMPLETED")
print("------------------------------")
print("Valid images:", len(valid_images))
print("Invalid images:", len(invalid_images))
print("Duplicate groups:", len(duplicates))
print("Exact duplicate files:", sum(
    len(paths) - 1 for paths in duplicates.values()
))
print("------------------------------")
print("Reports saved inside results/")
