
import csv
from pathlib import Path
from collections import defaultdict

# File locations
REPORT = Path("results/validation_report.csv")
OUTPUT = Path("results/duplicate_audit.csv")

# Store duplicate groups
groups = defaultdict(list)

# Read original validation report
with open(REPORT, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        groups[row["hash"]].append(row)

# Keep only cross-class duplicate groups
duplicates = []

for image_hash, images in groups.items():

    classes = {
        (item["category"], item["dish"])
        for item in images
    }

    if len(classes) > 1:
        duplicates.append((image_hash, images))

# Generate audit CSV
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "duplicate_group",
    "image_hash",
    "original_image_path",
    "category",
    "dish_name",
    "group_image_count",
    "cross_category_contamination",
    "review_status"
]

with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for group_number, (image_hash, images) in enumerate(
        duplicates, start=1
    ):

        categories = {
            item["category"]
            for item in images
        }

        contamination = (
            "YES" if len(categories) > 1 else "NO"
        )

        group_count = len(images)

        for item in images:

            writer.writerow({
                "duplicate_group": group_number,
                "image_hash": image_hash,
                "original_image_path": str(
                    Path("collected_images")
                    / item["category"]
                    / item["dish"]
                    / item["filename"]
                ),
                "category": item["category"],
                "dish_name": item["dish"],
                "group_image_count": group_count,
                "cross_category_contamination": contamination,
                "review_status": "Pending"
            })

print("\nDUPLICATE AUDIT COMPLETED!")
print("Total duplicate groups:", len(duplicates))
print("Total audited image records:",
      sum(len(images) for _, images in duplicates))
print("Saved to:", OUTPUT)
