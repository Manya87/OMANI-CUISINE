
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import hashlib
import shutil
import csv

SOURCE = Path("collected_images")
DEST = Path("cleaned_candidate")
REPORT = Path("results")

REPORT.mkdir(exist_ok=True)

# First, identify which classes contain each image hash
hash_classes = defaultdict(set)
image_records = []

for category in ["vegetarian", "non_vegetarian"]:
    for dish in sorted((SOURCE / category).iterdir()):

        if not dish.is_dir():
            continue

        for file in dish.iterdir():

            try:
                with Image.open(file) as img:
                    img.verify()

                image_hash = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()

                hash_classes[image_hash].add(
                    f"{category}/{dish.name}"
                )

                image_records.append(
                    (category, dish.name, file, image_hash)
                )

            except Exception:
                pass

# Copy only images with one unambiguous class label
counts = Counter()
excluded = 0

for category, dish, file, image_hash in image_records:

    if len(hash_classes[image_hash]) != 1:
        excluded += 1
        continue

    target = DEST / category / dish
    target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(file, target / file.name)
    counts[f"{category}/{dish}"] += 1

# Save class counts
with open(REPORT / "cleaned_class_counts.csv",
          "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)
    writer.writerow(["Class", "Valid images"])

    for name, count in sorted(counts.items()):
        writer.writerow([name, count])

print("\nCLEANING COMPLETED")
print("-----------------------------")
print("Original images:", len(image_records))
print("Excluded ambiguous images:", excluded)
print("Copied unambiguous images:", sum(counts.values()))
print("Classes with images:", len(counts))
print("New folder:", DEST)
print("Report: results/cleaned_class_counts.csv")
print("-----------------------------")
print("Original collected_images folder remains unchanged.")
