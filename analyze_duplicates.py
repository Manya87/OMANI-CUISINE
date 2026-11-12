
import csv
from collections import defaultdict
from pathlib import Path

report = Path("results/validation_report.csv")

groups = defaultdict(list)

with open(report, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        groups[row["hash"]].append(row)

same_class = []
cross_class = []

for image_hash, images in groups.items():

    if len(images) <= 1:
        continue

    dishes = set(
        (img["category"], img["dish"])
        for img in images
    )

    record = {
        "hash": image_hash,
        "count": len(images),
        "locations": [
            f'{img["category"]}/{img["dish"]}/{img["filename"]}'
            for img in images
        ]
    }

    if len(dishes) == 1:
        same_class.append(record)
    else:
        cross_class.append(record)

with open("results/duplicate_analysis.txt",
          "w", encoding="utf-8") as f:

    f.write("SAME CLASS DUPLICATES\n\n")

    for item in same_class:
        f.write(f'Copies: {item["count"]}\n')
        f.write("\n".join(item["locations"]))
        f.write("\n\n")

    f.write("\nCROSS CLASS DUPLICATES\n\n")

    for item in cross_class:
        f.write(f'Copies: {item["count"]}\n')
        f.write("\n".join(item["locations"]))
        f.write("\n\n")

print("DUPLICATE ANALYSIS COMPLETE")
print("Same-class duplicate groups:", len(same_class))
print("Cross-class duplicate groups:", len(cross_class))
print("Report: results/duplicate_analysis.txt")
