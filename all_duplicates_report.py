
import csv
from pathlib import Path
from collections import defaultdict

REPORT = Path("results/validation_report.csv")
OUTPUT = Path("results/all_117_duplicate_groups.txt")

groups = defaultdict(list)

with open(REPORT, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        groups[row["hash"]].append(row)

duplicates = [
    images
    for images in groups.values()
    if len({(x["category"], x["dish"]) for x in images}) > 1
]

with open(OUTPUT, "w", encoding="utf-8") as f:
    for number, images in enumerate(duplicates, start=1):
        labels = sorted({
            f'{item["category"]}/{item["dish"]}'
            for item in images
        })

        f.write(f"\nGROUP {number}\n")
        for label in labels:
            f.write(label + "\n")

print("COMPLETE REPORT GENERATED!")
print("Total duplicate groups:", len(duplicates))
print("Saved to:", OUTPUT)
