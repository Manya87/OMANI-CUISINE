
import csv
from collections import defaultdict, Counter
from pathlib import Path

source = Path("results/validation_report.csv")
output = Path("results/detailed_duplicate_report.txt")

groups = defaultdict(list)

with open(source, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        groups[row["hash"]].append(row)

cross_class = []
pair_counts = Counter()

for image_hash, images in groups.items():

    dishes = {
        f'{img["category"]}/{img["dish"]}'
        for img in images
    }

    if len(dishes) > 1:
        cross_class.append((image_hash, images))

        dish_list = sorted(dishes)

        for i in range(len(dish_list)):
            for j in range(i + 1, len(dish_list)):
                pair_counts[(dish_list[i], dish_list[j])] += 1

with open(output, "w", encoding="utf-8") as f:

    f.write("DETAILED CROSS-CLASS DUPLICATE REPORT\n")
    f.write("=" * 65 + "\n")

    for number, (image_hash, images) in enumerate(cross_class, 1):

        f.write(f"\nDUPLICATE GROUP {number}\n")
        f.write(f"Hash: {image_hash}\n")

        for img in images:
            f.write(
                f'  {img["category"]}/{img["dish"]}/'
                f'{img["filename"]}\n'
            )

    f.write("\n\nDISH PAIR SUMMARY\n")
    f.write("=" * 65 + "\n")

    for (dish1, dish2), count in pair_counts.most_common():
        f.write(f"{dish1} <--> {dish2}: {count} shared groups\n")

print("Detailed report generated successfully!")
print("Cross-class duplicate groups:", len(cross_class))
print("Saved to:", output)
