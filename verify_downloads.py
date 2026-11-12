
from pathlib import Path

BASE = Path("collected_images")

total = 0

for category in ["vegetarian", "non_vegetarian"]:
    print(f"\n{'='*50}")
    print(category.upper())
    print(f"{'='*50}")

    category_total = 0

    for folder in sorted((BASE / category).iterdir()):
        if folder.is_dir():
            count = sum(
                1 for f in folder.iterdir()
                if f.is_file()
            )

            print(f"{folder.name}: {count} images")
            category_total += count

    print(f"\n{category.upper()} TOTAL: {category_total}")
    total += category_total

print(f"\n{'='*50}")
print(f"GRAND TOTAL: {total} images")
print(f"{'='*50}")
