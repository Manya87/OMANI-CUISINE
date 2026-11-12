
from pathlib import Path

root = Path("collected_images")
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

total = 0

for category in ["vegetarian", "non_vegetarian"]:
    folder = root / category
    print(f"\n{category.upper()}")

    if not folder.exists():
        print("Folder not found!")
        continue

    for dish in sorted(folder.iterdir()):
        if dish.is_dir():
            count = sum(
                1 for file in dish.iterdir()
                if file.is_file() and file.suffix.lower() in extensions
            )
            total += count
            print(f"{dish.name}: {count} images")

print(f"\nTOTAL IMAGES: {total}")
