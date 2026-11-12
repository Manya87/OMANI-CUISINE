from pathlib import Path

base = Path("dataset_split")

for split in ["train", "validation", "test"]:
    split_path = base / split

    classes = []

    for category in split_path.iterdir():
        if category.is_dir():
            for dish in category.iterdir():
                if dish.is_dir():
                    classes.append(dish.name)

    print(f"\n{split.upper()}")
    print("Total classes:", len(classes))
    print("Dish folders:", classes)