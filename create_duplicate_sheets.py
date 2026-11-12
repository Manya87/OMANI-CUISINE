
import csv
import math
import textwrap
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw

BASE = Path("collected_images")
REPORT = Path("results/validation_report.csv")
OUTPUT = Path("results/duplicate_contact_sheets")

OUTPUT.mkdir(parents=True, exist_ok=True)

groups = defaultdict(list)

with open(REPORT, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        groups[row["hash"]].append(row)

duplicates = [
    (h, imgs)
    for h, imgs in groups.items()
    if len({(x["category"], x["dish"]) for x in imgs}) > 1
]

# 20 duplicate groups per contact sheet
PER_PAGE = 20
COLS = 4
CARD_W = 280
CARD_H = 245

for page_num in range(math.ceil(len(duplicates) / PER_PAGE)):

    page_groups = duplicates[
        page_num * PER_PAGE:(page_num + 1) * PER_PAGE
    ]

    rows = math.ceil(len(page_groups) / COLS)

    sheet = Image.new(
        "RGB",
        (COLS * CARD_W, rows * CARD_H),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, (image_hash, images) in enumerate(page_groups):

        x = (index % COLS) * CARD_W
        y = (index // COLS) * CARD_H

        first = images[0]

        image_path = (
            BASE / first["category"] /
            first["dish"] / first["filename"]
        )

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((CARD_W - 20, 145))

                px = x + (CARD_W - img.width) // 2
                sheet.paste(img, (px, y + 5))

        except Exception:
            draw.text((x + 10, y + 10), "Image unavailable", fill="red")

        draw.text(
            (x + 10, y + 155),
            f"GROUP {page_num * PER_PAGE + index + 1}",
            fill="blue"
        )

        labels = sorted({
            f'{item["category"]}/{item["dish"]}'
            for item in images
        })

        label_text = "\n".join(
            textwrap.wrap(" | ".join(labels), width=38)
        )

        draw.multiline_text(
            (x + 10, y + 175),
            label_text,
            fill="black",
            spacing=3
        )

    filename = OUTPUT / f"page_{page_num + 1:02d}.jpg"

    sheet.save(filename, quality=90)

    print("Created:", filename)

print("\nCONTACT SHEETS COMPLETED")
print("Total duplicate groups:", len(duplicates))
print("Total pages:", math.ceil(len(duplicates) / PER_PAGE))
