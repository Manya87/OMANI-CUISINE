
from icrawler.builtin import BingImageCrawler
from pathlib import Path

# Destination folder
target = Path("dataset/vegetarian/omani_halva")
target.mkdir(parents=True, exist_ok=True)

print("Starting Omani Halwa image collection...")

crawler = BingImageCrawler(
    storage={"root_dir": str(target)}
)

crawler.crawl(
    keyword="Traditional Omani Halwa food",
    max_num=10,
    min_size=(200, 200)
)

print("\nDownload process completed!")
print("Images saved in:", target)
