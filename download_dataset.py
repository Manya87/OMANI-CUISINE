
from pathlib import Path
from icrawler.builtin import BingImageCrawler
import time

# Temporary collection folder
BASE = Path("collected_images")

# 50 dish classes and their search keywords
DISHES = {
    "vegetarian": {
        "omani_halva": "Traditional Omani Halwa",
        "khubz_rakhal": "Omani Khubz Rakhal bread",
        "omani_bread": "Traditional Omani bread",
        "luqaimat": "Omani Luqaimat sweet dumplings",
        "shorbat_adas": "Omani lentil soup",
        "marqat_al_dal": "Omani lentil stew",
        "dengu_mqashad": "Omani Dengu Mqashad",
        "asida_bi_dibs": "Omani Asida date syrup",
        "khabissa_bil_findal": "Omani Khabissa",
        "sakhana": "Traditional Omani Sakhana food",
        "albadhinajan_mae_tawarikh": "Omani eggplant dates dish",
        "mushaltat": "Omani Mushaltat bread",
        "khabees": "Traditional Omani Khabees",
        "dates": "Omani dates fruit",
        "omani_kahwa": "Traditional Omani Kahwa coffee",
        "halwa_with_saffron": "Omani saffron halwa",
        "halwa_with_dates": "Omani date halwa",
        "halwa_with_nuts": "Omani nut halwa",
        "rice_with_saffron": "Omani saffron rice",
        "rice_with_onion": "Omani onion rice",
        "garlic_jareesh": "Omani garlic jareesh",
        "albessisa": "Traditional Omani Albessisa",
        "albessisa_coconut": "Omani coconut Albessisa",
        "alrazya": "Traditional Omani Al Razya food",
        "ajar_eggplant": "Omani eggplant ajar"
    },

    "non_vegetarian": {
        "omani_shuwa": "Traditional Omani Shuwa meat",
        "omani_majboos": "Traditional Omani Majboos",
        "chicken_majboos": "Omani chicken Majboos",
        "lamb_majboos": "Omani lamb Majboos",
        "fish_majboos": "Omani fish Majboos",
        "omani_harees": "Omani meat Harees",
        "chicken_harees": "Omani chicken Harees",
        "omani_qabooli": "Traditional Omani Qabooli",
        "camel_qabooli": "Omani camel meat Qabooli",
        "omani_mashuai": "Traditional Omani Mashuai fish",
        "omani_mishkak": "Omani Mishkak meat skewers",
        "omani_kebab": "Traditional Omani kebab",
        "muqalab": "Omani Muqalab meat dish",
        "maraq_maajeen": "Omani Maraq Maajeen",
        "maraq_mashakik": "Omani Maraq Mashakik",
        "mahmas_laham": "Omani Mahmas Laham meat",
        "marqat_al_dijaj": "Omani chicken stew",
        "marqat_dijaj_narjeel": "Omani chicken coconut stew",
        "salqat_al_dijaj": "Omani chicken broth",
        "marqat_pablo": "Omani Pablo fish stew",
        "salqat_al_samak": "Omani fish broth",
        "samak_mtafai": "Omani pan fried kingfish",
        "khambasha": "Traditional Omani Khambasha",
        "omani_fried_fish": "Omani fried fish",
        "omani_sayadiya": "Omani Sayadiya fish rice"
    }
}

IMAGES_PER_CLASS = 25

total_classes = 0

for category, dishes in DISHES.items():

    print("\n================================")
    print("CATEGORY:", category.upper())
    print("================================")

    for dish, keyword in dishes.items():

        target = BASE / category / dish
        target.mkdir(parents=True, exist_ok=True)

        # Do not overwrite an earlier collection
        existing = list(target.glob("*"))

        if existing:
            print(f"SKIPPING {dish}: folder already contains files")
            continue

        print(f"\nDownloading: {dish}")
        print("Search:", keyword)

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(target)}
            )

            crawler.crawl(
                keyword=keyword,
                max_num=IMAGES_PER_CLASS,
                min_size=(200, 200)
            )

            total_classes += 1

        except Exception as error:
            print("ERROR:", dish, error)

        time.sleep(3)

print("\n================================")
print("COLLECTION PROCESS FINISHED")
print("Classes processed:", total_classes)
print("Collection location:", BASE)
print("================================")
