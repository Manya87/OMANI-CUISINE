
from pathlib import Path

vegetarian = [
    "omani_halva", "khubz_rakhal", "omani_bread", "luqaimat",
    "shorbat_adas", "marqat_al_dal", "dengu_mqashad",
    "asida_bi_dibs", "khabissa_bil_findal", "sakhana",
    "albadhinajan_mae_tawarikh", "mushaltat", "khabees",
    "dates", "omani_kahwa", "halwa_with_saffron",
    "halwa_with_dates", "halwa_with_nuts", "rice_with_saffron",
    "rice_with_onion", "garlic_jareesh", "albessisa",
    "albessisa_coconut", "alrazya", "ajar_eggplant"
]

non_vegetarian = [
    "omani_shuwa", "omani_majboos", "chicken_majboos",
    "lamb_majboos", "fish_majboos", "omani_harees",
    "chicken_harees", "omani_qabooli", "camel_qabooli",
    "omani_mashuai", "omani_mishkak", "omani_kebab",
    "muqalab", "maraq_maajeen", "maraq_mashakik",
    "mahmas_laham", "marqat_al_dijaj", "marqat_dijaj_narjeel",
    "salqat_al_dijaj", "marqat_pablo", "salqat_al_samak",
    "samak_mtafai", "khambasha", "omani_fried_fish",
    "omani_sayadiya"
]

base = Path("dataset")

for category, dishes in [
    ("vegetarian", vegetarian),
    ("non_vegetarian", non_vegetarian)
]:
    for dish in dishes:
        (base / category / dish).mkdir(parents=True, exist_ok=True)

print("SUCCESS: All 50 dish folders created!")
print("Vegetarian classes:", len(vegetarian))
print("Non-vegetarian classes:", len(non_vegetarian))
print("Total classes:", len(vegetarian) + len(non_vegetarian))
