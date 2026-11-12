
import pandas as pd

report = pd.read_csv(
    "results/baseline_classification_report.csv",
    index_col=0
)

report.index = report.index.astype(str)

report = report[
    ~report.index.isin([
        "accuracy",
        "macro avg",
        "weighted avg"
    ])
]

for category in ["vegetarian", "non_vegetarian"]:

    data = report[
        report.index.str.startswith(category + "/")
    ]

    print("\n", category.upper())
    print("=" * 50)

    print("Number of classes:", len(data))
    print("Macro Precision:", round(data["precision"].mean(), 4))
    print("Macro Recall:", round(data["recall"].mean(), 4))
    print("Macro F1:", round(data["f1-score"].mean(), 4))

    print("Total test images:", int(data["support"].sum()))

    print(
        "Correct predictions:",
        int((data["recall"] * data["support"]).sum())
    )
