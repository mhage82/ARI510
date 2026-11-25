import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from run1 import FEATURES, SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("heat-refugee-three"))
    parser.add_argument("--output-dir", type=Path, default=Path("hard_results"))
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv", dtype={"image_id": str})
    validation = pd.read_csv(args.data_dir / "validation.csv", dtype={"image_id": str})
    test = pd.read_csv(args.data_dir / "test.csv", dtype={"image_id": str})

    # Run 2: give the less common classes more weight in the random forest.
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                               max_features=0.7, class_weight="balanced",
                               random_state=SEED, n_jobs=1),
    )
    model.fit(train[FEATURES], train["label"])

    val_predictions = model.predict(validation[FEATURES])
    accuracy = accuracy_score(validation["label"], val_predictions)
    macro_f1 = f1_score(validation["label"], val_predictions, average="macro", zero_division=0)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")

    submission = pd.DataFrame({
        "image_id": test["image_id"],
        "label": model.predict(test[FEATURES]),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_dir / "run2_forest_balanced.csv", index=False)


if __name__ == "__main__":
    main()
