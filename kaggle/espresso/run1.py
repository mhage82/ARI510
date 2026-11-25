import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROAST_DATES = {
    1: "2026-08-24", 2: "2026-08-09", 3: "2026-09-14", 4: "2026-09-14",
    5: "2026-09-22", 6: "2026-09-13", 7: "2026-09-22", 8: "2026-09-17",
    9: "2026-09-29", 10: "2026-09-29",
}
CATEGORICAL = ["bean_id", "water_spritz", "wdt_used", "tamping_method",
               "puck_screen_used", "tank_drained"]


def run(classifier, filename):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("espresso-extraction-prediction"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv", dtype={"id": str})
    test = pd.read_csv(args.data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"id": str})
    if list(sample.columns) != ["id", "extraction"] or sample.id.duplicated().any() or set(sample.id) != set(test.id):
        raise ValueError("Sample submission does not match the test IDs")

    # Reconstruct the date from the published roast date and days since roast.
    roast_dates = pd.to_datetime(train.bean_id.map(ROAST_DATES))
    if roast_dates.isna().any():
        raise ValueError("A bean's roast date is missing")
    groups = roast_dates + pd.to_timedelta(train.days_since_roast, unit="D")
    numeric = [col for col in train.columns if col not in CATEGORICAL + ["id", "extraction"]]
    columns = numeric + CATEGORICAL
    X = train[columns]
    y = train["extraction"]

    # Fit all preprocessing on each training fold, not on its validation fold.
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                      OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    model = make_pipeline(preprocessing, classifier)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = np.zeros(len(train), dtype=int)
    fold_scores = []
    for fold, (training_rows, validation_rows) in enumerate(cv.split(X, y, groups), 1):
        model.fit(X.iloc[training_rows], y.iloc[training_rows])
        actual = y.iloc[validation_rows]
        predicted = model.predict(X.iloc[validation_rows])
        predictions[validation_rows] = predicted
        score = cohen_kappa_score(actual, predicted, labels=[0, 1, 2], weights="quadratic")
        fold_scores.append(score)
        print(f"Fold {fold} QWK: {score:.4f}")

    qwk = cohen_kappa_score(y, predictions, labels=[0, 1, 2], weights="quadratic")
    accuracy = accuracy_score(y, predictions)
    print(f"Pooled validation QWK: {qwk:.4f}")
    print(f"Mean fold QWK: {np.mean(fold_scores):.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

    # Refit on all training data, then predict each test shot.
    model.fit(X, y)
    test_predictions = pd.DataFrame({"id": test.id, "extraction": model.predict(test[columns])})
    submission = sample[["id"]].merge(test_predictions, on="id", how="left", validate="one_to_one")
    if len(submission) != len(test) or submission.extraction.isna().any() or not set(submission.extraction).issubset({0, 1, 2}):
        raise ValueError("Invalid submission")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_dir / filename, index=False)


if __name__ == "__main__":
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                   class_weight="balanced", random_state=SEED, n_jobs=1)
    run(model, "run1_random_forest.csv")
