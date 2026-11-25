import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("premature-problem-solving"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv", dtype={"id": str})
    validation = pd.read_csv(args.data_dir / "validation.csv", dtype={"id": str})
    test = pd.read_csv(args.data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"id": str})

    # Add the start of the response separately so the model can focus on it.
    for data in (train, validation, test):
        data["response_start"] = data.response.str.split().str[:80].str.join(" ")

    columns = ["context", "response", "response_start"]
    X_train, y_train = train[columns], train["label"]
    X_val, y_val = validation[columns], validation["label"]
    features = ColumnTransformer([
        (col, TfidfVectorizer(ngram_range=(1, 2), max_features=4000,
                              sublinear_tf=True), col)
        for col in columns
    ])
    pipeline = make_pipeline(features, LinearSVC(class_weight="balanced", random_state=SEED))
    pipeline.fit(X_train, y_train)

    val_predictions = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    macro_f1 = f1_score(y_val, val_predictions, average="macro", zero_division=0)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")

    if list(sample.columns) != ["id", "label"] or sample.id.duplicated().any() or set(sample.id) != set(test.id):
        raise ValueError("Sample submission does not match the test IDs")
    test_predictions = pd.DataFrame({"id": test["id"], "label": pipeline.predict(test[columns])})
    submission = sample[["id"]].merge(test_predictions, on="id", how="left", validate="one_to_one")
    if len(submission) != len(test) or submission.label.isna().any() or not set(submission.label).issubset({0, 1}):
        raise ValueError("Invalid submission")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_dir / "run3_svm.csv", index=False)


if __name__ == "__main__":
    main()
