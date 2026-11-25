import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = [
    "frac_tree", "frac_vegetation", "frac_open_sky", "frac_street",
    "frac_sidewalk", "frac_building", "frac_vehicle", "frac_urban_furniture",
    "frac_shade_structure", "frac_water", "sky_blue", "ground_shadow",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("heat-refugee-three"))
    parser.add_argument("--output-dir", type=Path, default=Path("hard_results"))
    args = parser.parse_args()

    # Keep image IDs as text so they are copied exactly into the submission.
    train = pd.read_csv(args.data_dir / "train.csv", dtype={"image_id": str})
    validation = pd.read_csv(args.data_dir / "validation.csv", dtype={"image_id": str})
    test = pd.read_csv(args.data_dir / "test.csv", dtype={"image_id": str})
    X_train, y_train = train[FEATURES], train["label"]
    X_val, y_val = validation[FEATURES], validation["label"]

    models = {
        "run1_logistic": LogisticRegression(max_iter=2000, random_state=SEED),
        "run2_mlp": MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01,
                                  max_iter=1500, random_state=SEED),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"seed": SEED, "features": FEATURES, "runs": {}}

    for name, model in models.items():
        # Fit the missing-value and scaling steps on training data only.
        pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                 StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        val_predictions = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)
        macro_f1 = f1_score(y_val, val_predictions, average="macro", zero_division=0)
        results["runs"][name] = {
            "validation_accuracy": float(accuracy),
            "validation_macro_f1": float(macro_f1),
            "classification_report": classification_report(y_val, val_predictions, output_dict=True, zero_division=0),
        }
        print(f"{name}: accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}")

        predictions = pd.DataFrame({"image_id": test["image_id"],
                                    "label": pipeline.predict(test[FEATURES])})
        sample_path = args.data_dir / "sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path, dtype={"image_id": str})
            predictions = sample[["image_id"]].merge(predictions, on="image_id", validate="one_to_one")
        predictions.to_csv(args.output_dir / f"{name}.csv", index=False)

    best = max(results["runs"], key=lambda name: results["runs"][name]["validation_accuracy"])
    results["best_by_validation_accuracy"] = best
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best}")


if __name__ == "__main__":
    main()
