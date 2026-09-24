import json
import os

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from validate_data import (
    DATA_URL,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ALL_COLUMNS,
    validate_data,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
IMPROVEMENT_MARGIN = 0.05

MODEL_PATH = "artifacts/model.joblib"
METRICS_PATH = "artifacts/metrics.json"


def load_dataset():
    """Download and load the Banknote Authentication dataset."""

    df = pd.read_csv(
        DATA_URL,
        header=None,
        names=ALL_COLUMNS,
    )

    return df


def train_and_evaluate():
    """Train baseline and candidate model and apply quality gate."""

    os.makedirs("artifacts", exist_ok=True)

    # 1. Load and validate dataset

    df = load_dataset()
    validate_data(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 2. Reproducible train/validation split

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")

    # 3. Train DummyClassifier baseline

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_valid)

    baseline_f1 = f1_score(
        y_valid,
        baseline_predictions,
    )

    # 4. Intentionally weak candidate model
    # FAILURE A: candidate is deliberately the same
    # as the baseline so the quality gate fails.

    model = DummyClassifier(
        strategy="most_frequent"
    )

    model.fit(X_train, y_train)

    model_predictions = model.predict(X_valid)

    model_f1 = f1_score(
        y_valid,
        model_predictions,
    )

    # 5. Quality gate

    required_score = baseline_f1 + IMPROVEMENT_MARGIN

    gate_passed = model_f1 >= required_score

    print()
    print("========== MODEL EVALUATION ==========")
    print(f"Baseline F1:       {baseline_f1:.4f}")
    print(f"Model F1:          {model_f1:.4f}")
    print(f"Required F1:       {required_score:.4f}")
    print(f"Improvement margin: {IMPROVEMENT_MARGIN:.4f}")
    print(
        f"Quality gate:      "
        f"{'PASS' if gate_passed else 'FAIL'}"
    )
    print("=======================================")

    # 6. Save metrics report

    metrics = {
        "dataset": "Banknote Authentication",
        "metric": "F1-score",
        "baseline_score": round(float(baseline_f1), 4),
        "model_score": round(float(model_f1), 4),
        "improvement_margin": IMPROVEMENT_MARGIN,
        "required_score": round(float(required_score), 4),
        "gate_passed": gate_passed,
        "random_state": RANDOM_STATE,
        "validation_size": TEST_SIZE,
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    # 7. Save model only when quality gate passes

    if not gate_passed:
        print(
            "ERROR: Model did not meet the required "
            "quality improvement over the baseline."
        )

        # Make sure a failed run does not leave an old model.

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        raise SystemExit(1)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_and_evaluate()