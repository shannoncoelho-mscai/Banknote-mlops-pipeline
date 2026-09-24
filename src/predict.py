import joblib
import pandas as pd

from src.validate_data import FEATURE_COLUMNS


MODEL_PATH = "artifacts/model.joblib"


def predict(sample):
    """
    Make a prediction for one banknote sample.

    The sample must contain all required feature columns.
    """

    # Convert input dictionary into a DataFrame
    df = pd.DataFrame([sample])

    # Check for missing required features
    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep features in the expected order
    df = df[FEATURE_COLUMNS]

    # Load the trained pipeline
    model = joblib.load(MODEL_PATH)

    # Make prediction
    prediction = model.predict(df)

    return prediction[1]


if __name__ == "__main__":
    sample = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699,
    }

    result = predict(sample)

    print(f"Prediction: {result}")
    print(f"Prediction type: {type(result).__name__}")