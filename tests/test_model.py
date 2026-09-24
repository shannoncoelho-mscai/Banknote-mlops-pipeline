import joblib
import pytest

from src.predict import predict
from src.validate_data import FEATURE_COLUMNS


MODEL_PATH = "artifacts/model.joblib"


def test_saved_model_loads():
    """Test that the trained model can be loaded."""

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_valid_prediction():
    """Test that a valid sample produces one prediction."""

    sample = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699,
    }

    prediction = predict(sample)

    assert prediction in [0, 1]


def test_missing_feature_is_rejected():
    """Test that missing required features produce a clear error."""

    incomplete_sample = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        # entropy is intentionally missing
    }

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        predict(incomplete_sample)