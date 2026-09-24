import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/banknote_authentication.csv"
)

FEATURE_COLUMNS = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
]

TARGET_COLUMN = "class"

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_data():
    """Download and load the Banknote Authentication dataset."""

    df = pd.read_csv(
        DATA_URL,
        header=None,
        names=ALL_COLUMNS,
    )

    return df


def validate_data(df):
    """Validate the dataset structure and values."""

    print(f"Dataset shape: {df.shape}")

    # Check required columns
    missing_columns = [
        column
        for column in ALL_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Check for missing values
    if df[ALL_COLUMNS].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    # Check target values
    valid_targets = {0, 1}

    actual_targets = set(df[TARGET_COLUMN].unique())

    if not actual_targets.issubset(valid_targets):
        raise ValueError(
            f"Unexpected target values found: {actual_targets}"
        )

    # Check that features are numeric
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Feature column '{column}' is not numeric."
            )

    print("Required columns:", ALL_COLUMNS)
    print("Target values:", sorted(actual_targets))
    print("Missing values: 0")
    print("Data validation passed.")

    return True


if __name__ == "__main__":
    data = load_data()
    validate_data(data)