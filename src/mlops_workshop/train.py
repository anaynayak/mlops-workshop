"""Model training for NYC Taxi trip duration prediction."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

from mlops_workshop.features import get_feature_columns, get_target_column


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    """Train a Random Forest model for trip duration prediction.

    Args:
        df: DataFrame with features and target
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees

    Returns:
        Tuple of (trained model, X_test, y_test)
    """
    feature_cols = get_feature_columns()
    target_col = get_target_column()

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model: RandomForestRegressor, path: str | Path) -> None:
    """Save trained model to disk.

    Args:
        model: Trained model
        path: Path to save model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> RandomForestRegressor:
    """Load trained model from disk.

    Args:
        path: Path to saved model

    Returns:
        Loaded model
    """
    return joblib.load(path)
