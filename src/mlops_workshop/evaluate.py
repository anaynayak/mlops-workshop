"""Model evaluation for NYC Taxi trip duration prediction."""

import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    """Evaluate model predictions.

    Args:
        y_true: Actual values (Series or array)
        y_pred: Predicted values (Series or array)

    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to numpy arrays for consistent handling
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE - avoid division by zero
    mask = y_true > 0
    mape = (np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100

    return {
        "rmse": rmse,
        "rmse_minutes": rmse / 60,
        "mae": mae,
        "mae_minutes": mae / 60,
        "r2": r2,
        "mape": mape,
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Print evaluation metrics in readable format.

    Args:
        metrics: Dictionary with evaluation metrics
    """
    print(f"RMSE: {metrics['rmse']:.2f} seconds ({metrics['rmse_minutes']:.2f} minutes)")
    print(f"MAE:  {metrics['mae']:.2f} seconds ({metrics['mae_minutes']:.2f} minutes)")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
