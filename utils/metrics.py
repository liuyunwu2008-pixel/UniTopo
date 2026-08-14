"""Evaluation metrics for molecular property prediction."""

import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error


def evaluate_classification(y_true, y_pred):
    """ROC-AUC for classification. Handles NaN gracefully."""
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return float('nan')


def evaluate_regression(y_true, y_pred, metric='rmse'):
    """RMSE or MAE for regression."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    raise ValueError(f"Unknown metric: {metric}")
