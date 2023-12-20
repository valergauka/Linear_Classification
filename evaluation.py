import numpy as np
from linear_regression import LinearRegression, squared_error

def evaluate_linear_regression(X_train, y_train, X_val, y_val):
    model = LinearRegression(X_train, y_train)

    y_pred = model.predict(X_val)

    baseline_prediction = np.mean(y_train)
    baseline_predictions = np.full_like(y_val, baseline_prediction)

    validation_mse = squared_error(y_pred, y_val)
    baseline_mse = squared_error(baseline_predictions, y_val)

    return validation_mse, baseline_mse
