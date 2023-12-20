import numpy as np

def squared_error(y_pred, y):
    return np.mean((y_pred - y) ** 2)

class LinearRegression:
    def __init__(self, X, y):
        self.theta = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        return X @ self.theta

