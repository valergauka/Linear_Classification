from data_parser import parse_stackoverflow_data
from data_splitter import split_data
from evaluation import evaluate_linear_regression

data = parse_stackoverflow_data()
X_train, y_train, X_val, y_val = split_data(data)
validation_mse, baseline_mse = evaluate_linear_regression(X_train, y_train, X_val, y_val)
print(f"Validation MSE: {validation_mse}")
print(f"Baseline MSE: {baseline_mse}")
