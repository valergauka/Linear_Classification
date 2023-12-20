import numpy as np

def split_data(df):
    num_rows = len(df)
    train_size = int(0.8 * num_rows)

    # Training set
    X_train = np.c_[np.ones(train_size), df.iloc[train_size:, 1:].values.astype(float)]
    y_train = df.iloc[train_size:, 0].values.astype(float)

    # Validation set
    X_val = np.c_[np.ones(num_rows - train_size), df.iloc[:train_size, 1:].values.astype(float)]
    y_val = df.iloc[:train_size, 0].values.astype(float)

    return X_train, y_train, X_val, y_val

