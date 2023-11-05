import numpy as np

import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


from train import train_batch

if __name__ == "__main__":
    # neural network architecture
    NN_ARCHITECTURE = [
        {"input_dim": 2, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
    ]

    # number of samples in the data set
    N_SAMPLES = 1000

    # ratio between training and test sets
    TEST_PERCENTAGE = 0.1

    # number of epochs
    N_EPOCHS = 5000

    # learning rate value
    LR = 0.01

    X1, y1 = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
    X2, y2 = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=0)
    X2[:, 0] = X2[:, 0] + 2
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PERCENTAGE, random_state=42)

    # Preparation of the data set. Subsequent examples should be stored in subsequent columns.
    # We need to transpose the datasets.

    X_train = np.transpose(X_train)
    y_train = np.transpose(y_train.reshape((y_train.shape[0], 1)))
    X_test = np.transpose(X_test)
    y_test = np.transpose(y_test.reshape((y_test.shape[0], 1)))

    # params_values, cost_history, accuracy_history = train(X=X_train, Y=y_train, nn_architecture=NN_ARCHITECTURE,
    #       learning_rate=LR, epochs=N_EPOCHS)

    params_values_batch, cost_history_batch, accuracy_history_batch = train_batch(
        X_train, y_train, NN_ARCHITECTURE, N_EPOCHS, LR, 64)
