from models import init_layers, full_forward_propagation, get_cost_value, get_accuracy_value, full_backward_propagation, \
    update


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)

        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if (i % 50 == 0):
            if (verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if (callback is not None):
                callback(i, params_values)

    return params_values, cost_history, accuracy_history


def train_batch(X, Y, nn_architecture, epochs, learning_rate, batch_size=64, verbose=False, callback=None):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    # Beginning of additional code snippet
    examples_size = X.shape[1]
    batch_number = examples_size // batch_size
    # Ending of additional code snippet

    for i in range(epochs):
        # Beginning of additional code snippet
        batch_idx = epochs % batch_number
        X_batch = X[:, batch_idx * batch_size: (batch_idx + 1) * batch_size]
        Y_batch = Y[:, batch_idx * batch_size: (batch_idx + 1) * batch_size]
        # Ending of additional code snippet

        Y_hat, cashe = full_forward_propagation(X_batch, params_values, nn_architecture)

        cost = get_cost_value(Y_hat, Y_batch)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y_batch)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(Y_hat, Y_batch, cashe, params_values,
                                                                    nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if (i % 50 == 0):
            if (verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if (callback is not None):
                callback(i, params_values)

    return params_values, cost_history, accuracy_history