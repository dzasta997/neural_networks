from models import NeuralNetwork, get_cost_value, get_accuracy_value


def train(X, Y, nn_architecture, epochs, learning_rate, type):
    network = NeuralNetwork(
        input_dim=1,
        output_dim=1,
        num_hidden_layers=1,
        hidden_dim=1,
        architecture=nn_architecture,
        initialization="he",
        learning_rate=0.02
    )
    network.init_layers()
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, memory= network.full_forward_propagation(X)

        cost = get_cost_value(Y_hat=Y_hat, Y=Y, type=type)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat=Y_hat, Y=Y)
        accuracy_history.append(accuracy)

        grads_values = network.full_backward_propagation(Y_hat=Y_hat, Y=Y)
        params_values = network.update()

    return network, cost_history, accuracy_history


def train_batch(X, Y, nn_architecture, epochs, learning_rate, batch_size=64, verbose=False, callback=None):
    network = NeuralNetwork(
        input_dim=1,
        output_dim=1,
        num_hidden_layers=1,
        hidden_dim=1,
        architecture=nn_architecture,
        initialization="he",
        learning_rate=0.02
    )
    network.init_layers()

    # params_values = init_layers(nn_architecture, 2)
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

        Y_hat, memory = network.full_forward_propagation(X_batch)

        cost = get_cost_value(Y_hat, Y_batch)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y_batch)
        accuracy_history.append(accuracy)

        # just for debugging purposes
        grads_values = network.full_backward_propagation(Y_hat, Y_batch)
        params_values = network.update()

    return network, cost_history, accuracy_history
