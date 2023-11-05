import logging

import numpy as np


# these methods do not depend on the network
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def tan_h(Z):
    return (np.exp(Z)-np.exp(-1))/(np.exp(Z)+np.exp(-1))


def softmax(Z):
    exp = np.exp(Z-np.max(Z))
    return exp / exp.sum(axis=0, keepdims=True)


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tan_h_backward(dA, Z):
    tanh = tan_h(Z)
    return 1-np.square(tanh)


def softmax_backward(Z):
    soft_max = softmax(Z)
    return - np.outer(soft_max, soft_max) + np.diag(soft_max.flatten())


def init_layer_bias_uniform(out_dim: int):
    return np.random.uniform(low=-1, high=1, size=(out_dim, 1)) * 0.1


def init_layer_weights_uniform(out_dim: int, in_dim: int):
    return np.random.uniform(low=-1, high=1, size=(out_dim, in_dim)) * 0.1


def init_layer_weights_xavier(out_dim: int, in_dim: int):  # for sigmoid and tanh
    return np.random.randn(out_dim, in_dim) * np.sqrt(6 / in_dim + out_dim)  # or 1/in_dim


def init_layer_weights_he(out_dim: int, in_dim: int):
    return np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    elif activation == "tanh":
        activation_func = tan_h
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr


# TODO: replace with custom cost function
def get_cost_value(Y_hat, Y, type, eps=0.001):
    if type == "regression":
        return mean_squared_log_error(Y_hat=Y_hat, Y=Y)
    if type == "binary":
        return np.squeeze(binary_cross_entropy(Y_hat=Y_hat, Y=Y)) # TODO: check if this squeeze is ok here
    if type == "categorical":
        return np.squeeze(categorical_cross_entropy(Y_hat=Y_hat, Y=Y))
    raise Exception(f"Type {type} not implemented")


# use for regression
# https://medium.com/analytics-vidhya/root-mean-square-log-error-rmse-vs-rmlse-935c6cc1802a
# msle vs mse
# - mse explodes for big values
# - mse is symmetric and linear - failes to capture special relation between Y and Y_hat
# - mlse calculates relative error - scale of number doesn't affect error value
# - mlse gives more penalty for underestimated values
def mean_squared_error(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = (np.square(Y_hat - Y)).mean()
    return cost


def mean_squared_log_error(Y_hat, Y):
    return (np.square(np.log(Y_hat + 1) - np.log(Y + 1))).mean()


# for binary classification
def binary_cross_entropy(Y_hat, Y):
    m = Y.shape[1]
    return (1 / m) * (-np.dot(Y, np.log(Y_hat).T) - np.dot(1 - Y, np.log(1 - Y_hat).T))


# for multiclass classification
# a_max is maximum probability for given class - 1
# eps is a minimum probability taken into account for a classification
def categorical_cross_entropy(Y_hat, Y, eps=0.5):
    n = Y_hat.shape[0]
    return - np.sum(Y * np.log(np.clip(Y_hat, eps, 1.))) / n


# TODO: test if wee really need that one
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


# TODO: replace with custom accuracy function
def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]

    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    elif activation == "tanh":
        backward_activation_funcactivation_func = tan_h_backward
    elif activation == "softmax":
        backward_activation_funcactivation_func = softmax_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


class NeuralNetwork:
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_hidden_layers: int,
                 architecture: dict,
                 initialization: str,
                 learning_rate: float
                 ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.architecture = architecture
        self.initialization = initialization
        self.parameters = {}
        self.memory = {}
        self.gradients = {}
        self.learning_rate = learning_rate

    def init_layer(self, out_dim: int, in_dim: int):
        if self.initialization == 'xavier':
            return init_layer_weights_xavier(out_dim=out_dim, in_dim=in_dim)
        elif self.initialization == 'he':
            return init_layer_weights_he(out_dim=out_dim, in_dim=in_dim)
        elif self.initialization == 'uniform':
            return init_layer_weights_uniform(out_dim=out_dim, in_dim=in_dim)
        logging.info(f"method {self.initialization} not implemented, using uniform instead")
        return init_layer_weights_uniform(out_dim=out_dim, in_dim=in_dim)

    # TODO: check what is this seed for and if it is necessary
    def init_layers(self, seed=99):
        np.random.seed(seed)

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1

            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            self.parameters['W' + str(layer_idx)] = self.init_layer(
                out_dim=layer_output_size, in_dim=layer_input_size)
            self.parameters['b' + str(layer_idx)] = self.init_layer(
                out_dim=layer_output_size, in_dim=1)

        return self.parameters

    def full_forward_propagation(self, X):
        A_curr = X

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = self.parameters["W" + str(layer_idx)]
            b_curr = self.parameters["b" + str(layer_idx)]
            A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            self.memory["A" + str(idx)] = A_prev
            self.memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, self.memory

    def full_backward_propagation(self, Y_hat, Y, eps=0.000000000001):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        dA_prev = - (np.divide(Y, Y_hat + eps) - np.divide(1 - Y, 1 - Y_hat + eps))

        for layer_idx_prev, layer in reversed(list(enumerate(self.architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = self.memory["A" + str(layer_idx_prev)]
            Z_curr = self.memory["Z" + str(layer_idx_curr)]

            W_curr = self.parameters["W" + str(layer_idx_curr)]
            b_curr = self.parameters["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            self.gradients["dW" + str(layer_idx_curr)] = dW_curr
            self.gradients["db" + str(layer_idx_curr)] = db_curr

        return self.gradients

    def update(self):
        for layer_idx, layer in enumerate(self.architecture, 1):
            self.parameters["W" + str(layer_idx)] -= self.learning_rate * self.gradients["dW" + str(layer_idx)]
            self.parameters["b" + str(layer_idx)] -= self.learning_rate * self.gradients["db" + str(layer_idx)]

        return self.parameters
