import numpy as np
from scipy.stats import norm
from random import random


def initialise_params(input_size: int, hidden_size: int, output_size: int):
    layer1_weights = np.random.randn(input_size, hidden_size) * 0.01
    layer1_biases = np.zeros((1, hidden_size))

    layer2_weights = np.random.randn(input_size, hidden_size) * 0.01
    layer2_biases = np.zeros((1, hidden_size))
    params_dict = {
        "layer1_weights": layer1_weights,
        "layer1_biases": layer1_biases,
        "layer2_weights": layer2_weights,
        "layer2_biases": layer2_biases,
    }
    return params_dict


def forward_pass(params_dict: dict, X: np.ndarray):
    layer1_weights = params_dict["layer1_weights"]
    layer2_weights = params_dict["layer2_weights"]
    layer1_biases = params_dict["layer1_biases"]
    layer2_biases = params_dict["layer2_biases"]
    Z1 = np.dot(X, layer1_weights) + layer1_biases
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(A1, layer2_weights) + layer2_biases
    exp_Z2 = np.exp(Z2)
    exp_Z2_sum = np.sum(exp_Z2, axis=1, keepdims=True)
    A2 = exp_Z2 / exp_Z2_sum
    cache_dict = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache_dict


def compute_loss(A2, true_label):
    return
