import numpy as np


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


def compute_loss(A2: np.ndarray, Y: np.ndarray):
    # A2 and Y each have m rows (because we're feeding in m images all at once)
    # Y is a matrix of arrays, where each array (row) has 0 everywhere except for a 1 at the index of the class
    m = Y.shape[0]  # because this tells us how many rows (images) there are
    # cross entropy loss finds the sum of the logs of the probabilities assigned to what was the true class
    cross_entropy_loss = -1 / m * np.sum(Y * np.log(A2 + 1e-8))
    return cross_entropy_loss


def backward_pass(X: np.ndarray, Y: np.ndarray, cache_dict: dict, params_dict: dict):
    A1 = cache_dict["A1"]
    A2 = cache_dict["A2"]
    # dZ2 is an mx10 matrix which contains the error for every class for every image
    m = Y.shape[0]
    dZ2 = A2 - Y
    dW2 = 1 / m * np.matmul(A1.T, dZ2)
    db2 = np.mean(dZ2, axis=0, keepdims=True)
