import numpy as np


def initialise_params(input_size: int, hidden_size: int, output_size: int):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    params_dict = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }
    return params_dict


def forward_pass(X: np.ndarray, params_dict: dict):
    W1 = params_dict["W1"]
    W2 = params_dict["W2"]
    b1 = params_dict["b1"]
    b2 = params_dict["b2"]

    Z1 = np.dot(X, W1) + b1

    # run ReLU
    A1 = np.maximum(Z1, 0)

    Z2 = np.dot(A1, W2) + b2

    # exponentiate for making sure all numbers are positive while still preserving order
    # (because of monotonicity of the exponential)
    exp_Z2 = np.exp(Z2)
    exp_Z2_sum = np.sum(exp_Z2, axis=1, keepdims=True)
    A2 = exp_Z2 / exp_Z2_sum
    cache_dict = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache_dict


def compute_loss(A2: np.ndarray, Y: np.ndarray):
    """
    This function outputs the loss at the current step for the purpose of printing. It's not actually used
    in the function because the backward_pass function uses the derivative of cross-entropy loss instead of the value.
    """
    # A2 and Y each have m rows (because we're feeding in m images all at once)
    m = Y.shape[0]
    # Y is a matrix of arrays, where each array (row) has 0 everywhere except for a 1 at the index of the class
    # cross entropy loss finds the sum of the logs of the probabilities assigned to what was the true class
    cross_entropy_loss = -1 / m * np.sum(Y * np.log(A2 + 1e-8))
    return cross_entropy_loss


def backward_pass(X: np.ndarray, Y: np.ndarray, cache_dict: dict, params_dict: dict):
    A1 = cache_dict["A1"]
    A2 = cache_dict["A2"]
    m = Y.shape[0]

    # dZ2 is an mx10 matrix which contains the error for every class for every image
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    W2 = params_dict["W2"]
    Z1 = cache_dict["Z1"]
    dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = np.mean(dZ1, axis=0, keepdims=True)
    gradients_dict = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients_dict


def update_parameters(params_dict: dict, gradients_dict: dict, learning_rate: float):
    W1 = params_dict["W1"]
    W2 = params_dict["W2"]
    b1 = params_dict["b1"]
    b2 = params_dict["b2"]

    dW1 = gradients_dict["dW1"]
    dW2 = gradients_dict["dW2"]
    db1 = gradients_dict["db1"]
    db2 = gradients_dict["db2"]

    W1_new = W1 - dW1 * learning_rate
    W2_new = W2 - dW2 * learning_rate
    b1_new = b1 - db1 * learning_rate
    b2_new = b2 - db2 * learning_rate

    new_params = {
        "W1": W1_new,
        "W2": W2_new,
        "b1": b1_new,
        "b2": b2_new,
    }

    return new_params


def train_network(
    X: np.ndarray,
    hidden_size: int,
    output_size: int,
    epochs: int,
    Y: np.ndarray,
    learning_rate: float,
    print_progress: bool = False,
):
    # X is a matrix of image data. Each row represents data for an image, and each column represents a certain
    # feature.
    input_size = X.shape[1]  # get number of
    params_dict = initialise_params(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )
    for i in range(epochs + 1):
        A2, cache_dict = forward_pass(params_dict=params_dict, X=X)
        loss = compute_loss(A2, Y)
        gradients_dict = backward_pass(X, Y, cache_dict, params_dict)
        params_dict = update_parameters(params_dict, gradients_dict, learning_rate)
        if (print_progress) and (i % 100 == 0):
            print(f"Loss after {i} epochs: {loss}")
    return params_dict


def make_prediction(X: np.ndarray, params_dict: dict, classes_array: list):
    A2, _ = forward_pass(X=X, params_dict=params_dict)
    # find the index of the maximum value in each row of A2
    most_likely_indices = np.argmax(A2, axis=1)
    # find the highest probabilities in each row
    highest_probs = np.max(A2, axis=1)
    predicted_classes = np.array(classes_array)[most_likely_indices]
    return predicted_classes, highest_probs


def calculate_accuracy(preds, true_labels):
    correct_guesses = preds == true_labels
    accuracy = np.mean(correct_guesses)
    return accuracy
