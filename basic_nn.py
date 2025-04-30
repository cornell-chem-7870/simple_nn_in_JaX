import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# The Neural Network


def neural_network(x, weights, biases):
    num_layers = len(weights)

    for i in range(num_layers - 1):
        x = weights[i] @ x + biases[i]
        x = jnp.maximum(x, 0)  # Relu activation

    # No ReLU on the last layer:
    return weights[-1] @ x + biases[-1]


# Neural network that operates on multiple inputs
batched_neural_network = jax.vmap(neural_network, in_axes=(0, None, None))


## Training the network


def calculate_mean_square_error(prediction, target):
    return jnp.sum((prediction - target) ** 2)


def update_parameters(params, gradients, learning_rate=1e-3):
    new_params = []

    for w_i, g_i in zip(params, gradients):
        new_params.append(w_i - g_i * learning_rate)

    return new_params


def train_network(
    input_data, targets, num_epochs: int = 100, learning_rate: float = 1e-4
):
    num_input_features = input_data.shape[1]
    num_output_features = targets.shape[1]

    weights, biases = initialize_three_layer_nn_params(
        num_input_features, 64, 64, num_output_features
    )

    def evaluate_network(input_data, weights, biases):
        prediction = batched_neural_network(input_data, weights, biases)
        return calculate_mean_square_error(prediction, targets)

    evaluate_gradient = jax.value_and_grad(evaluate_network, argnums=(1, 2))

    all_losses = []
    for n in range(num_epochs):
        loss, grad = evaluate_gradient(input_data, weights, biases)
        weights = update_parameters(weights, grad[0], learning_rate)
        biases = update_parameters(biases, grad[1], learning_rate)
        all_losses.append(loss)
    return weights, biases, all_losses


# Setting up the network parameters


def initialize_weight_and_bias(incoming_size, outgoing_size, key):
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    weight = jax.random.normal(subkey1, shape=(outgoing_size, incoming_size))
    weight = weight * np.sqrt(2 / incoming_size)

    bias = jax.random.normal(subkey2, shape=(outgoing_size))
    return weight, bias, key


def initialize_three_layer_nn_params(
    input_size: int, hidden_1_size: int, hidden_2_size: int, output_size: int, key=None
):
    if key is None:
        key = jax.random.key(0)

    weight_in, bias_in, key = initialize_weight_and_bias(input_size, hidden_1_size, key)
    weight_mid, bias_mid, key = initialize_weight_and_bias(
        hidden_1_size, hidden_2_size, key
    )
    weight_out, bias_out, key = initialize_weight_and_bias(
        hidden_2_size, output_size, key
    )

    weights = [weight_in, weight_mid, weight_out]
    biases = [bias_in, bias_mid, bias_out]
    return weights, biases


def main():
    input_data = np.random.randn(100, 1)

    targets = input_data**3

    weights, biases, all_losses = train_network(input_data, targets)

    xax = np.linspace(-2, 2, 101)
    predicted = batched_neural_network(xax.reshape(-1, 1), weights, biases)

    plt.plot(xax, xax**3)
    plt.plot(xax, predicted, label="predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
