import numpy as np

# sigmoid activation function and its derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# input dataset for XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# expected outputs
targets = np.array([[0], [1], [1], [0]])

# initialize weights and biases with random values
np.random.seed(42)
weights_input_hidden = np.random.rand(2, 2)
weights_hidden_output = np.random.rand(2, 1)
bias_hidden = np.random.rand(2)
bias_output = np.random.rand(1)

# learning rate
learning_rate = 0.1
# number of training iterations
epochs = 10000

# training process
for epoch in range(epochs):
    # forward pass
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_layer_input)

    # calculate error
    error = targets - final_output

    # backpropagation
    d_final_output = error * sigmoid_derivative(final_output)
    error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * learning_rate
    bias_output += np.sum(d_final_output, axis=0) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate

# testing the trained network
print("trained outputs:")
for input_data in inputs:
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_layer_input)
    print(f"input: {input_data} => output: {final_output}")
