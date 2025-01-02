import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X_train = np.random.rand(100, 4) * 100  # Random feature data between 0 and 100
y_train = (np.random.rand(100, 1) > 0.5).astype(int)  # Random binary labels (0 or 1)

X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
# Normalize the data
X_train_normalized = (X_train - X_train_mean) / X_train_std

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Forward propagation function (same for both models)
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  # Hidden layer linear step
    A1 = relu(Z1)           # Hidden layer activation
    Z2 = np.dot(A1, W2) + b2 # Output layer linear step
    A2 = sigmoid(Z2)        # Output layer activation (probability)
    return A2

def predict(X, W1, b1, W2, b2):
    probabilities = forward_propagation(X, W1, b1, W2, b2)
    predictions = (probabilities > 0.5).astype(int)
    return predictions

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# @Forward Propagation: 
# Initialize weights and biases (randomly)
W1_model1 = np.random.randn(X_train_normalized.shape[1], 4) * 0.01
b1_model1 = np.zeros((1, 4))
W2_model1 = np.random.randn(4, 1) * 0.01
b2_model1 = np.zeros((1, 1))

# Forward pass only (without learning)
y_pred_model1 = predict(X_train_normalized, W1_model1, b1_model1, W2_model1, b2_model1)
acc_model1 = accuracy(y_train, y_pred_model1)

# @Forward Propagation with Backpropagation (Learning)
W1_model2 = np.random.randn(X_train_normalized.shape[1], 4) * 0.01
b1_model2 = np.zeros((1, 4))
W2_model2 = np.random.randn(4, 1) * 0.01
b2_model2 = np.zeros((1, 1))

# Hyperparameters for backpropagation
learning_rate = 0.1
num_epochs = 50000
losses = []  # To track loss for each epoch

# Backpropagation function
def backpropagation(X, y, W1, b1, W2, b2, learning_rate):
    m = X.shape[0]
    
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Compute the loss (binary cross-entropy)
    cost = -(1/m) * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))
    
    # Backpropagation (compute gradients)
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    
    dZ1 = np.dot(dZ2, W2.T) * (A1 > 0)  # Derivative of ReLU
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    # Update weights and biases using gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2, cost

# using Backpropagation with Loss Tracking
for epoch in range(num_epochs):
    W1_model2, b1_model2, W2_model2, b2_model2, cost = backpropagation(
        X_train_normalized, y_train, W1_model2, b1_model2, W2_model2, b2_model2, learning_rate
    )
    if epoch % 100 == 0:  # Print loss every 100 epochs
        print(f"Epoch {epoch}: Loss = {cost:.4f}")
    losses.append(cost)

# Make Predictions for (With Backpropagation)
y_pred_model2 = predict(X_train_normalized, W1_model2, b1_model2, W2_model2, b2_model2)
acc_model2 = accuracy(y_train, y_pred_model2)

# Print Results
print("\nModel 1 (Forward Propagation Only) - Accuracy: {:.2f}%".format(acc_model1))
print("Model 2 (With Backpropagation) - Accuracy: {:.2f}%".format(acc_model2))

# Loss Visualization for Model 2
plt.plot(range(0, num_epochs, 100), losses[::100], marker='o')
plt.title("Loss Function Over Epochs (Model 2 - Backpropagation)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
