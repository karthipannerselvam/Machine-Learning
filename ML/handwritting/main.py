import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target'].astype(int)

# Normalize the input data
X = X / 255.0

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.final_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output = self.sigmoid(self.final_input)
        return self.output

    def backward(self, X, y):
        output_loss = y - self.output
        d_output = output_loss * self.sigmoid_derivative(self.output)

        hidden_loss = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_loss * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:  # Print every 100 epochs
                loss = np.mean(np.square(y - self.output))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        return self.forward(X)

# Initialize the neural network
input_size = 784  # 28*28 pixels
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 10  # 10 classes (digits 0-9)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
nn.train(X_train, y_train, epochs=100)  # Start with fewer epochs

# Accuracy calculation
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

# Predict on the test set
y_pred = nn.predict(X_test)

# Calculate test accuracy
test_accuracy = accuracy(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Function to plot misclassified images
def plot_misclassified_images(X, y_true, y_pred):
    if isinstance(X, pd.DataFrame):
        X = X.values
    misclassified_indices = np.where(np.argmax(y_true, axis=1) != np.argmax(y_pred, axis=1))[0]
    
    plt.figure(figsize=(10, 5))
    for i, index in enumerate(misclassified_indices[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[index].reshape(28, 28), cmap='gray')
        plt.title(f'True: {np.argmax(y_true[index])}, Pred: {np.argmax(y_pred[index])}')
        plt.axis('off')
    plt.show()

# Plot misclassified images
plot_misclassified_images(X_test, y_test, y_pred)

# Display sample predictions
print("Sample Predictions:")
for i in range(5):  # Print first 5 predictions
    print(f'True: {np.argmax(y_test[i])}, Pred: {np.argmax(y_pred[i])}')
