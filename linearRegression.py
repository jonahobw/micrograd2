"""
A simple linear regression example using micrograd
"""

import matplotlib.pyplot as plt
import numpy as np
from micrograd.Value import Value

# Generate random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)

# m_true and b_true are random values between -10 and 10
m_true = np.random.random() * 20 - 10
b_true = np.random.random() * 20 - 10

Y = m_true * X + b_true + np.random.randn(100, 1)

# Define the linear regression model
x = Value(0.0, name="x")
y = Value(0.0, name="y")

# Define the model parameters
w = Value(0.0, name="w")
b = Value(0.0, name="b")

model = x * w + b

# Define the loss function
loss = (model - y) ** 2

num_epochs = 100
learning_rate = 0.01

for epoch in range(num_epochs):
    total_loss = 0
    for (x_data, y_data) in zip(X, Y):
        x.val = x_data
        y.val = y_data
        loss.reset()
        loss.forward()
        loss.backward()
        w.val -= learning_rate * w.grad
        b.val -= learning_rate * b.grad
        total_loss += loss.val
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(X)}, w: {w.val}, b: {b.val}")

# Get the final model parameters
print(f"Final model parameters: w = {w.val}, b = {b.val}")
print(f"True parameters: w = {m_true}, b = {b_true}")
print(f"Difference: w = {w.val - m_true}, b = {b.val - b_true}")

# Plot the model predictions
plt.scatter(X, Y, color='blue', label='Training data')
plt.plot(X, w.val * X + b.val, color='red', label='Model predictions')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
