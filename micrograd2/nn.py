"""Neural network implementation using micrograd for automatic differentiation."""

import random

import numpy as np

from micrograd2.value import Value


class Neuron:
    """A single neuron in a neural network."""

    def __init__(self, nin, activation="relu"):
        assert nin > 1
        self.weights = [Value(0.0) for _ in range(nin)]
        self.bias = Value(0.0)
        self.activation = activation

    def initialize_gaussian(self, mean=0.0, std=0.01):
        """Initialize weights and bias with Gaussian distribution."""
        for weight in self.weights:
            weight.val = np.random.normal(mean, std)
        self.bias.val = np.random.normal(mean, std)

    def initialize_uniform(self, low=-1.0, high=1.0):
        """Initialize weights and bias with uniform distribution."""
        for weight in self.weights:
            weight.val = random.uniform(low, high)
        self.bias.val = random.uniform(low, high)

    def forward(self, inputs):
        """Compute the forward pass of the neuron."""
        weighted_sum = sum(w * xi for w, xi in zip(self.weights, inputs)) + self.bias
        if self.activation == "relu":
            return weighted_sum.relu()
        if self.activation == "sigmoid":
            return weighted_sum.sigmoid()
        return weighted_sum

    def parameters(self):
        """Return all trainable parameters."""
        return self.weights + [self.bias]

    def __repr__(self):
        """Return string representation of the neuron."""
        return f"Neuron(nin={len(self.weights)}, w={self.weights}, b={self.bias})"

    def __call__(self, inputs):
        """Call forward pass."""
        return self.forward(inputs)


class Layer:
    """A layer of neurons in a neural network."""

    def __init__(self, nin, nout, activation="relu"):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def initialize_gaussian(self, mean=0.0, std=0.01):
        """Initialize all neurons with Gaussian distribution."""
        for neuron in self.neurons:
            neuron.initialize_gaussian(mean, std)

    def initialize_uniform(self, low=-1.0, high=1.0):
        """Initialize all neurons with uniform distribution."""
        for neuron in self.neurons:
            neuron.initialize_uniform(low, high)

    def forward(self, inputs):
        """Compute the forward pass of the layer."""
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def backward(self, prev_grad):
        """Compute the backward pass of the layer."""
        for neuron in self.neurons:
            neuron.backward(prev_grad)

    def parameters(self):
        """Return all trainable parameters."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, inputs):
        """Call forward pass."""
        return self.forward(inputs)

    def __repr__(self):
        """Return string representation of the layer."""
        return f"Layer(nin={len(self.neurons[0].weights)}, nout={len(self.neurons)})"


class MLP:
    """Multi-layer perceptron neural network."""

    def __init__(self, nin, nouts, hidden_layer_sizes, activation="relu"):
        self.nin = nin
        self.nouts = nouts
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []
        self.activation = activation
        self.build()
        self.initialize_uniform()

    def initialize_gaussian(self, mean=0.0, std=0.01):
        """Initialize all layers with Gaussian distribution."""
        for layer in self.layers:
            layer.initialize_gaussian(mean, std)

    def initialize_uniform(self, low=-1.0, high=1.0):
        """Initialize all layers with uniform distribution."""
        for layer in self.layers:
            layer.initialize_uniform(low, high)

    def build(self):
        """Build the network architecture."""
        in_size = self.nin
        for out_size in self.hidden_layer_sizes:
            self.layers.append(Layer(in_size, out_size, self.activation))
            in_size = out_size
        self.layers.append(Layer(in_size, self.nouts, activation=None))

    def forward(self, inputs):
        """Compute the forward pass of the network."""
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        """Return all trainable parameters."""
        return [p for layer in self.layers for p in layer.parameters()]

    def reset_grad(self):
        """Reset gradients of all parameters."""
        for param in self.parameters():
            param.reset_grad(recursively=False)

    def __repr__(self):
        """Return string representation of the network."""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def __call__(self, inputs):
        """Call forward pass."""
        return self.forward(inputs)
