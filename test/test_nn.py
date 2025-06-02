"""
Unit tests for the micrograd neural network classes (Neuron, Layer, MLP).
"""
import unittest
import numpy as np
from micrograd2.value import Value
from micrograd2.nn import Neuron, Layer, MLP

class TestMicrogradNN(unittest.TestCase):
    """Unit tests for the micrograd neural network classes (Neuron, Layer, MLP)."""

    def test_neuron_forward_relu(self):
        """Test forward pass of Neuron with ReLU activation."""
        n = Neuron(2, activation="relu")
        n.weights[0].val = 1.0
        n.weights[1].val = -2.0
        n.bias.val = 0.5
        x = [Value(2.0), Value(1.0)]
        out = n(x)
        expected = max(0, 1.0*2.0 + (-2.0)*1.0 + 0.5)
        self.assertAlmostEqual(out.val, expected, places=7)

    def test_neuron_forward_sigmoid(self):
        """Test forward pass of Neuron with sigmoid activation."""
        n = Neuron(2, activation="sigmoid")
        n.weights[0].val = 1.0
        n.weights[1].val = 1.0
        n.bias.val = 0.0
        x = [Value(0.0), Value(0.0)]
        out = n(x)
        expected = 1 / (1 + np.exp(0))
        self.assertAlmostEqual(out.val, expected, places=7)

    def test_layer_forward(self):
        """Test forward pass of a Layer with two neurons and ReLU activation."""
        l = Layer(2, 2, activation="relu")
        for i, neuron in enumerate(l.neurons):
            neuron.weights[0].val = 1.0 + i
            neuron.weights[1].val = -1.0 - i
            neuron.bias.val = 0.0
        x = [Value(1.0), Value(2.0)]
        out = l(x)
        self.assertEqual(len(out), 2)
        expected0 = max(0, 1.0*1.0 + (-1.0)*2.0)
        expected1 = max(0, 2.0*1.0 + (-2.0)*2.0)
        self.assertAlmostEqual(out[0].val, expected0, places=7)
        self.assertAlmostEqual(out[1].val, expected1, places=7)

    def test_mlp_forward(self):
        """Test forward pass of a simple MLP with one hidden layer."""
        mlp = MLP(2, 1, [2], activation="relu")
        # Set all weights and biases to known values
        for layer in mlp.layers:
            for neuron in layer.neurons:
                for w in neuron.weights:
                    w.val = 1.0
                neuron.bias.val = 0.0
        x = [Value(1.0), Value(2.0)]
        out = mlp(x)
        # Forward through first layer: each neuron: 1*1 + 1*2 = 3, relu(3) = 3
        # Second layer: 1*3 + 1*3 = 6, no activation
        self.assertTrue(isinstance(out, Value))
        self.assertAlmostEqual(out.val, 6.0, places=7)

    def test_mlp_backward(self):
        """Test backward pass (gradient flow) through a simple MLP."""
        mlp = MLP(2, 1, [2], activation="relu")
        for layer in mlp.layers:
            for neuron in layer.neurons:
                for w in neuron.weights:
                    w.val = 1.0
                neuron.bias.val = 0.0
        x = [Value(1.0), Value(2.0)]
        y_true = 10.0
        out = mlp(x)
        loss = (out - y_true) ** 2
        loss.backward()
        # Check that gradients are not zero
        grads = [p.grad for p in mlp.parameters()]
        self.assertTrue(any(abs(g) > 0 for g in grads))

if __name__ == "__main__":
    unittest.main() 