import random

import numpy as np

from micrograd2.Value import Value


class Neuron:
    def __init__(self, nin, activation="relu"):
        assert nin > 1
        self.w = [Value(0.0) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def initializeGaussian(self, mean=0.0, std=0.01):
        for w in self.w:
            w.val = np.random.normal(mean, std)
        self.b.val = np.random.normal(mean, std)

    def initializeUniform(self, lo=-1.0, hi=1.0):
        for w in self.w:
            w.val = random.uniform(lo, hi)
        self.b.val = random.uniform(lo, hi)

    def forward(self, x):
        v = sum(w * xi for w, xi in zip(self.w, x)) + self.b
        if self.activation == "relu":
            out = v.relu()
        elif self.activation == "sigmoid":
            out = v.sigmoid()
        else:
            out = v
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin={len(self.w)}, w={self.w}, b={self.b})"

    def __call__(self, x):
        return self.forward(x)


class NeuronOld:
    def __init__(self, nin, activation="relu"):
        assert nin > 1
        self.inp = [Value(0.0) for _ in range(nin)]
        self.w = [Value(0.0) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation
        self.out = self.constructGraph()

    def constructGraph(self):
        v = self.inp[0] * self.w[0]
        for (w, inp) in zip(self.w[1:], self.inp[1:]):
            v += inp * w
        v += self.b
        if self.activation == "relu":
            return v.relu()
        elif self.activation == "sigmoid":
            return v.sigmoid()
        else:
            return v

    def forward(self, x):
        for (xi, inp) in zip(x, self.inp):
            inp.val = xi
        return self.out.forward()

    def backward(self, prev_grad):
        self.out.backward(prev_grad)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin={len(self.inp)}, w={self.w}, b={self.b})"

    def __call__(self, x):
        return self.forward(x)


class Layer:
    def __init__(self, nin, nout, activation="relu"):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def initializeGaussian(self, mean=0.0, std=0.01):
        for n in self.neurons:
            n.initializeGaussian(mean, std)

    def initializeUniform(self, lo=-1.0, hi=1.0):
        for n in self.neurons:
            n.initializeUniform(lo, hi)

    def forward(self, x):
        out = [n(x) for n in self.neurons]
        # out = []
        # for n in self.neurons:
        #     print(f"Neuron: {n}")
        #     n_out = n(x)
        #     out.append(n_out)
        return out[0] if len(out) == 1 else out

    def backward(self, prev_grad):
        for neuron in self.neurons:
            neuron.backward(prev_grad)

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Layer(nin={len(self.neurons[0].w)}, nout={len(self.neurons)})"


class MLP:
    def __init__(self, nin, nouts, hidden_layer_sizes, activation="relu"):
        self.nin = nin
        self.nouts = nouts
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []
        self.activation = activation
        self.build()
        self.initializeUniform()

    def initializeGaussian(self, mean=0.0, std=0.01):
        for layer in self.layers:
            layer.initializeGaussian(mean, std)

    def initializeUniform(self, lo=-1.0, hi=1.0):
        for layer in self.layers:
            layer.initializeUniform(lo, hi)

    def build(self):
        in_size = self.nin
        for out_size in self.hidden_layer_sizes:
            self.layers.append(Layer(in_size, out_size, self.activation))
            in_size = out_size
        self.layers.append(Layer(in_size, self.nouts, activation=None))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # print(f"Layer {i}")
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def reset_grad(self):
        for p in self.parameters():
            p.reset_grad(recursively=False)

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def __call__(self, x):
        return self.forward(x)
