import random

import matplotlib.pyplot as plt
import numpy as np
from Value import Value


class PolynomialRegressionModel:
    def __init__(self, max_order):
        self.max_order = max_order

        # First coefficient is the bias.
        self.learnables = [Value(0, name=f"w_{i}") for i in range(max_order + 1)]

    def initializeUniform(self, lo=-1.0, hi=1.0):
        for w in self.learnables[1:]:  # keep bias as 0
            w.val = random.uniform(lo, hi)

    def initializeGaussian(self, mean=0.0, std=0.01):
        for w in self.learnables[1:]:
            w.val = np.random.normal(mean, std)

    def initializeConstant(self, val=1.0):
        for w in self.learnables[1:]:
            w.val = val

    def forward(self, x):
        if not isinstance(x, Value):
            x = Value(x, name="x")
        return sum([self.learnables[i] * x**i for i in range(self.max_order + 1)])

    def __call__(self, x):
        return self.forward(x)

    def update(self, lr, grad_clip=None):
        for learnable in self.learnables:
            learnable.val -= (
                min(learnable.grad * lr, grad_clip)
                if grad_clip
                else learnable.grad * lr
            )


def generateData(numDataPoints, degree, xRange=None, noise_range=None, coefRange=None):
    if not xRange:
        xRange = 5
    if not noise_range:
        noise_range = 1
    if not coefRange:
        coefRange = 1

    coefficients = np.random.random(degree + 1) * 2 * coefRange - coefRange
    x = np.linspace(-xRange, xRange, numDataPoints)
    y = np.zeros_like(x)
    for degree in range(degree + 1):
        y += coefficients[degree] * x**degree
    y += np.random.uniform(-noise_range, noise_range, numDataPoints)
    return x, y, coefficients


def generateDataOld(numDataPoints, degree=2, noise_range=0.1):
    coefficients = np.random.random(degree + 1) * 5
    x = np.linspace(-5, 5, numDataPoints)
    y = np.zeros_like(x)
    for deg in range(degree + 1):
        y += coefficients[deg] * x**deg
    y += np.random.uniform(-noise_range, noise_range, numDataPoints)
    return x, y, coefficients


def fitPolynomialModel(
    xTrain,
    yTrain,
    max_order,
    epochs=10,
    lr=0.001,
    draw=False,
    print_freq=1,
    grad_clip=None,
    weight_decay=None,
):
    lossVals = []
    drawn = False

    model = PolynomialRegressionModel(max_order)
    model.initializeGaussian()

    for i in range(epochs):
        epoch_loss = 0
        epoch_grad = [0] * len(model.learnables)
        for (x, y) in zip(xTrain, yTrain):
            y_pred = model(x)
            loss = (y - y_pred) ** 2  # MSE
            if weight_decay is not None:
                loss += weight_decay * sum(
                    [w**2 for w in model.learnables[1:]]
                )  # L2 regularization
            loss.name = f"Loss_{i}"
            epoch_loss += loss.val
            loss.backward()
            model.update(lr, grad_clip=grad_clip)
            epoch_grad = [
                epoch_grad[i] + learnable.grad
                for i, learnable in enumerate(model.learnables)
            ]
            loss.reset_grad()

        if not drawn and draw:
            d = loss.draw_dot(rankdir="LR")
            d.render(view=True)
            drawn = True

        if i % print_freq == 0:
            print(
                f"Epoch {i}, loss: {loss.val/len(xTrain):.4f}, "
                + f"vals: {[f'{learnable.val:.4f}' for learnable in model.learnables]} "
                + f"Gradients: {[f'{x/len(xTrain):.4f}' for x in epoch_grad]}"
            )

        lossVals.append(epoch_loss / len(xTrain))

    return lossVals, model


if __name__ == "__main__":
    degree = 4
    noise_range = 2
    numDataPoints = 10
    epochs = 30
    lr = 0.1 * 0.1**degree
    grad_clip = 0.01 / lr if degree > 1 else None
    weight_decay = 0.1
    draw = True
    print_freq = 1

    x_orig, y_orig, coef = generateData(
        numDataPoints, degree=degree, noise_range=noise_range
    )
    lossVals, model = fitPolynomialModel(
        x_orig,
        y_orig,
        max_order=degree,
        epochs=epochs,
        lr=lr,
        draw=draw,
        print_freq=print_freq,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
    )
    print("Data coefficients:", coef)

    plt.close("all")
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(lossVals)
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].scatter(x_orig, y_orig, label=f"Data: {coef}")
    preds = []
    for xi in x_orig:
        y_pred = model(xi)
        preds.append(y_pred.val)
    ax[1].plot(x_orig, preds, label=f"Predictions")
    ax[1].legend()
    ax[1].set_title("Predictions")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    y_range = min(y_orig), max(y_orig)
    pad = 0.1 * (y_range[1] - y_range[0]) + 0.1

    ax[1].set_ylim(y_range[0] - pad, y_range[1] + pad)
    plt.tight_layout()
    plt.show()
