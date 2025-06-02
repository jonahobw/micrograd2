"""
Module for polynomial regression using automatic differentiation.
"""

import random
from typing import Tuple, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from micrograd2.value import Value


class PolynomialRegressionModel:
    """A polynomial regression model using automatic differentiation."""

    def __init__(self, max_order: int) -> None:
        """Initialize the polynomial regression model.

        Args:
            max_order (int): Maximum order of the polynomial.
        """
        self.max_order = max_order

        # First coefficient is the bias.
        self.learnables = [Value(0, name=f"w_{i}") for i in range(max_order + 1)]

    def initialize_uniform(self, low: float = -1.0, high: float = 1.0) -> None:
        """Initialize weights with uniform distribution."""
        for weight in self.learnables[1:]:  # keep bias as 0
            weight.val = random.uniform(low, high)

    def initialize_gaussian(self, mean: float = 0.0, std: float = 0.01) -> None:
        """Initialize weights with Gaussian distribution."""
        for weight in self.learnables[1:]:
            weight.val = np.random.normal(mean, std)

    def initialize_constant(self, value: float = 1.0) -> None:
        """Initialize weights with constant value."""
        for weight in self.learnables[1:]:
            weight.val = value

    def forward(self, x_value: Union[float, Value]) -> Value:
        """Forward pass of the model.

        Args:
            x_value (float or Value): Input value.

        Returns:
            Value: Output of the polynomial model.
        """
        if not isinstance(x_value, Value):
            x_value = Value(x_value, name="x")
        return sum(weight * x_value**i for i, weight in enumerate(self.learnables))

    def __call__(self, x_value: Union[float, Value]) -> Value:
        """Call forward pass of the model."""
        return self.forward(x_value)

    def update(self, learning_rate: float, grad_clip: Optional[float] = None) -> None:
        """Update model parameters using gradient descent.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            grad_clip (float, optional): Maximum gradient value. Defaults to None.
        """
        for learnable in self.learnables:
            # Ensure gradient is not None before updating
            if learnable.grad is None:
                continue
            
            if grad_clip is not None:
                if learnable.grad >= 0:
                    update = min(learnable.grad, grad_clip)
                else:
                    update = max(learnable.grad, -grad_clip)
            else:
                update = learnable.grad
            
            learnable.val -= update * learning_rate


def generate_data(
    num_data_points: int,
    degree: int,
    x_range: Optional[float] = None,
    noise_range: Optional[float] = None,
    coef_range: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic polynomial data.

    Args:
        num_data_points (int): Number of data points to generate.
        degree (int): Degree of the polynomial.
        x_range (float, optional): Range of x values. Defaults to 5.
        noise_range (float, optional): Range of noise to add. Defaults to 1.
        coef_range (float, optional): Range of coefficients. Defaults to 1.

    Returns:
        tuple: (x values, y values, true coefficients)
    """
    x_range = 50.0 if x_range is None else x_range
    noise_range = 1.0 if noise_range is None else noise_range
    coef_range = 1.0 if coef_range is None else coef_range

    coefficients = np.random.random(degree + 1) * 2 * coef_range - coef_range
    x_values = np.linspace(-x_range, x_range, num_data_points)
    y_values = np.zeros_like(x_values)
    for deg in range(degree + 1):
        y_values += coefficients[deg] * x_values**deg
    y_values += np.random.uniform(-noise_range, noise_range, num_data_points)
    return x_values, y_values, coefficients


class TrainingConfig:
    """Configuration for training the polynomial regression model."""

    def __init__(
        self,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        draw: bool = False,
        print_freq: int = 1,
        grad_clip: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ):
        """Initialize training configuration.

        Args:
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for gradient descent.
            draw (bool): Whether to draw computation graph.
            print_freq (int): Frequency of printing progress.
            grad_clip (float, optional): Maximum gradient value.
            weight_decay (float, optional): L2 regularization strength.
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.draw = draw
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay


def fit_polynomial_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_order: int,
    config: TrainingConfig,
) -> Tuple[List[float], PolynomialRegressionModel]:
    """Fit a polynomial regression model to the data.

    Args:
        x_train (np.ndarray): Training x values.
        y_train (np.ndarray): Training y values.
        max_order (int): Maximum order of the polynomial.
        config (TrainingConfig): Training configuration.

    Returns:
        tuple: (loss values, trained model)
    """
    training_losses = []
    drawn = False

    model = PolynomialRegressionModel(max_order)
    model.initialize_gaussian()

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        epoch_grad = [0] * len(model.learnables)
        for x_val, y_val in zip(x_train, y_train):
            pred = model(x_val)
            loss = (y_val - pred) ** 2  # MSE
            if config.weight_decay is not None:
                loss += config.weight_decay * sum(
                    w**2 for w in model.learnables[1:]
                )  # L2 regularization
            loss.name = f"Loss_{epoch}"
            epoch_loss += loss.val
            loss.backward()
            model.update(config.learning_rate, grad_clip=config.grad_clip)
            epoch_grad = [
                epoch_grad[i] + learnable.grad
                for i, learnable in enumerate(model.learnables)
            ]
            loss.reset_grad()

        if not drawn and config.draw:
            graph = loss.draw_dot(rankdir="LR")
            graph.render(view=True)
            drawn = True

        if epoch % config.print_freq == 0:
            print(
                f"Epoch {epoch}, loss: {loss.val/len(x_train):.4f}, "
                + f"vals: {[f'{learnable.val:.4f}' for learnable in model.learnables]} "
                + f"Gradients: {[f'{x/len(x_train):.4f}' for x in epoch_grad]}"
            )

        training_losses.append(epoch_loss / len(x_train))

    return training_losses, model


if __name__ == "__main__":
    DEGREE = 3
    NOISE_RANGE = 2
    NUM_DATA_POINTS = 50
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.1 * 0.1**DEGREE
    GRAD_CLIP = 0.01 / LEARNING_RATE if DEGREE > 1 else None
    WEIGHT_DECAY = None
    DRAW = True
    PRINT_FREQ = 1

    config = TrainingConfig(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        draw=DRAW,
        print_freq=PRINT_FREQ,
        grad_clip=GRAD_CLIP,
        weight_decay=WEIGHT_DECAY,
    )

    x_orig, y_orig, coef = generate_data(
        NUM_DATA_POINTS, degree=DEGREE, noise_range=NOISE_RANGE
    )
    training_losses, trained_model = fit_polynomial_model(
        x_orig,
        y_orig,
        max_order=DEGREE,
        config=config,
    )
    print("Data coefficients:", coef)

    plt.close("all")
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(training_losses)
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].scatter(x_orig, y_orig, label=f"Data: {coef}")
    predictions = []
    for xi in x_orig:
        pred = trained_model(xi)
        predictions.append(pred.val)
    ax[1].plot(x_orig, predictions, label="Predictions")
    ax[1].legend()
    ax[1].set_title("Predictions")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    y_range = min(y_orig), max(y_orig)
    pad = 0.1 * (y_range[1] - y_range[0]) + 0.1

    ax[1].set_ylim(y_range[0] - pad, y_range[1] + pad)
    plt.tight_layout()
    plt.show()
