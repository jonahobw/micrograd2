"""Activation functions for micrograd2.

This module contains activation functions like ReLU and Sigmoid.
"""
import math
from typing import List
from .base import Op
from ..value import Value

class ReluOp(Op):
    """Rectified Linear Unit (ReLU) activation function."""
    def __init__(self, operand: Value) -> None:
        super().__init__([operand], 1)
        self.opName = "ReLU"

    def forward(self) -> float:
        return max(0, self.operands[0].val)

    def backward(self, prev_grad: float = 1.0) -> None:
        if self.operands[0].val > 0:
            self.operands[0].grad += prev_grad

class SigmoidOp(Op):
    """Sigmoid activation function."""
    def __init__(self, operand: Value) -> None:
        super().__init__([operand], 1)
        self.opName = "Sigmoid"

    def forward(self) -> float:
        return 1 / (1 + math.exp(-self.operands[0].val))

    def backward(self, prev_grad: float = 1.0) -> None:
        sigmoid_val = self.forward()
        self.operands[0].grad += prev_grad * sigmoid_val * (1 - sigmoid_val) 