"""Arithmetic operations for micrograd2.

This module contains basic arithmetic operations like add, multiply, divide, and power.
"""
import math
from typing import List

from ..value import Value
from .base import Op


class IdentityOp(Op):
    """Identity operation that returns its input unchanged."""

    def __init__(self, operand: Value) -> None:
        super().__init__([operand], 1)
        self.opName = "identity"

    def forward(self) -> float:
        return self.operands[0].val

    def backward(self, prev_grad: float = 1.0) -> None:
        self.operands[0].grad += prev_grad


class AddOp(Op):
    """Addition operation."""

    def __init__(self, operands: List[Value]) -> None:
        super().__init__(operands, 2)
        self.opName = "+"

    def forward(self) -> float:
        return sum([x.val for x in self.operands])

    def backward(self, prev_grad: float = 1.0) -> None:
        for value in self.operands:
            value.grad += prev_grad


class MulOp(Op):
    """Multiplication operation."""

    def __init__(self, operands: List[Value]) -> None:
        super().__init__(operands, 2)
        self.opName = "*"

    def forward(self) -> float:
        return math.prod([x.val for x in self.operands])

    def backward(self, prev_grad: float = 1.0) -> None:
        self.operands[0].grad += prev_grad * self.operands[1].val
        self.operands[1].grad += prev_grad * self.operands[0].val


class DivOp(Op):
    """Division operation."""

    def __init__(self, operands: List[Value]) -> None:
        super().__init__(operands, 2)
        self.opName = "/"

    def forward(self) -> float:
        if self.operands[1].val == 0:
            raise ValueError("Division by zero is not allowed")
        return self.operands[0].val / self.operands[1].val

    def backward(self, prev_grad: float = 1.0) -> None:
        if self.operands[1].val == 0:
            raise ValueError("Cannot compute gradient for division by zero")
        values = [x.val for x in self.operands]
        self.operands[0].grad += prev_grad / values[1]
        self.operands[1].grad += -1 * values[0] * prev_grad / (values[1] ** 2)


class PowOp(Op):
    """Power operation."""

    def __init__(self, operand: Value, exponent: float) -> None:
        super().__init__([operand], 1)
        self.opName = f"^{exponent}"
        self.exponent = exponent

    def forward(self) -> float:
        return self.operands[0].val ** self.exponent

    def backward(self, prev_grad: float = 1.0) -> None:
        n = self.exponent
        a = self.operands[0].val
        self.operands[0].grad += prev_grad * n * (a ** (n - 1))
