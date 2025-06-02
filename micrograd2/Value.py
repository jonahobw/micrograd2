"""A micrograd implementation for automatic differentiation, adapted from Karpathy.

This module provides a simple implementation of automatic differentiation
using a computational graph. It supports basic mathematical operations
and common activation functions used in neural networks.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple, Union

from graphviz import Digraph


class Op(ABC):
    """Base class for all operations in the computational graph."""

    def __init__(self, operands: List["Value"], num_args: int) -> None:
        super().__init__()
        assert len(operands) == num_args
        self.operands: List["Value"] = operands
        self.op_name: str = ""

    @abstractmethod
    def forward(self) -> float:
        """Compute the forward pass of the operation."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, prev_grad: float = 1.0) -> None:
        """Compute the backward pass (gradient) of the operation."""
        raise NotImplementedError


class IdentityOp(Op):
    """Identity operation that returns its input unchanged."""

    def __init__(self, operand: "Value") -> None:
        super().__init__([operand], 1)
        self.op_name = "identity"

    def forward(self) -> float:
        """Return the input value unchanged."""
        return self.operands[0].val

    def backward(self, prev_grad: float = 1.0) -> None:
        """Pass the gradient through unchanged."""
        self.operands[0].grad = prev_grad


class AddOp(Op):
    """Addition operation that sums its inputs."""

    def __init__(self, operands: List["Value"]) -> None:
        super().__init__(operands, 2)
        self.op_name = "+"

    def forward(self) -> float:
        """Sum the input values."""
        return sum(x.val for x in self.operands)

    def backward(self, prev_grad: float = 1.0) -> None:
        """Distribute the gradient equally to all inputs."""
        for value in self.operands:
            value.grad += prev_grad


class MulOp(Op):
    """Multiplication operation that multiplies its inputs."""

    def __init__(self, operands: List["Value"]) -> None:
        super().__init__(operands, 2)
        self.op_name = "*"

    def forward(self) -> float:
        """Multiply the input values."""
        return math.prod(x.val for x in self.operands)

    def backward(self, prev_grad: float = 1.0) -> None:
        """Apply the chain rule for multiplication."""
        self.operands[0].grad += prev_grad * self.operands[1].val
        self.operands[1].grad += prev_grad * self.operands[0].val


class DivOp(Op):
    """Division operation that divides its inputs."""

    def __init__(self, operands: List["Value"]) -> None:
        super().__init__(operands, 2)
        self.op_name = "/"

    def forward(self) -> float:
        """Divide the first input by the second."""
        if self.operands[1].val == 0:
            raise ValueError("Division by zero is not allowed")
        return self.operands[0].val / self.operands[1].val

    def backward(self, prev_grad: float = 1.0) -> None:
        """Apply the chain rule for division."""
        if self.operands[1].val == 0:
            raise ValueError("Cannot compute gradient for division by zero")
        values = [x.val for x in self.operands]
        self.operands[0].grad += prev_grad / values[1]
        self.operands[1].grad += -1 * values[0] * prev_grad / (values[1] ** 2)


class PowOp(Op):
    """Power operation that raises a value to a power."""

    def __init__(self, operand: "Value", exponent: Union[float, int, "Value"]) -> None:
        super().__init__([operand], 1)
        self.op_name = f"^{exponent}"
        self.exponent = exponent if isinstance(exponent, (float, int)) else exponent.val

    def forward(self) -> float:
        """Raise the input to the specified power."""
        return self.operands[0].val ** self.exponent

    def backward(self, prev_grad: float = 1.0) -> None:
        """Apply the chain rule for power."""
        exp = self.exponent
        activation = self.operands[0].val
        if activation == 0.0 and exp <= 0:
            return  # cannot raise 0 to a negative power
        self.operands[0].grad += prev_grad * exp * (activation ** (exp - 1))


class ReluOp(Op):
    """Rectified Linear Unit activation function."""

    def __init__(self, operand: "Value") -> None:
        super().__init__([operand], 1)
        self.op_name = "ReLU"

    def forward(self) -> float:
        """Apply ReLU activation."""
        return max(0, self.operands[0].val)

    def backward(self, prev_grad: float = 1.0) -> None:
        """Apply the gradient for ReLU."""
        if self.operands[0].val > 0:
            self.operands[0].grad += prev_grad


class SigmoidOp(Op):
    """Sigmoid activation function."""

    def __init__(self, operand: "Value") -> None:
        super().__init__([operand], 1)
        self.op_name = "Sigmoid"
        self.memory = None

    def forward(self) -> float:
        """Apply sigmoid activation."""
        self.memory = 1 / (1 + math.exp(-self.operands[0].val))
        return self.memory

    def backward(self, prev_grad: float = 1.0) -> None:
        """Apply the gradient for sigmoid."""
        self.operands[0].grad += prev_grad * self.memory * (1 - self.memory)


class AbsOp(Op):
    """Absolute value operation."""

    def __init__(self, operand: "Value") -> None:
        super().__init__([operand], 1)
        self.op_name = "abs"

    def forward(self) -> float:
        """Compute absolute value."""
        return abs(self.operands[0].val)

    def backward(self, prev_grad: float = 1.0) -> None:
        """Apply the gradient for absolute value."""
        if self.operands[0].val > 0:
            self.operands[0].grad += prev_grad
        elif self.operands[0].val < 0:
            self.operands[0].grad -= prev_grad


class Value:
    """Represents a value in the computational graph with automatic differentiation."""

    def __init__(
        self,
        val: Union[int, float],
        operation: Op = None,
        children: Tuple["Value", ...] = (),
        name: Optional[str] = None,
    ) -> None:
        self.val: float = float(val)
        assert isinstance(self.val, (int, float)), "Value must be a number"
        self.operation: Op = operation if operation else IdentityOp(self)
        self.children: Tuple["Value", ...] = children
        self.grad: float = 0.0
        self.name: str = name

    def detach(self) -> None:
        """Detach this value from the computational graph."""
        self.operation = IdentityOp(self)
        self.children = ()
        self.grad = 0.0

    def reset_grad(self, recursively: bool = True) -> None:
        """Reset the gradient of this value and optionally its children."""
        self.grad = 0
        if recursively:
            for child in self.children:
                child.reset_grad(recursively=True)

    def topological_sort(self) -> List["Value"]:
        """Perform a topological sort of the computational graph."""
        seen = set()
        values = []

        def dfs(node):
            seen.add(node)
            for child in node.children:
                if child not in seen:
                    dfs(child)
            values.append(node)

        dfs(self)
        return values

    def forward(self, recursively: bool = True) -> float:
        """Compute the forward pass of this value."""
        if not recursively:
            self.val = self.operation.forward()
            return self.val

        values = self.topological_sort()

        for value in values:
            value.forward(recursively=False)
        return self.val

    def backward(self, prev_grad: float = 1.0) -> None:
        """Compute the backward pass (gradient) of this value."""
        self.grad = prev_grad

        values = self.topological_sort()

        for value in reversed(values):
            value.operation.backward(prev_grad=value.grad)

    def relu(self) -> "Value":
        """Apply ReLU activation to this value."""
        new_val = Value(0, operation=ReluOp(self), children=(self,))
        new_val.forward(recursively=False)
        return new_val

    def sigmoid(self) -> "Value":
        """Apply sigmoid activation to this value."""
        new_val = Value(0, operation=SigmoidOp(self), children=(self,))
        new_val.forward(recursively=False)
        return new_val

    def abs(self) -> "Value":
        """Apply absolute value to this value."""
        new_val = Value(0, operation=AbsOp(self), children=(self,))
        new_val.forward(recursively=False)
        return new_val

    def __add__(self, other: Union["Value", float, int]) -> "Value":
        """Add this value with another value or number."""
        other = other if isinstance(other, Value) else Value(other)
        new_val = Value(0, operation=AddOp((self, other)), children=(self, other))
        new_val.forward(recursively=False)
        return new_val

    def __mul__(self, other: Union["Value", float, int]) -> "Value":
        """Multiply this value with another value or number."""
        other = other if isinstance(other, Value) else Value(other)
        new_val = Value(0, operation=MulOp((self, other)), children=(self, other))
        new_val.forward(recursively=False)
        return new_val

    def __truediv__(self, other: Union["Value", float, int]) -> "Value":
        """Divide this value by another value or number."""
        other = other if isinstance(other, Value) else Value(other)
        new_val = Value(0, operation=DivOp((self, other)), children=(self, other))
        new_val.forward(recursively=False)
        return new_val

    def __pow__(self, exponent: Union[float, int]) -> "Value":
        """Raise this value to a power."""
        new_val = Value(0, operation=PowOp(self, exponent), children=(self,))
        new_val.forward(recursively=False)
        return new_val

    def __neg__(self) -> "Value":
        """Negate this value."""
        return self * -1

    def __radd__(self, other: Union[float, int]) -> "Value":
        """Add a number to this value (right addition)."""
        return self + other

    def __sub__(self, other: Union["Value", float, int]) -> "Value":
        """Subtract another value or number from this value."""
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> "Value":
        """Subtract this value from a number (right subtraction)."""
        return other + (-self)

    def __rmul__(self, other: Union[float, int]) -> "Value":
        """Multiply a number with this value (right multiplication)."""
        return self * other

    def __rtruediv__(self, other: Union[float, int]) -> "Value":
        """Divide a number by this value (right division)."""
        return other * self**-1

    def draw_dot(self, output_format: str = "png", rankdir: str = "LR") -> Digraph:
        """Draw the computational graph using graphviz."""
        assert rankdir in ["LR", "TB"]
        nodes, edges = trace(self)
        dot = Digraph(format=output_format, graph_attr={"rankdir": rankdir})

        for node in nodes:
            dot.node(
                name=str(id(node)),
                label=f"{{ {node.name} | val {node.val:.4f} | grad {node.grad:.4f} }}",
                shape="record",
            )
            if not isinstance(node.operation, IdentityOp):
                dot.node(
                    name=str(id(node)) + node.operation.op_name,
                    label=node.operation.op_name,
                    shape="ellipse",
                )
                dot.edge(str(id(node)) + node.operation.op_name, str(id(node)))

        for node1, node2 in edges:
            dot.edge(str(id(node1)), str(id(node2)) + node2.operation.op_name)

        return dot

    def __repr__(self) -> str:
        """Return a string representation of this value."""
        return f"{self.name}: value={self.val:.4f}, grad={self.grad:.4f}"


def trace(root: Value, set_names: bool = True) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    """Trace the computational graph."""
    nodes, edges = set(), set()
    idx = 0

    def build(node: Value) -> None:
        nonlocal idx
        if node not in nodes:
            nodes.add(node)
            if node.name == "" and set_names:
                node.name = f"v{idx}"
                idx += 1
            for child in node.children:
                edges.add((child, node))
                build(child)

    build(root)
    return nodes, edges
