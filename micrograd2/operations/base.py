"""Base operation class for micrograd2.

This module contains the base operation class that all other operations inherit from.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..value import Value

class Op(ABC):
    """Base class for all operations in the computational graph.
    
    Attributes:
        operands: List of input values for the operation
        numArgs: Number of expected operands
        opName: String identifier for the operation
    """
    def __init__(self, operands: List[Value], numArgs: int) -> None:
        super().__init__()
        assert len(operands) == numArgs
        self.operands: List[Value] = operands
        self.opName: str = ""

    @abstractmethod
    def forward(self) -> float:
        """Compute the forward pass of the operation.
        
        Returns:
            float: The result of the operation
        """
        pass

    @abstractmethod
    def backward(self, prev_grad: float = 1.0) -> None:
        """Compute the backward pass (gradient) of the operation.
        
        Args:
            prev_grad: The gradient from the next operation in the graph
        """
        pass 