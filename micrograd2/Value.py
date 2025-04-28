from abc import ABC, abstractmethod
import math
from graphviz import Digraph
from typing import List, Tuple, Set, Optional, Union, Any


class Op(ABC):
    """Base class for all operations in the computational graph.
    
    Attributes:
        operands: List of input values for the operation
        numArgs: Number of expected operands
        opName: String identifier for the operation
    """
    def __init__(self, operands: List['Value'], numArgs: int) -> None:
        super().__init__()
        assert len(operands) == numArgs
        self.operands: List['Value'] = operands
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


class IdentityOp(Op):
    def __init__(self, operand):
        super().__init__(operand, 1)
        self.opName = "identity"

    def forward(self):
        return self.operands[0].val

    def backward(self, prev_grad=1):
        return prev_grad


class AddOp(Op):
    def __init__(self, operands):
        super().__init__(operands, 2)
        self.opName = "+"

    def forward(self):
        return sum([x.val for x in self.operands])

    def backward(self, prev_grad=1):
        for value in self.operands:
            value.grad += prev_grad
    
class MulOp(Op):
    def __init__(self, operands):
        super().__init__(operands, 2)
        self.opName = "*"
    
    def forward(self):
        return math.prod([x.val for x in self.operands])

    def backward(self, prev_grad=1):
        self.operands[0].grad += prev_grad * self.operands[1].val
        self.operands[1].grad += prev_grad * self.operands[0].val

class DivOp(Op):
    def __init__(self, operands):
        super().__init__(operands, 2)
        self.opName = "/"
    
    def forward(self):
        if self.operands[1].val == 0:
            raise ValueError("Division by zero is not allowed")
        return self.operands[0].val / self.operands[1].val

    def backward(self, prev_grad=1):
        if self.operands[1].val == 0:
            raise ValueError("Cannot compute gradient for division by zero")
        values = [x.val for x in self.operands]
        self.operands[0].grad += prev_grad/values[1]
        self.operands[1].grad += -1 * values[0] * prev_grad / (values[1] ** 2)


class PowOp(Op):
    def __init__(self, operand, exponent):
        super().__init__(operand, 1)
        self.opName = f"^{exponent}"
        self.exponent = exponent

    def forward(self):
        return self.operands[0].val ** self.exponent

    def backward(self, prev_grad=1):
        n = self.exponent
        a = self.operands[0].val
        self.operands[0].grad += prev_grad * n * (a ** (n - 1))

class ReluOp(Op):
    def __init__(self, operand):
        super().__init__(operand, 1)
        self.opName = "ReLU"

    def forward(self):
        return max(0, self.operands[0].val)

    def backward(self, prev_grad=1):
        if self.operands[0].val > 0:
            self.operands[0].grad += prev_grad

class SigmoidOp(Op):
    def __init__(self, operand):
        super().__init__(operand, 1)
        self.opName = "Sigmoid"

    def forward(self):
        return 1 / (1 + math.exp(-self.operands[0].val))

    def backward(self, prev_grad=1):
        self.operands[0].grad += prev_grad * self.operands[0].val * (1 - self.operands[0].val)

class Value:
    """Represents a value in the computational graph with automatic differentiation.
    
    Attributes:
        val: The numerical value
        op: The operation that produced this value
        children: Tuple of child values in the computational graph
        grad: The gradient of this value
        name: Unique identifier for this value
    """
    varNum: int = 0
    varNames: Set[str] = set()

    def __init__(self, val: Union[int, float], op: Optional[Op] = None, 
                 children: Tuple['Value', ...] = (), name: Optional[str] = None) -> None:
        self.val: float = float(val)
        assert isinstance(val, (int, float)), "Value must be a number"
        self.op: Op = op if op else IdentityOp((self,))
        self.children: Tuple['Value', ...] = children
        self.grad: float = 0.0
        self.name: str = self.setName(name)

    def setName(self, name: Optional[str]) -> str:
        """Set a unique name for this value.
        
        Args:
            name: Optional name to use. If None, generates a unique name.
            
        Returns:
            str: The assigned name
            
        Raises:
            ValueError: If the requested name is already in use
        """
        # remove old name
        if hasattr(self, "name") and self.name in self.varNames:
            self.varNames.remove(self.name)
        
        if name is None:
            name = f"v{Value.varNum}"
            Value.varNum += 1

        if name in self.varNames:
            raise ValueError(f"Name {name} already exists.")
        else:
            self.varNames.add(name)
        return name

    def detach(self) -> None:
        """Detach this value from the computational graph.
        
        This removes all references to children and sets the operation to identity.
        Also cleans up the name from varNames.
        """
        if self.name in self.varNames:
            self.varNames.remove(self.name)
        self.op = IdentityOp((self,))
        self.children = ()
        self.grad = 0.0

    def reset_grad(self, recursively=True):
        self.grad = 0
        if recursively:
            for child in self.children:
                child.reset(recursively=True)
    
    def topologicalSort(self):
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

    def forward(self, recursively=True):
        if not recursively:
            self.val = self.op.forward()
            return self.val
        
        values = self.topologicalSort()

        for value in values:
            value.forward(recursively=False)
        return self.val

    def backward(self, prev_grad=1):
        self.grad = prev_grad

        values = self.topologicalSort()

        for value in reversed(values):
            value.op.backward(prev_grad=value.grad)
    
    def relu(self):
        return Value(max(0, self.val), op=ReluOp((self,)), children=(self,))
    
    def sigmoid(self):
        new_val = Value(0, op=SigmoidOp((self,)), children=(self,))
        new_val.forward(recursively=False)
        return new_val

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        new_val = Value(0, op=AddOp((self, other)), children=(self, other))
        new_val.forward(recursively=False)
        return new_val

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        new_val = Value(self.val * other.val, op=MulOp((self, other)), children=(self, other))
        new_val.forward(recursively=False)
        return new_val
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        new_val = Value(self.val/other.val, op=DivOp((self, other)), children=(self, other))
        new_val.forward(recursively=False)
        return new_val
    
    def __pow__(self, exponent):
        new_val = Value(self.val ** exponent, op=PowOp((self,), exponent), children=(self,))
        new_val.forward(recursively=False)
        return new_val
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def draw_dot(root, format='png', rankdir='LR'):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        assert rankdir in ['LR', 'TB']
        nodes, edges = trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) 
        
        for n in nodes:
            dot.node(name=str(id(n)), label = "{ %s | val %.4f | grad %.4f }" % (n.name, n.val, n.grad), shape='record')
            if not isinstance(n.op, IdentityOp):
                dot.node(name=str(id(n)) + n.op.opName, label=n.op.opName, shape='ellipse')
                dot.edge(str(id(n)) + n.op.opName, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2.op.opName)
        
        return dot
    
    def __repr__(self):
        return f"{self.name}: value={self.val:.4f}, grad={self.grad:.4f}"

    @classmethod
    def cleanup(cls) -> None:
        """Clean up all variable names and reset the counter.
        
        This should be called when you want to start fresh with a new graph.
        """
        cls.varNames.clear()
        cls.varNum = 0


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


if __name__ == "__main__":
    a = Value(2, name='a')
    b = Value(3, name='b')
    c = a - b
    d = a * b
    e = c / d
    f = e ** 2

    f.forward()
    f.backward()

    print(f)
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)

    dot = Value.draw_dot(f, format='png', rankdir='LR')
    dot.render('graph', view=True)