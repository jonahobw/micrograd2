"""Visualization utilities for micrograd2.

This module contains functions for visualizing the computational graph.
"""
from typing import Set, Tuple

from graphviz import Digraph

from ..operations.arithmetic import IdentityOp
from ..value import Value


def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    """Trace the computational graph from a root value.
    
    Args:
        root: The root value of the computational graph
        
    Returns:
        Tuple containing:
            - Set of all nodes in the graph
            - Set of edges in the graph as (child, parent) pairs
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root: Value, format: str = 'png', rankdir: str = 'LR') -> Digraph:
    """Create a visualization of the computational graph.
    
    Args:
        root: The root value of the computational graph
        format: Output format (png, svg, etc.)
        rankdir: Graph direction (LR for left-to-right, TB for top-to-bottom)
        
    Returns:
        A graphviz Digraph object
        
    Raises:
        AssertionError: If rankdir is not 'LR' or 'TB'
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