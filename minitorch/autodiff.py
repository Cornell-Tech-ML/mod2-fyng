from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    left = list(vals)
    left[arg] -= epsilon
    right = list(vals)
    right[arg] += epsilon
    return (f(*right) - f(*left)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Compute the derivative of the variable with respect to the output of the computation graph."""
        ...

    @property
    def unique_id(self) -> int:
        """A unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf node of the computational graph"""
        ...

    def is_constant(self) -> bool:
        """Check if this variable is a constant value"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return list of parent variables"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply chain rule with respect to each variable, and return the result in a list of tuples"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    sorted_nodes = []
    visited = set()

    def visit(node: Variable) -> None:
        if node.unique_id in visited:
            return
        if not node.is_leaf():
            for parent in node.parents:
                visit(parent)
        visited.add(node.unique_id)
        if not node.is_constant():
            sorted_nodes.append(node)

    visit(variable)
    return reversed(sorted_nodes)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    gradient_dict = {}
    sorted_variables = list(topological_sort(variable))

    for var in sorted_variables:
        gradient_dict[var.unique_id] = 0
    gradient_dict[variable.unique_id] = deriv

    for var in sorted_variables:
        if var.is_leaf():
            var.accumulate_derivative(gradient_dict[var.unique_id])
        else:
            for parent, gradient in var.chain_rule(gradient_dict[var.unique_id]):
                gradient_dict[parent.unique_id] += gradient


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve tuple of values saved in `saved_values`."""
        return self.saved_values
