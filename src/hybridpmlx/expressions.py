from __future__ import annotations

import ast
from typing import Any, Mapping


class SafeExpression:
    """Compile a small, differentiable expression language without eval()."""

    FUNCTIONS = {"exp", "log", "sqrt", "sigmoid", "tanh", "abs", "pow", "clamp"}
    BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    UNARY = (ast.UAdd, ast.USub)

    def __init__(self, expression: str):
        self.expression = expression.strip()
        self.tree = ast.parse(self.expression, mode="eval")
        self.names: set[str] = set()
        self._validate(self.tree)

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            self._validate(node.body)
        elif isinstance(node, ast.Name):
            self.names.add(node.id)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, self.BINARY):
            self._validate(node.left); self._validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, self.UNARY):
            self._validate(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in self.FUNCTIONS:
                raise ValueError(f"Unsupported function: {node.func.id}")
            for arg in node.args:
                self._validate(arg)
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
        else:
            raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    def __call__(self, namespace: Mapping[str, Any], ops: Mapping[str, Any]) -> Any:
        return self._run(self.tree.body, namespace, ops)

    def _run(self, node: ast.AST, ns: Mapping[str, Any], ops: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Name):
            if node.id not in ns:
                raise KeyError(f"Unknown symbol in expression: {node.id}")
            return ns[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = self._run(node.operand, ns, ops)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = self._run(node.left, ns, ops), self._run(node.right, ns, ops)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
        if isinstance(node, ast.Call):
            args = [self._run(arg, ns, ops) for arg in node.args]
            return ops[node.func.id](*args)
        raise RuntimeError("Expression validation and execution diverged")

