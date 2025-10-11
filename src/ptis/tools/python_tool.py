"""Restricted Python execution tool."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict


class PythonToolError(Exception):
    """Raised when execution fails or violates sandbox constraints."""


_ALLOWED_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
}

_ALLOWED_GLOBALS = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_ALLOWED_GLOBALS.update(_ALLOWED_BUILTINS)


@dataclass(slots=True)
class PythonTool:
    """Executes deterministic Python expressions in a sandbox."""

    def run(self, code: str) -> str:
        try:
            parsed = ast.parse(code, mode="exec")
        except SyntaxError as exc:  # pragma: no cover - direct mapping to exception
            raise PythonToolError(f"语法错误: {exc}") from exc
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Call)):
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_BUILTINS:
                        raise PythonToolError("禁止调用未授权函数")
                else:
                    raise PythonToolError("禁止使用导入或全局语句")
        local_env: Dict[str, Any] = {}
        globals_env: Dict[str, Any] = {"__builtins__": _ALLOWED_BUILTINS}
        globals_env.update(_ALLOWED_GLOBALS)
        try:
            exec(compile(parsed, "<python-tool>", "exec"), globals_env, local_env)
        except Exception as exc:  # pragma: no cover - depends on user code
            raise PythonToolError(str(exc)) from exc
        result = local_env.get("result")
        if result is None:
            expr = ast.Expression(parsed.body[-1].value) if parsed.body and isinstance(parsed.body[-1], ast.Expr) else None
            if expr is not None:
                result = eval(compile(expr, "<python-tool>", "eval"), globals_env, local_env)
        return str(result)
