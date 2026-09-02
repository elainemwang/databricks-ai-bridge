"""Agent tools package — plain functions plus their Chat Completions schemas.

No tool framework: a tool is a Python function decorated with ``@tool`` (defined here), which builds
the ``{"type": "function", ...}`` schema the model expects from the function's signature and
docstring. Every module in this package is auto-imported, and every ``@tool`` function is collected
by ``all_tools()`` into ``{name: (func, schema)}`` — the schemas go to the model, the funcs run when
it calls them. Drop a new ``*.py`` here with a ``@tool`` function and it's picked up automatically.

``@tool`` keeps the vanilla spirit: it only introspects the signature (via ``inspect`` + a small type
map) to save you hand-writing JSON Schema. For a parameter shape it can't express, write the schema
dict yourself and pass ``schema=`` to ``@tool``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from typing import Any

# Attribute stamped on a decorated function to carry its Chat Completions tool schema.
_SCHEMA_ATTR = "__mason_tool_schema__"

# Minimal Python-annotation → JSON-Schema type map. Extend it, or pass an explicit `schema=`.
_JSON_TYPES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(func: Callable | None = None, *, schema: dict[str, Any] | None = None):
    """Mark a function as an agent tool, attaching its Chat Completions schema.

    With no arguments the schema is derived from the signature (parameter types → JSON types, all
    required) and the docstring (the tool description). Pass ``schema=`` to supply your own instead.
    """

    def wrap(fn: Callable) -> Callable:
        setattr(fn, _SCHEMA_ATTR, schema or _schema_from_signature(fn))
        return fn

    return wrap(func) if func is not None else wrap


def _schema_from_signature(fn: Callable) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        json_type = _JSON_TYPES.get(param.annotation, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def all_tools() -> dict[str, tuple[Callable, dict[str, Any]]]:
    """Every ``@tool`` function across the package's modules, as ``{name: (func, schema)}``."""
    tools: dict[str, tuple[Callable, dict[str, Any]]] = {}
    for module in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{module.name}")
        for _, obj in inspect.getmembers(mod, callable):
            schema = getattr(obj, _SCHEMA_ATTR, None)
            if schema is not None:
                tools[schema["function"]["name"]] = (obj, schema)
    return tools
