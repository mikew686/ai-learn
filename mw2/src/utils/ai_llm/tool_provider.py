"""Tool provider for function calling."""

from typing import Any, Callable


class DictToolProvider:
    """Convenience subclass for tools + functions dict."""

    def __init__(
        self,
        tools: list[dict[str, Any]] | None = None,
        functions: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """
        Args:
            tools: OpenAI-format tool specs (list of function definitions).
            functions: Map of tool name -> callable; called with parsed JSON args.
        """
        self._specs = list(tools) if tools else []
        self._functions = dict(functions) if functions else {}

    def add_tool(self, spec: dict[str, Any], func: Callable[..., Any]) -> None:
        """Add a tool spec and its handler. Name is taken from spec['function']['name']."""
        self._specs.append(spec)
        name = spec.get("function", {}).get("name", "")
        if name:
            self._functions[name] = func

    def get_specs(self) -> list[dict[str, Any]]:
        return self._specs

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._functions[name](**arguments)
