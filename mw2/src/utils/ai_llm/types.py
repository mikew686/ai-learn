"""Request and response types for the AI LLM module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .tool_provider import DictToolProvider


@dataclass
class ChatRequest:
    """
    Request container for chat completions.
    Convention from co-hermes: all params in one object.
    model is required; all other fields are optional.
    """

    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_provider: "DictToolProvider | None" = None
    retrieval_store: "RetrievalStore | None" = None
    log_chat_completion: Callable[[float, float, object], None] | None = None
    idempotency_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    response_format: type | None = None

    def __init__(
        self,
        model: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        tool_provider: "DictToolProvider | None" = None,
        retrieval_store: "RetrievalStore | None" = None,
        log_chat_completion: Callable[[float, float, object], None] | None = None,
        idempotency_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        response_format: type | None = None,
    ) -> None:
        """
        Args:
            model: Model ID (required).
            messages: Chat messages; defaults to [].
            tool_provider: Optional tool provider for function calling.
            retrieval_store: Optional retrieval store for RAG.
            log_chat_completion: Called with (start_ts, end_ts, response) after each LLM call.
            idempotency_key: Used only on first LLM call (stripped for follow-ups).
            temperature: Sampling temperature for create.
            max_tokens: Max tokens for create.
            top_p: Top-p (nucleus) sampling for create.
            response_format: Pydantic model or schema for structured output.
        """
        self.model = model
        self.messages = messages if messages is not None else []
        self.tool_provider = tool_provider
        self.retrieval_store = retrieval_store
        self.log_chat_completion = log_chat_completion
        self.idempotency_key = idempotency_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.response_format = response_format

    def get_create_params(
        self, *, exclude_response_format: bool = False
    ) -> dict[str, Any]:
        """Build kwargs for chat.completions.create. Set exclude_response_format for tool loop."""
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.response_format is not None and not exclude_response_format:
            params["response_format"] = self.response_format
        return params

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Append a message. role: system|user|assistant|tool. For tool, pass tool_call_id."""
        self.messages.append({"role": role, "content": content, **kwargs})
