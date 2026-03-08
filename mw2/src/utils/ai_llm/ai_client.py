"""AI client for OpenRouter API with run_turn and tool loop."""

import json
import os
import time

from config import load_config
from openai import OpenAI

from .retrieval_store import RetrievalStore
from .types import ChatRequest


def _parse_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


class OpenRouterKeyMissingError(Exception):
    """Raised when OPENROUTER_API_KEY is not set."""

    def __init__(self) -> None:
        super().__init__(
            "OPENROUTER_API_KEY is required but not set. "
            "Set it in the environment or .env file."
        )


class AIClient:
    """
    OpenAI-compatible client configured for OpenRouter.
    Access the underlying client via .client for API usage.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY. Raises if not set.
            base_url: OpenRouter API base URL. If None, uses OPENROUTER_BASE_URL or default.
            timeout: Request timeout in seconds. If None, uses OPENROUTER_TIMEOUT or SDK default (600s).
            max_retries: Retries for transient errors. If None, uses OPENROUTER_MAX_RETRIES or SDK default (2).
        """
        load_config()
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise OpenRouterKeyMissingError()

        base = base_url or os.getenv("OPENROUTER_BASE_URL", self.OPENROUTER_BASE_URL)
        _timeout = (
            _parse_float(os.getenv("OPENROUTER_TIMEOUT"))
            if timeout is None
            else timeout
        )
        _max_retries = (
            _parse_int(os.getenv("OPENROUTER_MAX_RETRIES"))
            if max_retries is None
            else max_retries
        )

        kwargs: dict = {"base_url": base, "api_key": key}
        if _timeout is not None:
            kwargs["timeout"] = _timeout
        if _max_retries is not None:
            kwargs["max_retries"] = _max_retries

        self._client = OpenAI(**kwargs)
        response = self._client.models.list()
        self._models = {m.id: m.model_dump() for m in response.data}

    def models(self) -> dict[str, dict]:
        """Return dict of model id -> model data (loaded at init)."""
        return self._models

    @property
    def client(self) -> OpenAI:
        """Raw OpenAI client for direct API usage (e.g. chat.completions.create)."""
        return self._client

    def run_turn(self, req: ChatRequest) -> tuple[object, object | None]:
        """
        Run one turn: RetrievalStore pre-call (if provided), then _run_loop.
        Returns (result, raw_response). result is parsed when response_format is set, else the raw response.
        raw_response is None on retrieval cache hit.
        """
        query: str | None = None
        embedding: list[float] = []

        if req.retrieval_store is not None:
            query = _last_user_content(req.messages)
            if query is not None:
                parsed, _examples, emb = req.retrieval_store.get(query)
                embedding = emb
                if parsed is not None:
                    return parsed, None

        result, raw_response = self._run_loop(req)

        if (
            req.retrieval_store is not None
            and query is not None
            and embedding
            and raw_response is not None
        ):
            req.retrieval_store.store(query, embedding, raw_response)

        return result, raw_response

    def _run_loop(self, req: ChatRequest) -> tuple[object, object]:
        """
        Internal tool loop: call API, handle tool_calls, repeat until done.
        When response_format is set, uses beta.chat.completions.parse for structured output after the loop.
        Returns (result, raw_response). result is parsed when response_format is set, else the raw response.
        """
        extra_headers: dict[str, str] = {}
        if req.idempotency_key:
            extra_headers["Idempotency-Key"] = req.idempotency_key

        followup_headers = {
            k: v for k, v in extra_headers.items() if k != "Idempotency-Key"
        }

        tools = req.tool_provider.get_specs() if req.tool_provider else None
        has_tools = tools is not None
        has_response_format = req.response_format is not None
        create_params = req.get_create_params(exclude_response_format=has_tools)

        messages = list(req.messages)

        def _do_create(kwargs: dict) -> object:
            h = extra_headers if kwargs.get("extra_headers") else followup_headers
            kwargs["extra_headers"] = h if h else None
            return self._client.chat.completions.create(**kwargs)

        def _do_parse(kwargs: dict) -> object:
            h = extra_headers if kwargs.get("extra_headers") else followup_headers
            kwargs["extra_headers"] = h if h else None
            return self._client.beta.chat.completions.parse(**kwargs)

        if has_response_format and not has_tools:
            parse_kwargs_direct: dict = {
                "model": req.model,
                "messages": messages,
                "response_format": req.response_format,
                **req.get_create_params(),
            }
            parse_kwargs_direct["extra_headers"] = (
                extra_headers if extra_headers else None
            )
            start_ts = time.time()
            response = _do_parse(parse_kwargs_direct)
            end_ts = time.time()
            if req.log_chat_completion:
                req.log_chat_completion(start_ts, end_ts, response)
            return response.choices[0].message.parsed, response

        create_kwargs: dict = {
            "model": req.model,
            "messages": messages,
            **create_params,
        }
        if has_tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        create_kwargs["extra_headers"] = extra_headers if extra_headers else None

        start_ts = time.time()
        response = _do_create(create_kwargs)
        end_ts = time.time()

        if req.log_chat_completion:
            req.log_chat_completion(start_ts, end_ts, response)

        msg = response.choices[0].message
        tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None

        while tool_calls and req.tool_provider is not None:
            messages.append(_message_to_dict(msg))

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    result = {"error": "Failed to parse tool arguments as JSON"}
                else:
                    result = req.tool_provider.execute(name, args)

                if isinstance(result, (dict, list)):
                    result_content = json.dumps(result)
                else:
                    result_content = str(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    }
                )

            create_kwargs_followup: dict = {
                "model": req.model,
                "messages": messages,
                "extra_headers": followup_headers if followup_headers else None,
                **create_params,
            }
            if has_tools:
                create_kwargs_followup["tools"] = tools
                create_kwargs_followup["tool_choice"] = "auto"

            start_ts = time.time()
            response = _do_create(create_kwargs_followup)
            end_ts = time.time()

            if req.log_chat_completion:
                req.log_chat_completion(start_ts, end_ts, response)

            msg = response.choices[0].message
            tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None

        if has_response_format:
            if has_tools:
                messages.append(_message_to_dict(msg))
            parse_kwargs: dict = {
                "model": req.model,
                "messages": messages,
                "response_format": req.response_format,
                **create_params,
            }
            start_ts = time.time()
            response = _do_parse(parse_kwargs)
            end_ts = time.time()
            if req.log_chat_completion:
                req.log_chat_completion(start_ts, end_ts, response)
            parsed = response.choices[0].message.parsed
            return parsed, response

        msg = response.choices[0].message
        return getattr(msg, "content", None), response


def _last_user_content(messages: list[dict]) -> str | None:
    """Return content of last user message, or None."""
    for m in reversed(messages):
        if m.get("role") == "user" and "content" in m:
            c = m["content"]
            return c if isinstance(c, str) else str(c)
    return None


def _message_to_dict(msg: object) -> dict:
    """Convert OpenAI message object to dict for API."""
    d: dict = {"role": getattr(msg, "role", "assistant")}
    content = getattr(msg, "content", None)
    if content is not None:
        d["content"] = content
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls is not None:
        tc_list = []
        for tc in tool_calls:
            tc_dict = {
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": (
                        getattr(tc.function, "name", "")
                        if hasattr(tc, "function")
                        else ""
                    ),
                    "arguments": (
                        getattr(tc.function, "arguments", "{}")
                        if hasattr(tc, "function")
                        else "{}"
                    ),
                },
            }
            tc_list.append(tc_dict)
        d["tool_calls"] = tc_list
    return d
