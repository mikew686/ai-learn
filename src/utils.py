import json
import os
import textwrap
from openai import OpenAI


def create_client(openrouter_key: str | None = None) -> OpenAI:
    """
    Create and configure an OpenAI client.

    If OPENROUTER_API_KEY environment variable is set, uses OpenRouter API.
    Otherwise, uses OpenAI API directly.

    Args:
        openrouter_key: OpenRouter API key if available, None otherwise.
                       If None, will check OPENROUTER_API_KEY environment variable.

    Returns:
        Configured OpenAI client instance
    """
    if openrouter_key is None:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_key:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
    else:
        return OpenAI()


def format_response(content: str) -> str:
    """
    Format response content with ">> " prefix on each line for multiline output.

    Args:
        content: Response content string (may contain newlines)

    Returns:
        Formatted string with each line prefixed with ">> "
    """
    lines = content.strip().split("\n")
    return "\n".join(f">> {line}" if line.strip() else ">>" for line in lines)


def print_indented(
    label: str,
    content: str,
    indent: int = 2,
    width: int = 120,
    max_length: int | None = None,
):
    """
    Print a label followed by indented multiline content with text wrapping.

    Args:
        label: Label text (e.g., "Phrase:", "Language Code:")
        content: Content to print (may contain newlines)
        indent: Number of spaces for indentation (default: 2)
        width: Maximum width for wrapped text (default: 120)
        max_length: If set, truncate content to this many characters (default: None)
    """
    if max_length is not None and len(content) > max_length:
        content = content[:max_length] + "\n... [truncated]"
    print(f"{label}:")
    indent_str = " " * indent
    # Calculate available width after indentation
    available_width = width - indent

    # Split by existing newlines first, then wrap each section
    sections = content.split("\n")
    for section in sections:
        if section.strip():
            # Wrap text to fit within available width
            wrapped_lines = textwrap.wrap(section, width=available_width)
            for line in wrapped_lines:
                print(f"{indent_str}{line}")
        else:
            # Preserve empty lines
            print()


def print_response_timing(elapsed_time: float, label: str = ""):
    """
    Print response timing information.

    Args:
        elapsed_time: Elapsed time in seconds
        label: Optional label to prefix the output
    """
    prefix = f"{label} - " if label else ""
    print(f"{prefix}Response Time: {elapsed_time:.3f}s")


def print_token_usage(response, label: str = ""):
    """
    Print token usage information for a response.

    Args:
        response: OpenAI API response object
        label: Optional label to prefix the output
    """
    prefix = f"{label} - " if label else ""
    print(
        f"{prefix}Token Usage - Prompt: {response.usage.prompt_tokens}, "
        f"Completion: {response.usage.completion_tokens}, "
        f"Total: {response.usage.total_tokens}"
    )


def _message_summary(messages: list, max_length: int = 200) -> str:
    """Build a short summary of the messages list for logging."""
    if not messages:
        return "(no messages)"
    roles = []
    for m in messages:
        if isinstance(m, dict):
            roles.append(m.get("role", "?"))
        elif hasattr(m, "role"):
            roles.append(getattr(m.role, "value", m.role) if hasattr(m.role, "value") else str(m.role))
        else:
            roles.append("?")
    summary = f"{len(messages)} message(s): {', '.join(roles)}"
    # Append last user content if present and short enough
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else (getattr(m, "role", None) and (getattr(m.role, "value", None) or str(m.role)))
        if role != "user":
            continue
        content = ""
        if isinstance(m, dict):
            content = m.get("content") or m.get("text") or ""
        elif hasattr(m, "content"):
            content = m.content or ""
        if isinstance(content, str) and content.strip():
            if len(content) <= max_length:
                summary += f" | last user: {content.strip()}"
            else:
                summary += f" | last user: {content.strip()[:max_length]}..."
            break
    return summary


def _response_type(response) -> str:
    """Return explicit response type: 'text', 'schema', 'tool_calls', or combined (e.g. 'text, tool_calls')."""
    try:
        msg = response.choices[0].message
    except (IndexError, AttributeError):
        return "unknown"
    types = []
    if getattr(msg, "content", None) and str(msg.content).strip():
        types.append("text")
    if getattr(msg, "parsed", None) is not None:
        types.append("schema")
    if getattr(msg, "tool_calls", None) and msg.tool_calls:
        types.append("tool_calls")
    return ", ".join(types) if types else "empty"


def _format_tool_calls(tool_calls: list, args_max_length: int = 300) -> str:
    """Format tool call names and arguments for logging."""
    lines = [f"{len(tool_calls)} tool call(s):"]
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", "?") or "?"
        raw_args = getattr(getattr(tc, "function", None), "arguments", None) or "{}"
        try:
            args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            args_str = json.dumps(args_obj, indent=2)
        except (json.JSONDecodeError, TypeError):
            args_str = str(raw_args)
        if len(args_str) > args_max_length:
            args_str = args_str[:args_max_length] + "\n... [truncated]"
        lines.append(f"  {name}:")
        for arg_line in args_str.split("\n"):
            lines.append(f"    {arg_line}")
    return "\n".join(lines)


def _response_summary(response, max_length: int = 400) -> str:
    """Build a short summary of the API response for logging."""
    try:
        msg = response.choices[0].message
    except (IndexError, AttributeError):
        return "(no choices)"
    parts = []
    if getattr(msg, "content", None) and str(msg.content).strip():
        content = str(msg.content).strip()
        if len(content) <= max_length:
            parts.append(content)
        else:
            parts.append(content[:max_length] + "...")
    if getattr(msg, "tool_calls", None) and msg.tool_calls:
        parts.append(_format_tool_calls(msg.tool_calls))
    if getattr(msg, "parsed", None) is not None:
        parts.append("(structured/parsed)")
    return "\n".join(parts) if parts else "(empty)"


def _print_indented_block(
    indent_str: str,
    content: str,
    width: int = 120,
    extra_indent: int = 2,
) -> None:
    """Print content on the next line(s) in an indented, wrapped block."""
    block_indent = indent_str + " " * extra_indent
    available_width = max(20, width - len(block_indent))
    sections = content.split("\n")
    for section in sections:
        if section.strip():
            for line in textwrap.wrap(section, width=available_width):
                print(f"{block_indent}{line}")
        else:
            print()


class OpenAILog:
    """
    Collects and formats OpenAI request/response pairs for test and example programs.

    Callers register each API call with register(); each entry is printed in a
    consistent format (request type, message summary, response summary, token
    use with running total, optional timing). At the end, call print_summary()
    for aggregate statistics.
    """

    def __init__(
        self,
        indent: int = 2,
        width: int = 120,
        max_message_length: int = 200,
        max_response_length: int = 400,
    ):
        self.indent = indent
        self.width = width
        self.max_message_length = max_message_length
        self.max_response_length = max_response_length
        self._entries: list[dict] = []
        self._running_prompt_tokens = 0
        self._running_completion_tokens = 0
        self._total_elapsed = 0.0

    def register(
        self,
        request_type: str,
        messages: list,
        response,
        *,
        elapsed_time: float | None = None,
        label: str | None = None,
    ) -> None:
        """
        Register one OpenAI request/response and print a formatted log entry.

        Args:
            request_type: e.g. "chat.completions.create", "beta.chat.completions.parse"
            messages: The messages list sent in the request (for summary).
            response: The API response object (must have .usage and .choices).
            elapsed_time: Optional elapsed time in seconds for this call.
            label: Optional short label (e.g. "Initial (with tools)", "Final").
        """
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        self._running_prompt_tokens += prompt_tokens
        self._running_completion_tokens += completion_tokens
        if elapsed_time is not None:
            self._total_elapsed += elapsed_time

        entry = {
            "request_type": request_type,
            "messages": messages,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time,
            "label": label,
        }
        self._entries.append(entry)

        # Format and print this entry
        indent_str = " " * self.indent
        req_label = f" [{label}]" if label else ""
        print(f"\n{indent_str}--- Request #{len(self._entries)} ---")
        print(f"{indent_str}Request type: {request_type}{req_label}")
        print(f"{indent_str}Response type: {_response_type(response)}")
        msg_summary = _message_summary(messages, self.max_message_length)
        print(f"{indent_str}Message:")
        _print_indented_block(indent_str, msg_summary, self.width)
        resp_summary = _response_summary(response, self.max_response_length)
        print(f"{indent_str}Response:")
        _print_indented_block(indent_str, resp_summary, self.width)
        print(
            f"{indent_str}Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
            f"Total: {total_tokens} (running: {self._running_prompt_tokens} in, "
            f"{self._running_completion_tokens} out)"
        )
        if elapsed_time is not None:
            print(f"{indent_str}Response time: {elapsed_time:.3f}s")

    def print_summary(self) -> None:
        """Print a summary of all registered transactions."""
        n = len(self._entries)
        if n == 0:
            return
        total_tokens = self._running_prompt_tokens + self._running_completion_tokens
        print("\n" + "=" * 60)
        print("OpenAI API summary")
        print("=" * 60)
        print(f"  Requests sent: {n}")
        print(f"  Responses received: {n}")
        print(f"  Total prompt tokens: {self._running_prompt_tokens:,}")
        print(f"  Total completion tokens: {self._running_completion_tokens:,}")
        print(f"  Total tokens: {total_tokens:,}")
        if n > 0:
            print(f"  Average tokens per request: {total_tokens / n:.1f}")
        if self._total_elapsed > 0:
            print(f"  Total time: {self._total_elapsed:.3f}s")
            print(f"  Average time per request: {self._total_elapsed / n:.3f}s")
        print("=" * 60)
