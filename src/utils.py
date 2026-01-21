import os
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
