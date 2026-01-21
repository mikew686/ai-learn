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


def print_indented(label: str, content: str, indent: int = 2, width: int = 120):
    """
    Print a label followed by indented multiline content with text wrapping.

    Args:
        label: Label text (e.g., "Phrase:", "Language Code:")
        content: Content to print (may contain newlines)
        indent: Number of spaces for indentation (default: 2)
        width: Maximum width for wrapped text (default: 60)
    """
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
