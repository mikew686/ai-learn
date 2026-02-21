"""
Pattern 2: Prompt Engineering

Use case: Translation with system role optimization — compare token use and
behavior when instructions live in the system message vs repeated in each user message.

Patterns shown:
  - **Prompt engineering (primary)**: System vs user prompts; role-setting and
    constraint injection in the system message.
  - **Stateless vs stateful**: Example A repeats instructions per user message;
    Example B sets the system prompt once and reuses it across two user messages.
  - **Token efficiency**: Same two phrases translated both ways; total tokens
    and timing illustrate the cost of repeating the system prompt.

Details:
  - Two separate runs (Example A stateless, Example B stateful), each with its
    own OpenAILog. Output: translations plus per-request and summary logs.

Usage:
    python -m src.system_prompt_example [--model MODEL] [--temperature T] [--max-tokens N]
"""

import argparse
import os
from openai import OpenAI
from utils import (
    create_client,
    format_response,
    OpenAILog,
)

# Prompt for Québécois French translation
QUEBEC_FRENCH_TRANSLATE = """You are an expert translator specializing in Québécois French from Montréal.

Your task is to translate English text into authentic, informal Québécois French.

Translation Requirements:
1. Use tutoiement (informal "you") form exclusively
2. Always use contractions: 'j'suis' (never 'je suis'), 't'es' (never 'tu es'), 'c'est' (never 'ce est')
3. Use Montréal-specific slang and expressions:
   - 'boulot' for work/job
   - 'bureau' for office
   - 'frette' or 'bière' for drink
4. Prefer colloquial expressions over formal language
5. Avoid literal translations and Parisian French phrasing
6. For ambiguous phrases, infer meaning as a native Montréal speaker would

Output Format:
- Output only the translation(s), no explanations, metadata, or comments
- Maintain consistent tone, slang level, and register across all translations
- Sound authentic to a native Québécois speaker from Montréal"""


def run_stateless_example(
    client: OpenAI,
    model: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Run Example A: Stateless approach where system prompt is repeated in each user message.
    Uses a separate OpenAILog for this message train.
    """
    print("=" * 60)
    print("EXAMPLE A: Stateless User-only Prompts")
    print("=" * 60)

    log = OpenAILog()
    # Call 1 - Stateless (instructions repeated in user message)
    message1 = "I'm running late to the office."
    r1_messages = [
        {
            "role": "user",
            "content": f"{QUEBEC_FRENCH_TRANSLATE}\n\nTranslate: '{message1}'",
        }
    ]
    r1_kwargs = {"model": model, "messages": r1_messages}
    if temperature is not None:
        r1_kwargs["temperature"] = temperature
    if max_tokens is not None:
        r1_kwargs["max_tokens"] = max_tokens
    log.start_call()
    r1 = client.chat.completions.create(**r1_kwargs)
    log.register(
        "chat.completions.create",
        r1_messages,
        r1,
        label="Stateless 1",
    )

    # Call 2 - Stateless (instructions repeated again)
    message2 = "Let's grab a drink after work."
    r2_messages = [
        {
            "role": "user",
            "content": f"{QUEBEC_FRENCH_TRANSLATE}\n\nTranslate: '{message2}'",
        }
    ]
    r2_kwargs = {"model": model, "messages": r2_messages}
    if temperature is not None:
        r2_kwargs["temperature"] = temperature
    if max_tokens is not None:
        r2_kwargs["max_tokens"] = max_tokens
    log.start_call()
    r2 = client.chat.completions.create(**r2_kwargs)
    log.register(
        "chat.completions.create",
        r2_messages,
        r2,
        label="Stateless 2",
    )

    print(f"\n{message1}")
    print(format_response(r1.choices[0].message.content))

    print(f"\n{message2}")
    print(format_response(r2.choices[0].message.content))

    log.print_summary()


def run_stateful_example(
    client: OpenAI,
    model: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Run Example B: Stateful approach where system prompt is set once and reused.
    Uses a separate OpenAILog for this message train.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE B: Stateful System Prompt")
    print("=" * 60)

    log = OpenAILog()
    message1 = "I'm running late to the office."
    message2 = "Let's grab a drink after work."

    messages = [
        {
            "role": "system",
            "content": QUEBEC_FRENCH_TRANSLATE
            + "\n\nIf multiple user messages are provided, translate ALL of them, one per line.",
        },
        {"role": "user", "content": message1},
        {"role": "user", "content": message2},
    ]

    # Single stateful session with system + 2 user messages
    response_kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        response_kwargs["temperature"] = temperature
    if max_tokens is not None:
        response_kwargs["max_tokens"] = max_tokens
    log.start_call()
    response = client.chat.completions.create(**response_kwargs)
    log.register(
        "chat.completions.create",
        messages,
        response,
        label="Stateful",
    )

    print(f"\n{message1}")
    print(f"{message2}")
    print(format_response(response.choices[0].message.content))

    log.print_summary()


def main():
    """Demonstrate stateless vs stateful prompts using system roles."""
    # Environment variable detection and client configuration:
    # - If OPENROUTER_API_KEY is set, uses OpenRouter API (base_url: https://openrouter.ai/api/v1)
    # - Otherwise, uses OpenAI API directly (default behavior)
    #
    # Model configuration (priority order):
    # 1. Command line argument (--model)
    # 2. MODEL environment variable
    # 3. Default: "openai/gpt-4.1-mini" for OpenRouter, "gpt-4.1-mini" for OpenAI
    #    (gpt-4.1-mini is cost-effective and well-suited for translation tasks)
    #
    # Demonstrates the difference between stateless and stateful prompts using system roles:
    # - Example A: Stateless approach (system prompt repeated in each user message)
    # - Example B: Stateful approach (system prompt set once, reused across messages)
    parser = argparse.ArgumentParser(
        description="Demonstrate stateless vs stateful system prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides MODEL env var and default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omit to use API default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per response (omit to use API default)",
    )
    args = parser.parse_args()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    client = create_client()

    # Determine default model based on API provider
    if openrouter_key:
        default_model = "openai/gpt-4.1-mini"
        print("Using OpenRouter API\n")
    else:
        default_model = "gpt-4.1-mini"
        print("Using OpenAI API\n")

    # Model selection: command line > environment variable > default
    model = args.model or os.getenv("MODEL", default_model)
    print(f"Using model: {model}\n")

    run_stateless_example(
        client,
        model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    run_stateful_example(
        client,
        model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
