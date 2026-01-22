import argparse
import os
import time
from openai import OpenAI
from prompts import quebec_french_translate
from utils import (
    create_client,
    format_response,
    print_response_timing,
    print_token_usage,
)


def run_stateless_example(client: OpenAI, model: str):
    """
    Run Example A: Stateless approach where system prompt is repeated in each user message.

    Args:
        client: OpenAI client instance
        model: Model name to use for completions
    """
    print("=" * 60)
    print("EXAMPLE A: Stateless User-only Prompts")
    print("=" * 60)

    # Call 1 - Stateless (instructions repeated in user message)
    message1 = "I'm running late to the office."
    start1 = time.time()
    r1 = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{quebec_french_translate}\n\nTranslate: '{message1}'",
            }
        ],
    )
    elapsed1 = time.time() - start1

    # Call 2 - Stateless (instructions repeated again)
    message2 = "Let's grab a drink after work."
    start2 = time.time()
    r2 = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{quebec_french_translate}\n\nTranslate: '{message2}'",
            }
        ],
    )
    elapsed2 = time.time() - start2

    print(f"\n{message1}")
    print(format_response(r1.choices[0].message.content))
    print_token_usage(r1)
    print_response_timing(elapsed1)

    print(f"\n{message2}")
    print(format_response(r2.choices[0].message.content))
    print_token_usage(r2)
    print_response_timing(elapsed2)

    total_tokens = r1.usage.total_tokens + r2.usage.total_tokens
    total_time = elapsed1 + elapsed2
    print(f"\nTotal Tokens (Example A): {total_tokens}")
    print_response_timing(total_time, "Total Response Time")


def run_stateful_example(client: OpenAI, model: str):
    """
    Run Example B: Stateful approach where system prompt is set once and reused.

    Args:
        client: OpenAI client instance
        model: Model name to use for completions
    """
    print("\n" + "=" * 60)
    print("EXAMPLE B: Stateful System Prompt")
    print("=" * 60)

    message1 = "I'm running late to the office."
    message2 = "Let's grab a drink after work."

    messages = [
        {
            "role": "system",
            "content": quebec_french_translate
            + "\n\nIf multiple user messages are provided, translate ALL of them, one per line.",
        },
        {"role": "user", "content": message1},
        {"role": "user", "content": message2},
    ]

    # Single stateful session with system + 2 user messages
    start = time.time()
    response = client.chat.completions.create(model=model, messages=messages)
    elapsed = time.time() - start

    print(f"\n{message1}")
    print(f"{message2}")
    print(format_response(response.choices[0].message.content))
    print_token_usage(response)
    print_response_timing(elapsed)

    print(f"\nTotal Tokens (Example B): {response.usage.total_tokens}")
    print_response_timing(elapsed, "Total Response Time")


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

    run_stateless_example(client, model)
    run_stateful_example(client, model)


if __name__ == "__main__":
    main()
