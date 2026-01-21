import argparse
import json
import os
from openai import OpenAI

# Shared system prompt - same for both examples
system_prompt = """You are an expert translator specializing in Québécois French from Montréal.

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


def create_client(openrouter_key: str | None) -> OpenAI:
    """
    Create and configure an OpenAI client.
    
    Args:
        openrouter_key: OpenRouter API key if available, None otherwise
        
    Returns:
        Configured OpenAI client instance
    """
    if openrouter_key:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
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
    lines = content.strip().split('\n')
    return '\n'.join(f">> {line}" if line.strip() else ">>" for line in lines)


def print_token_usage(response, label: str = ""):
    """
    Print token usage information for a response.
    
    Args:
        response: OpenAI API response object
        label: Optional label to prefix the output
    """
    prefix = f"{label} - " if label else ""
    print(f"{prefix}Token Usage - Prompt: {response.usage.prompt_tokens}, "
          f"Completion: {response.usage.completion_tokens}, "
          f"Total: {response.usage.total_tokens}")


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
    r1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"{system_prompt}\n\nTranslate: '{message1}'"}]
    )

    # Call 2 - Stateless (instructions repeated again)
    message2 = "Let's grab a drink after work."
    r2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"{system_prompt}\n\nTranslate: '{message2}'"}]
    )

    print(f"\n{message1}")
    print(format_response(r1.choices[0].message.content))
    print_token_usage(r1)

    print(f"\n{message2}")
    print(format_response(r2.choices[0].message.content))
    print_token_usage(r2)

    total_tokens = r1.usage.total_tokens + r2.usage.total_tokens
    print(f"\nTotal Tokens (Example A): {total_tokens}")


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
        {"role": "system", "content": system_prompt + "\n\nIf multiple user messages are provided, translate ALL of them, one per line."},
        {"role": "user", "content": message1},
        {"role": "user", "content": message2},
    ]

    # Single stateful session with system + 2 user messages
    response = client.chat.completions.create(model=model, messages=messages)

    print(f"\n{message1}")
    print(f"{message2}")
    print(format_response(response.choices[0].message.content))
    print_token_usage(response)
    
    print(f"\nTotal Tokens (Example B): {response.usage.total_tokens}")


if __name__ == "__main__":
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
    parser = argparse.ArgumentParser(description="Demonstrate stateless vs stateful system prompts")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides MODEL env var and default)"
    )
    args = parser.parse_args()
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    client = create_client(openrouter_key)
    
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
