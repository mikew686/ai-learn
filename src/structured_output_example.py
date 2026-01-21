import argparse
import os
import time
from pydantic import BaseModel, Field
from prompts import language_assessment_prompt
from utils import (
    create_client,
    print_indented,
    print_response_timing,
    print_token_usage,
)


class LanguageAssessment(BaseModel):
    """Assessment of a language phrase with region-specific details."""

    language_code: str = Field(
        description="ISO 639-1 language code (e.g., 'en', 'fr', 'es')"
    )
    region_code: str = Field(
        description="ISO 3166-1 alpha-2 region code (e.g., 'CA', 'US', 'MX', 'FR')"
    )
    description: str = Field(
        description=(
            "English description of the language variant detected, "
            "including dialect characteristics and linguistic features"
        )
    )
    variant: str = Field(
        description=(
            "Specific regional or dialectal variant name (e.g., "
            "'Quebec French', 'Latin American Spanish', "
            "'Canadian English', 'Parisian French', "
            "'Southern US English')"
        )
    )


def assess_lang(phrase: str, client=None, model: str = None):
    """
    Assess the language, variant, and region of a given phrase using structured output.

    Args:
        phrase: The text phrase to analyze
        client: OpenAI client instance (creates one if not provided)
        model: Model name to use (uses default if not provided)

    Returns:
        Tuple of (LanguageAssessment object, response object, elapsed_time) for access to token usage and timing
    """
    if client is None:
        client = create_client()

    if model is None:
        # Determine default model based on API provider
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            model = "openai/gpt-4o-mini"
        else:
            model = "gpt-4o-mini"

    prompt = language_assessment_prompt.format(phrase=phrase)

    start_time = time.time()
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=LanguageAssessment,
    )
    elapsed_time = time.time() - start_time

    return response.choices[0].message.parsed, response, elapsed_time


def main():
    """Run examples of language assessment using structured output."""
    parser = argparse.ArgumentParser(
        description="Assess language variants using structured output"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides MODEL env var and default)",
    )
    args = parser.parse_args()

    client = create_client()

    # Determine default model based on API provider
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        default_model = "openai/gpt-4.1-mini"
        print("Using OpenRouter API\n")
    else:
        default_model = "gpt-4.1-mini"
        print("Using OpenAI API\n")

    # Model selection: command line > environment variable > default
    model = args.model or os.getenv("MODEL", default_model)
    print(f"Using model: {model}\n")

    # Example phrases demonstrating different language variants
    examples = [
        "I'm going to grab some poutine and head to the Tim Hortons.",
        "J'vais aller chercher une poutine pis après ça, on va au dépanneur.",
        "Je vais acheter une baguette et ensuite nous irons au café.",
        "Voy a comprar unas tortillas y luego vamos al mercado.",
        "I'm fixin' to go to the store, y'all want anything?",
        "Ich geh' mal zum Intershop, dann kaufen wir uns was.",
        "I'll be going to the store to buy some groceries.",
        "Stay where you're at, and I'll come where you're to.",
        "I'm after going to the shop, so I'll be back in a minute.",
    ]

    print("=" * 60)
    print("Language Assessment Examples")
    print("=" * 60)

    for i, phrase in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print_indented("Phrase", phrase)

        try:
            assessment, response, elapsed_time = assess_lang(phrase, client, model)
            print_indented("Language Code", assessment.language_code)
            print_indented("Region Code", assessment.region_code)
            print_indented("Variant", assessment.variant)
            print_indented("Description", assessment.description)
            print_token_usage(response)
            print_response_timing(elapsed_time)
        except Exception as e:
            print_indented("Error", str(e))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
