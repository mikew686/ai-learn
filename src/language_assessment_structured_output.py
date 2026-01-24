"""
Pattern 2: Structured Output

This example demonstrates advanced structured output features:
- Pydantic models for type-safe, validated responses
- Enums for constrained values (LanguageCode, RegionCode, ConfidenceLevel)
- Nested models (LinguisticFeature)
- Arrays and lists (linguistic_features, alternative_interpretations)
- Field descriptions that guide model behavior

Use Case: Language assessment with structured validation
"""

import argparse
import os
import time
from collections import Counter
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import (
    create_client,
    print_indented,
    print_response_timing,
)


# Prompt for language assessment
LANGUAGE_ASSESSMENT_PROMPT = """Analyze the following phrase and identify the language,
regional variant, and dialect characteristics.

Identify:
1. The ISO 639-1 language code (e.g., 'en', 'fr', 'es', 'de')
2. The ISO 3166-1 alpha-2 region code (e.g., 'CA', 'US', 'MX', 'FR', 'DE')
3. A clear English description of the language variant, including
   dialect characteristics, linguistic features, and grammatical
   constructions that indicate regional origin
4. The specific regional or dialectal variant name (e.g., Eastern
   Canadian English, Irish English, Quebec French, Latin American
   Spanish, Parisian French, Southern US English, East German)
5. The semantic meaning of the phrase in English (not a translation, but
   an explanation of what the phrase means, including any implied context,
   tone, or cultural nuances)
6. Your confidence level in the assessment (high, medium, low)
7. Specific linguistic features detected (idioms, slang, regionalisms, etc.)
8. Alternative interpretations if confidence is not high

Phrase: "{phrase}"

Pay careful attention to:
- Grammatical constructions and syntax patterns
- Regional vocabulary and expressions
- Preposition usage and phrasal verb patterns
- Contractions and informal speech markers
- Word order and sentence structure
- Idioms, slang, and colloquialisms
- Regional markers and cultural references
- Semantic meaning and implied context

For each linguistic feature, identify:
- The type of feature (idiom, slang, regionalism, colloquialism, etc.)
- The specific word or phrase that demonstrates it
- An explanation of its regional significance

Provide a detailed assessment of the language characteristics,
regional markers, and variant type. Consider all regional variants
globally, not just American English."""


# Advanced structured output: Using enums for constrained values
class LanguageCode(str, Enum):
    """ISO 639-1 language codes."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    POLISH = "pl"
    TURKISH = "tr"
    GREEK = "el"


class RegionCode(str, Enum):
    """ISO 3166-1 alpha-2 region codes."""

    US = "US"
    CA = "CA"
    GB = "GB"
    AU = "AU"
    NZ = "NZ"
    IE = "IE"
    FR = "FR"
    BE = "BE"
    CH = "CH"
    ES = "ES"
    MX = "MX"
    AR = "AR"
    CO = "CO"
    DE = "DE"
    AT = "AT"
    IT = "IT"
    PT = "PT"
    BR = "BR"
    CN = "CN"
    JP = "JP"
    KR = "KR"
    RU = "RU"
    SA = "SA"
    EG = "EG"
    IN = "IN"


class ConfidenceLevel(str, Enum):
    """Confidence level for language assessment."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Advanced structured output: Nested model for linguistic features
class LinguisticFeature(BaseModel):
    """Individual linguistic feature detected in the phrase."""

    feature_type: str = Field(
        description="Type of linguistic feature (e.g., 'idiom', 'slang', 'regionalism', 'colloquialism')"
    )
    example: str = Field(
        description="The specific word or phrase that demonstrates this feature"
    )
    description: str = Field(
        description="Explanation of the feature and its regional significance"
    )


# Advanced structured output: Main model with enums, arrays, and nested models
class LanguageAssessment(BaseModel):
    """Assessment of a language phrase with region-specific details."""

    language_code: LanguageCode = Field(description="ISO 639-1 language code")
    region_code: RegionCode = Field(description="ISO 3166-1 alpha-2 region code")
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
    meaning: str = Field(
        description=(
            "The semantic meaning or interpretation of the phrase in English. "
            "This is not a translation, but rather an explanation of what the phrase means, "
            "including any implied context, tone, or cultural nuances."
        )
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence level in the assessment"
    )
    linguistic_features: List[LinguisticFeature] = Field(
        description="List of specific linguistic features detected in the phrase",
        default_factory=list,
    )
    alternative_interpretations: Optional[List[str]] = Field(
        description="Alternative language or region interpretations if confidence is not high",
        default=None,
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

    prompt = LANGUAGE_ASSESSMENT_PROMPT.format(phrase=phrase)

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
        description="Assess language variants using structured output with advanced features"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides MODEL env var and default)",
    )
    parser.add_argument(
        "--example-phrases",
        action="store_true",
        help="Use the example phrases list instead of the default single phrase",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override with a custom phrase to analyze (takes precedence over --example-phrases)",
    )
    args = parser.parse_args()

    client = create_client()

    # Determine default model based on API provider
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        default_model = "openai/gpt-5.2"
        print("Using OpenRouter API\n")
    else:
        default_model = "gpt-5.2"
        print("Using OpenAI API\n")

    # Model selection: command line > environment variable > default
    model = args.model or os.getenv("MODEL", default_model)
    print(f"Using model: {model}\n")
    print("Advanced Structured Output Features:")
    print("  - Enums for language codes, region codes, and confidence levels")
    print("  - Arrays for linguistic features and alternative interpretations")
    print("  - Nested models for complex structures")
    print()

    # Determine which phrases to use
    # Priority: --prompt > --example-phrases > default single phrase
    if args.prompt:
        examples = [args.prompt]
        print(f"Analyzing custom phrase: {args.prompt}\n")
    elif args.example_phrases:
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
        print("Using example phrases list\n")
    else:
        # Default: single phrase
        examples = ["Ca va, mec?"]
        print("Using default phrase\n")

    print("=" * 60)
    print("Language Assessment Examples")
    print("=" * 60)

    # Track statistics for summary
    assessments = []
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    language_counts = Counter()
    region_counts = Counter()
    variant_counts = Counter()
    confidence_counts = Counter()

    for i, phrase in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print_indented("Phrase", phrase)

        try:
            assessment, response, elapsed_time = assess_lang(phrase, client, model)
            assessments.append(assessment)

            # Collect statistics
            total_tokens += response.usage.total_tokens
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            total_time += elapsed_time
            language_counts[assessment.language_code.value] += 1
            region_counts[assessment.region_code.value] += 1
            variant_counts[assessment.variant] += 1
            confidence_counts[assessment.confidence.value] += 1

            print_indented(
                "Language Code",
                f"{assessment.language_code.value} ({assessment.language_code.name})",
            )
            print_indented(
                "Region Code",
                f"{assessment.region_code.value} ({assessment.region_code.name})",
            )
            print_indented("Variant", assessment.variant)
            print_indented("Confidence", assessment.confidence.value)
            print_indented("Description", assessment.description)

            # Print semantic meaning prominently
            print("\nSemantic Meaning:")
            print_indented("", assessment.meaning)

            # Show linguistic features (array) - always print this section
            print("\nLinguistic Features Detected:")
            if assessment.linguistic_features:
                print(f"  Total: {len(assessment.linguistic_features)}")
                for idx, feature in enumerate(assessment.linguistic_features, 1):
                    print(f"\n  Feature {idx}:")
                    print_indented("    Type", feature.feature_type)
                    print_indented("    Example", feature.example)
                    print_indented("    Description", feature.description)
            else:
                print_indented("  (none detected)", "")

            # Show alternative interpretations (optional array)
            if assessment.alternative_interpretations:
                print("\nAlternative Interpretations:")
                for alt in assessment.alternative_interpretations:
                    print_indented("  -", alt)

            # Print detailed token usage
            print("\nToken Usage Details:")
            usage = response.usage
            print(f"  Prompt tokens: {usage.prompt_tokens:,}")
            print(f"  Completion tokens: {usage.completion_tokens:,}")
            print(f"  Total tokens: {usage.total_tokens:,}")
            # Show additional details if available (for some API versions)
            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                print(f"  Prompt token details: {usage.prompt_tokens_details}")
            if (
                hasattr(usage, "completion_tokens_details")
                and usage.completion_tokens_details
            ):
                print(f"  Completion token details: {usage.completion_tokens_details}")
            print_response_timing(elapsed_time)
        except Exception as e:
            print_indented("Error", str(e))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal Examples Processed: {len(assessments)}")
    print("\nToken Usage Details:")
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Prompt Tokens: {total_prompt_tokens:,}")
    print(f"  Completion Tokens: {total_completion_tokens:,}")
    if assessments:
        print(f"  Average Tokens per Example: {total_tokens/len(assessments):.1f}")
        print(f"  Average Prompt Tokens: {total_prompt_tokens/len(assessments):.1f}")
        print(
            f"  Average Completion Tokens: {total_completion_tokens/len(assessments):.1f}"
        )
    print("\nTiming:")
    print(f"  Total Time: {total_time:.3f}s")
    print(
        f"  Average Time per Example: {total_time/len(assessments) if assessments else 0:.3f}s"
    )

    print("\nLanguage Distribution:")
    for lang, count in language_counts.most_common():
        percentage = (count / len(assessments)) * 100 if assessments else 0
        print(f"  {lang}: {count} ({percentage:.1f}%)")

    print("\nRegion Distribution:")
    for region, count in region_counts.most_common():
        percentage = (count / len(assessments)) * 100 if assessments else 0
        print(f"  {region}: {count} ({percentage:.1f}%)")

    print("\nTop Variants:")
    for variant, count in variant_counts.most_common(5):
        percentage = (count / len(assessments)) * 100 if assessments else 0
        print(f"  {variant}: {count} ({percentage:.1f}%)")

    print("\nConfidence Distribution:")
    for conf, count in confidence_counts.most_common():
        percentage = (count / len(assessments)) * 100 if assessments else 0
        print(f"  {conf}: {count} ({percentage:.1f}%)")

    # Count total linguistic features
    total_features = sum(len(a.linguistic_features) for a in assessments)
    avg_features = total_features / len(assessments) if assessments else 0
    print(
        f"\nLinguistic Features Detected: {total_features} (avg {avg_features:.1f} per phrase)"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
