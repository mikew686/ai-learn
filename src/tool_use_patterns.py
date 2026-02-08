"""
Pattern 4: Function Calling / Tool Use

Use case: Translation with language analysis tools — model uses a tool to
identify the source language (and optionally cultural context), then translates.

Patterns shown:
  - **Tool use (primary)**: Tool schema (JSON Schema), model selects tools and
    arguments; we execute tools and send results back; final response from model.
  - **Sequential tool use**: One tool call, then one final response (default mode).
  - **Parallel tool use**: Model may call multiple tools in one response; we run
    them and send all results back; one final response.
  - **Interleaved tool use**: For reasoning models (e.g. o3-mini); multiple
    tool-call rounds with reasoning between; supports multi-step workflows.

Details:
  - Modes: --mode sequential (default), parallel, or interleaved. One OpenAILog
    per run. Nested assess_lang calls (for the analyze_language tool) use the same log.
  - See eng-dev-patterns/function_calling_tool_use.md.

Usage:
    python -m src.tool_use_patterns [--model MODEL] [--mode sequential|parallel|interleaved] [--target LANG] [--prompt PHRASE] [--example-phrases] [--temperature T] [--max-tokens N]
"""

import argparse
import json
import os
import time
from enum import Enum
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from utils import (
    create_client,
    OpenAILog,
    print_indented,
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


# Enums for language assessment
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


# Pydantic models for language assessment
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


def assess_lang(
    phrase: str,
    client=None,
    model: str = None,
    log: OpenAILog | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Assess the language, variant, and region of a given phrase using structured output.

    Args:
        phrase: The text phrase to analyze
        client: OpenAI client instance (creates one if not provided)
        model: Model name to use (uses default if not provided)
        log: Optional OpenAILog to register the request/response
        temperature: Sampling temperature (omit to use API default)
        max_tokens: Max tokens per response (omit to use API default)

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
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    parse_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": LanguageAssessment,
    }
    if temperature is not None:
        parse_kwargs["temperature"] = temperature
    if max_tokens is not None:
        parse_kwargs["max_tokens"] = max_tokens
    response = client.beta.chat.completions.parse(**parse_kwargs)
    elapsed_time = time.time() - start_time

    if log is not None:
        log.register(
            "beta.chat.completions.parse",
            messages,
            response,
            elapsed_time=elapsed_time,
            label="assess_lang",
        )

    return response.choices[0].message.parsed, response, elapsed_time


def analyze_language(
    phrase: str,
    client=None,
    model: str = None,
    log: OpenAILog | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Analyze the language, variant, and region of a given phrase.

    Args:
        phrase: The text phrase to analyze
        log: Optional OpenAILog to register the request/response

    Returns:
        Dictionary with language_code, region_code, description, and variant
    """
    assessment, _, _ = assess_lang(
        phrase,
        client=client,
        model=model,
        log=log,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "language_code": assessment.language_code,
        "region_code": assessment.region_code,
        "description": assessment.description,
        "variant": assessment.variant,
    }


# Define the tool schema
analyze_language_tool = {
    "type": "function",
    "function": {
        "name": "analyze_language",
        "description": (
            "Analyze a phrase to identify its language, regional variant, "
            "and dialect characteristics. Returns ISO language code, region "
            "code, description, and variant name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phrase": {
                    "type": "string",
                    "description": "The text phrase to analyze for language identification",
                }
            },
            "required": ["phrase"],
        },
    },
}


def translate_phrase(
    phrase: str,
    target_language: str,
    client: OpenAI,
    model: str,
    log: OpenAILog,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Translate a phrase using tool calling to first identify the language.

    This example demonstrates sequential tool use: one tool call, then a final response.

    Args:
        phrase: The phrase to translate
        target_language: Target language for translation (e.g., 'English', 'French')
        client: OpenAI client instance
        model: Model name to use
        log: OpenAILog instance for consistent request/response logging
    """
    system_prompt = f"""You are a translation assistant. Your task is to:
1. Identify the language of the user's phrase using the analyze_language tool
2. If the original language is the same as the target language ({target_language}), 
   do not translate. Instead, inform the user that the phrase is already in {target_language}.
3. If the languages are different, translate the phrase to {target_language}
4. Provide the original language, target language, and translation (or note that no translation is needed)

Use the analyze_language tool to identify the source language before translating."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Translate this phrase to {target_language}: {phrase}",
        },
    ]

    start_time = time.time()

    # Initial call with tools
    create_kwargs = {
        "model": model,
        "messages": messages,
        "tools": [analyze_language_tool],
        "tool_choice": "auto",
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    log.start_call()
    response = client.chat.completions.create(**create_kwargs)
    log.register(
        "chat.completions.create",
        messages,
        response,
        label="Sequential initial",
    )

    message = response.choices[0].message
    messages.append(message)

    # Handle tool calls (sequential mode: one tool call, then final response)
    if message.tool_calls:
        # In sequential mode, we typically expect one tool call
        # Process the first tool call and return
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "analyze_language":
                # Execute the function
                result = analyze_language(
                    **arguments,
                    client=client,
                    model=model,
                    log=log,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Send results back to the model
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

                # Get final response with translation
                final_kwargs = {"model": model, "messages": messages}
                if temperature is not None:
                    final_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    final_kwargs["max_tokens"] = max_tokens
                log.start_call()
                final_response = client.chat.completions.create(**final_kwargs)
                log.register(
                    "chat.completions.create",
                    messages,
                    final_response,
                    label="Sequential final",
                )

                elapsed_time = time.time() - start_time
                return response, message, final_response, result, elapsed_time

    elapsed_time = time.time() - start_time
    return response, message, None, None, elapsed_time


def translate_with_parallel_tools(
    phrase: str,
    target_language: str,
    client: OpenAI,
    model: str,
    log: OpenAILog,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Translate a phrase using parallel tool calls.

    Demonstrates parallel tool use: the model can call multiple tools simultaneously
    in a single response. All tool calls are executed in parallel, then results are
    sent back together for the final response.

    Args:
        phrase: The phrase to translate
        target_language: Target language for translation
        client: OpenAI client instance
        model: Model name to use
        log: OpenAILog instance for consistent request/response logging

    Returns:
        Tuple of (initial_response, tool_results, final_response, elapsed_time)
    """
    # Additional tools for parallel execution
    get_cultural_context_tool = {
        "type": "function",
        "function": {
            "name": "get_cultural_context",
            "description": (
                "Get cultural context for a language code. "
                "Use this to understand cultural nuances that may affect translation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language_code": {
                        "type": "string",
                        "description": "ISO 639-1 language code",
                    },
                },
                "required": ["language_code"],
            },
        },
    }

    def get_cultural_context(language_code: str) -> dict:
        """Get cultural context for a language code."""
        contexts = {
            "en": "Direct communication, clarity valued",
            "es": "Formal/informal distinctions (tú vs. usted)",
            "fr": "Politeness and formality important",
            "de": "Direct and precise communication",
        }
        return {
            "language_code": language_code,
            "context": contexts.get(language_code, "General context"),
        }

    system_prompt = f"""You are a translation assistant. Your task is to:
1. Identify the language of the user's phrase using analyze_language
2. Get cultural context for both source and target languages using get_cultural_context
3. Translate the phrase to {target_language} considering cultural nuances
4. Provide the translation with cultural notes

You can call multiple tools in parallel to gather information efficiently."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Translate this phrase to {target_language}: {phrase}",
        },
    ]

    start_time = time.time()

    # Initial call with multiple tools available
    create_kwargs = {
        "model": model,
        "messages": messages,
        "tools": [analyze_language_tool, get_cultural_context_tool],
        "tool_choice": "auto",
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    log.start_call()
    response = client.chat.completions.create(**create_kwargs)
    log.register(
        "chat.completions.create",
        messages,
        response,
        label="Parallel initial",
    )

    message = response.choices[0].message
    messages.append(message)

    # Handle parallel tool calls
    # Note: "Parallel" here means the model can call multiple tools in one response.
    # Tool execution is sequential (which is fine for simple synchronous functions).
    # For I/O-bound tools, consider using concurrent.futures for true parallelism.
    tool_results = {}
    if message.tool_calls:
        # Execute all tool calls (model may call multiple in one response)
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "analyze_language":
                result = analyze_language(
                    **arguments,
                    client=client,
                    model=model,
                    log=log,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif function_name == "get_cultural_context":
                result = get_cultural_context(**arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}

            tool_results[tool_call.id] = result

            # Add all tool results to messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        # Get final response with all tool results
        final_kwargs = {"model": model, "messages": messages}
        if temperature is not None:
            final_kwargs["temperature"] = temperature
        if max_tokens is not None:
            final_kwargs["max_tokens"] = max_tokens
        log.start_call()
        final_response = client.chat.completions.create(**final_kwargs)
        log.register(
            "chat.completions.create",
            messages,
            final_response,
            label="Parallel final",
        )

        elapsed_time = time.time() - start_time
        return response, tool_results, final_response, elapsed_time

    elapsed_time = time.time() - start_time
    return response, {}, None, elapsed_time


def translate_with_interleaved_tools(
    phrase: str,
    target_language: str,
    client: OpenAI,
    model: str,
    log: OpenAILog,
    max_iterations: int = 5,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Translate a phrase using interleaved tool calls (for reasoning models).

    Demonstrates interleaved tool use: reasoning models (like o-series) can make
    tool calls, receive results, reason about them, and make additional tool calls
    in the same conversation. This allows for iterative refinement and multi-step
    reasoning with tools.

    This pattern is particularly useful with reasoning models (o3, o4-mini) that
    support extended reasoning and can make multiple tool calls in sequence.

    Args:
        phrase: The phrase to translate
        target_language: Target language for translation
        client: OpenAI client instance
        model: Model name to use (should be a reasoning model like o3-mini, o4-mini)
        log: OpenAILog instance for consistent request/response logging
        max_iterations: Maximum number of tool-call iterations (default: 5)

    Returns:
        Tuple of (all_responses, all_tool_results, final_response, elapsed_time)
    """
    get_translation_suggestions_tool = {
        "type": "function",
        "function": {
            "name": "get_translation_suggestions",
            "description": (
                "Get multiple translation suggestions for a phrase. "
                "Use this to explore different translation options and compare them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string", "description": "Phrase to translate"},
                    "source_lang": {
                        "type": "string",
                        "description": "Source language code",
                    },
                    "target_lang": {
                        "type": "string",
                        "description": "Target language code",
                    },
                },
                "required": ["phrase", "source_lang", "target_lang"],
            },
        },
    }

    def get_translation_suggestions(
        phrase: str, source_lang: str, target_lang: str
    ) -> dict:
        """Get multiple translation suggestions."""
        # Simulated translation suggestions
        suggestions = {
            ("en", "fr"): [
                "Allons passer du temps ensemble, qu'en dis-tu?",
                "On va traîner, tu en penses quoi?",
                "Sortons ensemble, ça te dit?",
            ],
            ("fr", "en"): [
                "Let's go hang out, what do you say?",
                "Let's spend time together, what do you think?",
                "Shall we go out together?",
            ],
        }
        key = (source_lang, target_lang)
        return {
            "suggestions": suggestions.get(key, ["Translation not available"]),
            "count": len(suggestions.get(key, [])),
        }

    system_prompt = """You are a translation assistant using advanced reasoning.

Your task is to:
1. Identify the source language using analyze_language
2. Get translation suggestions using get_translation_suggestions
3. Reason about which translation best captures the meaning and cultural context
4. If needed, get more information or refine your approach
5. Provide the final translation with explanation

You can make multiple tool calls in sequence, reasoning between each one.
Use interleaved tool calls to iteratively refine your translation."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Translate this phrase to {target_language}: {phrase}",
        },
    ]

    start_time = time.time()
    all_responses = []
    all_tool_results = []
    iteration = 0

    # Interleaved tool calling loop
    # Reasoning models can make multiple tool calls in sequence, reasoning between each.
    # This loop continues until the model provides a final response (no more tool calls).
    while iteration < max_iterations:
        iteration += 1

        # Make API call (may return tool calls or final response)
        create_kwargs = {
            "model": model,
            "messages": messages,
            "tools": [analyze_language_tool, get_translation_suggestions_tool],
            "tool_choice": "auto",
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        log.start_call()
        response = client.chat.completions.create(**create_kwargs)
        log.register(
            "chat.completions.create",
            messages,
            response,
            label=f"Interleaved iter {iteration}",
        )

        all_responses.append(response)
        message = response.choices[0].message
        messages.append(message)

        # If no tool calls and we have content, we have the final response
        if not message.tool_calls and message.content:
            elapsed_time = time.time() - start_time
            return all_responses, all_tool_results, response, elapsed_time

        # Handle tool calls for this iteration
        # Model may call multiple tools, then reason about results in next iteration
        iteration_tool_results = {}
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "analyze_language":
                result = analyze_language(
                    **arguments,
                    client=client,
                    model=model,
                    log=log,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif function_name == "get_translation_suggestions":
                result = get_translation_suggestions(**arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}

            iteration_tool_results[tool_call.id] = result
            all_tool_results.append((iteration, tool_call.id, result))

            # Add tool result to messages for next iteration
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    # If we hit max iterations, return the last response
    elapsed_time = time.time() - start_time
    return all_responses, all_tool_results, response, elapsed_time


def main():
    """
    Demonstrate tool use for translation with language analysis.

    Three modes are supported:
    - sequential: One tool call, then final response (2 API calls total)
    - parallel: Multiple tools called in one response, then final response (2 API calls total)
    - interleaved: Multiple tool calls in sequence with reasoning (N API calls, N >= 2)
    """
    parser = argparse.ArgumentParser(
        description="Translate phrases using tool calling for language analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides MODEL env var and default)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="English",
        help="Target language for translation (default: English)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "parallel", "interleaved"],
        default="sequential",
        help="Tool use mode: sequential (default), parallel, or interleaved (for reasoning models)",
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
        help="Override with a custom phrase to translate (takes precedence over --example-phrases)",
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
    print(f"Using model: {model}")
    print(f"Target language: {args.target}")
    print(f"Tool use mode: {args.mode}")

    # Mode-specific notes
    if args.mode == "parallel":
        print("  → Multiple tools can be called in one response")
    elif args.mode == "interleaved":
        print("  → Reasoning models can make multiple tool calls in sequence")
        print("  → Recommended for o3-mini, o4-mini models")
    print()

    # Determine which phrases to use
    # Priority: --prompt > --example-phrases > default single phrase
    if args.prompt:
        examples = [args.prompt]
        print(f"Translating custom phrase: {args.prompt}\n")
    elif args.example_phrases:
        # Example phrases demonstrating different language variants
        examples = [
            "I'm gonna grab some stuff from the store, then we can head out.",
            "J'vais aller chercher une poutine pis après ça, on va au dépanneur.",
            "Voy a comprar unas tortillas y luego vamos al mercado.",
            "Ich geh' mal zum Intershop, dann kaufen wir uns was.",
        ]
        print("Using example phrases list\n")
    else:
        # Default: single phrase
        examples = ["Ca va, mec?"]
        print("Using default phrase\n")

    print("=" * 60)
    print(f"Translation Tool Use Examples - Mode: {args.mode}")
    print("=" * 60)

    log = OpenAILog()
    if args.mode == "interleaved":
        print(
            "\nNote: Interleaved mode is designed for reasoning models (o3-mini, o4-mini). "
            "These models can make multiple tool calls in sequence with reasoning between them."
        )

    for i, phrase in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print_indented("Phrase", phrase)

        try:
            if args.mode == "sequential":
                (
                    initial_response,
                    intermediate_message,
                    final_response,
                    tool_result,
                    elapsed_time,
                ) = translate_phrase(
                    phrase,
                    args.target,
                    client,
                    model,
                    log,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                if tool_result and final_response:
                    print("\nIntermediate Message (Assistant with Tool Call):")
                    if intermediate_message.tool_calls:
                        for tool_call in intermediate_message.tool_calls:
                            print_indented("Tool", tool_call.function.name)
                            print_indented(
                                "Arguments",
                                json.dumps(
                                    json.loads(tool_call.function.arguments), indent=2
                                ),
                            )
                    if intermediate_message.content:
                        print_indented("Content", intermediate_message.content)

                    print("\nTool Call Response:")
                    print_indented("Language Code", tool_result["language_code"])
                    print_indented("Region Code", tool_result["region_code"])
                    print_indented("Variant", tool_result["variant"])
                    print_indented("Description", tool_result["description"])

                    print("\nFinal Response:")
                    print_indented(
                        "Translation", final_response.choices[0].message.content
                    )
                else:
                    print_indented(
                        "Response", initial_response.choices[0].message.content
                    )

            elif args.mode == "parallel":
                (
                    initial_response,
                    tool_results,
                    final_response,
                    elapsed_time,
                ) = translate_with_parallel_tools(
                    phrase,
                    args.target,
                    client,
                    model,
                    log,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                if tool_results and final_response:
                    print("\nParallel Tool Calls:")
                    for tool_id, result in tool_results.items():
                        print_indented("Tool Result", json.dumps(result, indent=2))

                    print("\nFinal Response:")
                    print_indented(
                        "Translation", final_response.choices[0].message.content
                    )
                else:
                    print_indented(
                        "Response", initial_response.choices[0].message.content
                    )

            elif args.mode == "interleaved":
                (
                    all_responses,
                    all_tool_results,
                    final_response,
                    elapsed_time,
                ) = translate_with_interleaved_tools(
                    phrase,
                    args.target,
                    client,
                    model,
                    log,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                print(f"\nInterleaved Tool Calls ({len(all_responses)} iterations):")
                for iteration, tool_id, result in all_tool_results:
                    print_indented(
                        f"Iteration {iteration}",
                        json.dumps(result, indent=2),
                    )

                if final_response:
                    print("\nFinal Response:")
                    print_indented(
                        "Translation", final_response.choices[0].message.content
                    )
        except Exception as e:
            print_indented("Error", str(e))

    log.print_summary()

    print("\n" + "=" * 60)
    print("\nTool Use Patterns Summary:")
    print("  Sequential: 1 tool call → 1 final response (2 API calls total)")
    print("    - Efficient for single tool operations")
    print("    - Standard pattern for most use cases")
    print()
    print(
        "  Parallel: Multiple tools in 1 response → 1 final response (2 API calls total)"
    )
    print("    - Efficient for independent operations")
    print("    - Model can call multiple tools simultaneously")
    print()
    print("  Interleaved: N tool calls in sequence → 1 final response (N+1 API calls)")
    print("    - For reasoning models (o3-mini, o4-mini)")
    print("    - Model reasons between tool calls")
    print("    - Supports iterative refinement")


if __name__ == "__main__":
    main()
