"""
Pattern 4: Schema-Driven Inference

This example demonstrates combining multiple patterns:
- Function Calling / Tool Use: Tools for language lookup and cultural context
- Structured Output: Pydantic models for validated translation results
- Schema-Driven Inference: Minimal prompts with detailed schemas

Key Concepts:
- Tool schema descriptions as implicit instructions
- Pydantic field descriptions guide output generation
- Minimal prompts with detailed schemas
- Combining explicit and implicit instructions

The prompt is intentionally simplified - much of the detail is inferred by the model from:
- Tool descriptions: The model understands what tools are available and their purposes
- Pydantic field descriptions: Guide the model on what data to extract and how to format it
- Tool execution results: The model uses results from tool calls to inform structured output

This pattern reduces prompt verbosity while maintaining high-quality, validated outputs.

Use Case: Translation with tools and structured output
"""

import argparse
import json
import os
import time
from pydantic import BaseModel, Field
from openai import OpenAI
from utils import (
    create_client,
    print_indented,
    print_response_timing,
    print_token_usage,
)


class TranslationResult(BaseModel):
    """Structured translation result."""

    source_language: str = Field(
        description="Source language name in plain English (e.g., 'English', 'French', 'Spanish')"
    )
    source_language_code: str = Field(
        description="Source language code (ISO 639-1, e.g., 'en', 'fr', 'es')"
    )
    target_language: str = Field(
        description="Target language name in plain English (e.g., 'English', 'French', 'Spanish')"
    )
    target_language_code: str = Field(
        description="Target language code (ISO 639-1, e.g., 'en', 'fr', 'es')"
    )
    translated_text: str = Field(description="The translated text")
    confidence: str = Field(description="Translation confidence: low, medium, high")
    cultural_notes: str = Field(
        description="Cultural or contextual notes about the translation (always in English)"
    )


# Tool functions
def lookup_language_code(language_name: str) -> dict:
    """Look up ISO 639-1 language code for a language name."""
    language_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
    }
    code = language_map.get(language_name.lower(), "unknown")
    return {
        "language_name": language_name,
        "language_code": code,
        "found": code != "unknown",
    }


def get_cultural_context(language_code: str, phrase: str) -> dict:
    """Get cultural context for a phrase in a given language."""
    # Simulated cultural context lookup
    contexts = {
        "en": "English-speaking cultures generally use direct communication and value clarity. Regional variations exist (British, American, Australian, etc.) with different idioms and expressions.",
        "es": "Spanish-speaking cultures often use formal/informal distinctions (tú vs. usted). Regional variations are significant (Spain, Latin America) with different vocabulary and expressions.",
        "fr": "French culture values politeness and formality in language. The use of 'vous' (formal) vs. 'tu' (informal) is important in social interactions.",
        "de": "German language has formal and informal address forms (Sie vs. du). German communication tends to be direct and precise.",
        "it": "Italian culture emphasizes warmth and expressiveness in communication. Regional dialects vary significantly across Italy.",
        "pt": "Portuguese-speaking cultures (Brazil, Portugal) have distinct variations. Brazilian Portuguese is more informal, while European Portuguese maintains more formality.",
        "zh": "Chinese culture emphasizes respect and hierarchy in communication. Different forms exist for Mandarin and Cantonese, with regional variations.",
        "ja": "Japanese culture emphasizes respect and context in communication. Multiple levels of formality exist (keigo) and indirect communication is common.",
        "ko": "Korean culture has complex honorific systems (jondaetmal vs. banmal) that reflect social hierarchy and relationships.",
        "ru": "Russian language has formal and informal address forms (вы vs. ты). Communication can be direct, and formality depends on context.",
        "ar": "Arabic-speaking cultures value politeness and respect. Different dialects exist across regions, and formality levels vary by context.",
        "hi": "Hindi-speaking cultures use different levels of formality. Regional variations exist, and English words are often integrated (Hinglish).",
    }
    return {
        "language_code": language_code,
        "context": contexts.get(language_code, "General cultural context"),
        "formality_level": "varies",
    }


# Define tool schemas
lookup_language_code_tool = {
    "type": "function",
    "function": {
        "name": "lookup_language_code",
        "description": "Look up ISO 639-1 language code for a language name",
        "parameters": {
            "type": "object",
            "properties": {
                "language_name": {
                    "type": "string",
                    "description": "Name of the language (e.g., 'Spanish', 'French')",
                }
            },
            "required": ["language_name"],
        },
    },
}

get_cultural_context_tool = {
    "type": "function",
    "function": {
        "name": "get_cultural_context",
        "description": "Get cultural context information for a phrase in a language",
        "parameters": {
            "type": "object",
            "properties": {
                "language_code": {
                    "type": "string",
                    "description": "ISO 639-1 language code",
                },
                "phrase": {
                    "type": "string",
                    "description": "The phrase to get context for",
                },
            },
            "required": ["language_code", "phrase"],
        },
    },
}


def translate_with_tools_and_structured(
    source_text: str, target_language: str, client: OpenAI, model: str
):
    """
    Translate text using tool calling and structured output.

    This example demonstrates combining Function Calling / Tool Use with Structured Output
    patterns (see eng-dev-patterns/function_calling_tool_use.md and
    eng-dev-patterns/structured_output.md).

    The prompt is intentionally simplified - much of the detail is inferred by the model from:
    - Tool descriptions: The model understands what tools are available and their purposes
      from the tool schema definitions (lookup_language_code, get_cultural_context)
    - Pydantic field descriptions: The TranslationResult model's Field descriptions guide
      the model on what data to extract and how to format it
    - Tool execution results: The model uses results from tool calls to inform its
      structured output generation

    This pattern reduces prompt verbosity while maintaining high-quality, validated outputs.

    Args:
        source_text: The text to translate
        target_language: Target language for translation (e.g., 'French', 'Spanish')
        client: OpenAI client instance
        model: Model name to use

    Returns:
        Tuple of (translation_result, initial_response, final_response, elapsed_time)
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"Translate the following text to {target_language} and provide a structured translation result. "
                f"Important: Cultural notes must always be written in English. "
                f"Text to translate: {source_text}"
            ),
        }
    ]

    tools = [lookup_language_code_tool, get_cultural_context_tool]

    start_time = time.time()

    # Initial call with tools
    initial_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = initial_response.choices[0].message
    messages.append(message)

    # Handle tool calls
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "lookup_language_code":
                result = lookup_language_code(**arguments)
            elif function_name == "get_cultural_context":
                result = get_cultural_context(**arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}

            # Send results back to the model
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    # Final call with structured output
    final_response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=TranslationResult,
    )

    elapsed_time = time.time() - start_time
    translation_result = final_response.choices[0].message.parsed

    return translation_result, initial_response, final_response, elapsed_time


def main():
    """Run combined tool calling and structured output example."""
    parser = argparse.ArgumentParser(
        description="Combined tool calling and structured output example"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides MODEL env var and default)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override the default phrase to translate",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Override the target language (default: French)",
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

    # Example: Translate a phrase
    source_text = (
        args.prompt if args.prompt else "Let's go hang out, what do you say to that?"
    )
    target_language = args.target if args.target else "French"

    print(f"Source text: {source_text}")
    print(f"Target language: {target_language}\n")

    try:
        (
            translation_result,
            initial_response,
            final_response,
            elapsed_time,
        ) = translate_with_tools_and_structured(
            source_text, target_language, client, model
        )

        print("✓ Success")
        print_indented(
            "Source Language",
            f"{translation_result.source_language} ({translation_result.source_language_code})",
        )
        print_indented(
            "Target Language",
            f"{translation_result.target_language} ({translation_result.target_language_code})",
        )
        print_indented("Translated Text", translation_result.translated_text)
        print_indented("Confidence", translation_result.confidence)
        print_indented("Cultural Notes", translation_result.cultural_notes)

        # Token usage for all steps
        print("\nToken Usage:")
        print_token_usage(initial_response, "Initial (with tools)")
        print_token_usage(final_response, "Final (including tools results)")

        # Total token usage
        initial_tokens = initial_response.usage.total_tokens
        final_tokens = final_response.usage.total_tokens
        total_tokens = initial_tokens + final_tokens
        print(f"\nTotal Tokens: {total_tokens}")

        print_response_timing(elapsed_time)
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
