import argparse
import json
import os
import time
from openai import OpenAI
from structured_output_example import assess_lang
from utils import (
    create_client,
    print_indented,
    print_response_timing,
    print_token_usage,
)


def analyze_language(phrase: str) -> dict:
    """
    Analyze the language, variant, and region of a given phrase.

    Args:
        phrase: The text phrase to analyze

    Returns:
        Dictionary with language_code, region_code, description, and variant
    """
    assessment, _, _ = assess_lang(phrase)
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


def translate_phrase(phrase: str, target_language: str, client: OpenAI, model: str):
    """
    Translate a phrase using tool calling to first identify the language.

    Args:
        phrase: The phrase to translate
        target_language: Target language for translation (e.g., 'English', 'French')
        client: OpenAI client instance
        model: Model name to use
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
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[analyze_language_tool],
        tool_choice="auto",
    )

    message = response.choices[0].message
    messages.append(message)

    # Handle tool calls
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "analyze_language":
                # Execute the function
                result = analyze_language(**arguments)

                # Send results back to the model
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

                # Get final response with translation
                final_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )

                elapsed_time = time.time() - start_time
                return response, message, final_response, result, elapsed_time

    elapsed_time = time.time() - start_time
    return response, message, None, None, elapsed_time


def main():
    """Demonstrate tool use for translation with language analysis."""
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
    print(f"Target language: {args.target}\n")

    # Example phrases to translate
    examples = [
        "I'm gonna grab some stuff from the store, then we can head out.",
        "J'vais aller chercher une poutine pis après ça, on va au dépanneur.",
        "Voy a comprar unas tortillas y luego vamos al mercado.",
        "Ich geh' mal zum Intershop, dann kaufen wir uns was.",
    ]

    print("=" * 60)
    print("Translation Tool Use Examples")
    print("=" * 60)

    for i, phrase in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print_indented("Phrase", phrase)

        try:
            (
                initial_response,
                intermediate_message,
                final_response,
                tool_result,
                elapsed_time,
            ) = translate_phrase(phrase, args.target, client, model)

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
                print_indented("Translation", final_response.choices[0].message.content)

                print("\nToken Usage (Initial + Final):")
                print_token_usage(initial_response)
                print_token_usage(final_response)
            else:
                print_indented("Response", initial_response.choices[0].message.content)
                print_token_usage(initial_response)

            print_response_timing(elapsed_time)
        except Exception as e:
            print_indented("Error", str(e))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
