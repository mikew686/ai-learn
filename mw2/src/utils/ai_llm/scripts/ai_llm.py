#!/usr/bin/env python3
"""
AI LLM CLI. Run with: python -m utils.ai_llm.scripts.ai_llm <command> [options]

Commands: models, chat, lang_example, lang_analyze, translate.

Recommended models for translation (fast, inexpensive, works with openai/text-embedding-3-small):
  google/gemini-3.1-flash-lite-preview  2026-03-03  (default)
  google/gemini-2.5-flash-lite          2025-07-22
  openai/gpt-4o-mini                    2024-07-18
  amazon/nova-micro-v1                  2024-12-05
"""

import argparse
import json
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from utils.ai_llm import AIClient, ChatRequest, DictToolProvider

DATA_AI_LLM_DIR = Path(__file__).resolve().parent / "data" / "ai_llm"


def print_indented(
    label: str,
    content: str,
    indent: int = 2,
    width: int = 120,
    max_length: int | None = None,
) -> None:
    """Print a label followed by indented multiline content with text wrapping."""
    if max_length is not None and len(content) > max_length:
        content = content[:max_length] + "\n... [truncated]"
    if label:
        print(f"{label}:")
    indent_str = " " * indent
    available_width = width - indent
    sections = content.split("\n")
    for section in sections:
        if section.strip():
            for line in textwrap.wrap(section, width=available_width):
                print(f"{indent_str}{line}")
        else:
            print()


class LanguageDetails(BaseModel):
    """Structured language identification. Extract from user input or phrase. Use empty string for optional fields when not specified or inferable."""

    language_name: str = Field(
        description="Full English name of the language. Required. Example: French, Spanish, English."
    )
    language_code: str = Field(
        description="ISO 639-1 two-letter code. Required. Example: en, fr, es, zh."
    )
    region_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 region code (CA, US, MX, FR). Use when a region or dialect is indicated; otherwise empty string.",
    )
    dialect_name: str = Field(
        default="",
        description="Specific dialect or regional variant when identifiable. Example: Quebec French, Southern US English, Latin American Spanish. Empty if no variant indicated.",
    )
    audience_name: str = Field(
        default="",
        description="Target audience or register when relevant. Example: children, academic, formal, casual. Empty if not specified.",
    )
    canonical_name: str = Field(
        default="",
        description="Short human-readable label: dialect and audience. Format: 'Quebec French / Elementary School' or 'Quebec French'. Empty if generic.",
    )
    language_notes: str = Field(
        default="",
        description="Always include formality level (formal/informal) and cultural context when relevant (e.g. gendered nouns, honorifics, regional conventions). Empty only if no applicable notes.",
    )


class TranslationWithNotes(BaseModel):
    """Translation plus ARPABET phonetic transcription and brief notes (cultural/contextual)."""

    translated_text: str = Field(
        description="The translation in the target language and dialect. Use natural, idiomatic phrasing."
    )
    phonetic_spelling: str = Field(
        description="ARPABET transcription of the translation: space-separated 2-letter phoneme codes (AA, AE, B, CH, etc.) with stress digits 0/1/2 after vowels. See https://en.wikipedia.org/wiki/ARPABET"
    )
    notes: str = Field(
        description="Brief cultural or contextual notes about the translation (in English)"
    )


EXTRACT_LANGUAGE_DETAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_language_details",
        "description": "Extract structured language details. Pass exactly one of reference or example—never both, never neither.",
        "parameters": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": "Language specification to parse. Use when the user describes a variant in words (e.g. Quebec French for children, formal Latin American Spanish). Required if example is omitted.",
                },
                "example": {
                    "type": "string",
                    "description": "Phrase or sentence to identify. Use when the user provides text to analyze (e.g. buenos días, je m'appelle Marie). Required if reference is omitted.",
                },
            },
            "required": [],
        },
    },
}


def _run_extract_language_details(
    client: AIClient,
    model: str,
    *,
    reference: str | None = None,
    example: str | None = None,
    log_chat_completion=None,
) -> dict:
    """Extract LanguageDetails from reference or example. Exactly one of reference/example required."""
    has_ref = reference and reference.strip()
    has_ex = example and example.strip()
    if has_ref == has_ex:  # both or neither
        return {"error": "Exactly one of reference or example is required."}

    text = (reference if has_ref else example).strip()
    system = (
        "Parse this language specification into structured details. Extract language, dialect, region, audience, and cultural context from the description."
        if has_ref
        else "Identify the language of this phrase and extract structured details. Infer dialect, region, register, and cultural context from the text."
    )
    req = ChatRequest(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        response_format=LanguageDetails,
        temperature=0,
        log_chat_completion=log_chat_completion,
    )
    result, _ = client.run_turn(req)
    if result is None:
        return {"error": "Failed to extract language details"}
    return result.model_dump(mode="json") if hasattr(result, "model_dump") else result


def make_extract_language_details_handler(
    client: AIClient,
    model: str,
    *,
    log_chat_completion=None,
):
    """Return a tool handler that extracts LanguageDetails via structured output."""

    def extract_language_details(
        reference: str | None = None, example: str | None = None
    ) -> dict:
        print(
            f"  [tool] extract_language_details(reference={reference!r}, example={example!r})"
        )
        return _run_extract_language_details(
            client,
            model,
            reference=reference,
            example=example,
            log_chat_completion=log_chat_completion,
        )

    return extract_language_details


GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date, time, and timezone for a location. Use when the user asks for the time or date. Pass timezone (IANA e.g. America/New_York, Europe/Paris) when a location is mentioned; omit for the user's local time. Returns: date (YYYY-MM-DD), time (HH:MM:SS), timezone (IANA name used), local_timezone (user's system timezone for reference), datetime (ISO). Always report the timezone in your response.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone (e.g. America/New_York, Europe/London, Asia/Tokyo). Omit for the user's local time.",
                },
            },
            "required": [],
        },
    },
}


def _get_localzone():
    """Return system local timezone. Uses astimezone() (stdlib) for portability."""
    return datetime.now().astimezone().tzinfo or ZoneInfo("UTC")


def get_current_time(timezone: str | None = None) -> dict:
    """Return date, time, and timezone for a location. If timezone is omitted, use system local."""
    print(f"  [tool] get_current_time(timezone={timezone!r})")
    local_tz = _get_localzone()
    if timezone:
        try:
            tz = ZoneInfo(timezone)
        except Exception:
            return {
                "error": f"Unknown timezone: {timezone}",
                "local_timezone": str(local_tz),
            }
    else:
        tz = local_tz
    now = datetime.now(tz)
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": str(tz),
        "local_timezone": str(local_tz),
        "datetime": now.isoformat(),
    }


def log_chat_completion(start_ts: float, end_ts: float, response: object) -> None:
    """Log LLM call to data/ai_llm/YYYYMMDD/<timestamp>.json and print a one-line summary."""
    duration = end_ts - start_ts
    model = getattr(response, "model", "?")
    usage = getattr(response, "usage", None)
    prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
    compl_tok = getattr(usage, "completion_tokens", None) if usage else None
    cost = getattr(usage, "cost", None) if usage else None
    tok_str = (
        f"{prompt_tok}+{compl_tok} tok"
        if prompt_tok is not None and compl_tok is not None
        else "? tok"
    )
    cost_str = f"${cost:.6f}" if cost is not None else "$?"
    print(f"  [llm] {duration:.1f}s | {model} | {tok_str} | {cost_str}")

    dt = datetime.fromtimestamp(start_ts)
    day_dir = DATA_AI_LLM_DIR / dt.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.strftime("%Y-%m-%dT%H-%M-%S") + f".{int((start_ts % 1) * 1_000_000):06d}"
    out_path = day_dir / f"{ts}.json"
    payload = {"start": start_ts, "end": end_ts, "duration": duration}
    if hasattr(response, "model_dump"):
        try:
            payload["response"] = response.model_dump(mode="json")
        except TypeError:
            payload["response"] = response.model_dump()
    else:
        payload["response"] = str(response)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def cmd_models(client: AIClient, args: argparse.Namespace) -> None:
    """Fetch models, write to data/models.json, print summary."""
    default_path = Path(__file__).resolve().parent / "data" / "models.json"
    out_path = Path(args.output) if args.output else default_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models_dict = client.models()
    with open(out_path, "w") as f:
        json.dump(models_dict, f, indent=2)
    count = len(models_dict)
    owners = {mid.split("/")[0] for mid in models_dict}
    print(f"{count} models, {len(owners)} owners")


def cmd_chat(client: AIClient, args: argparse.Namespace) -> None:
    """Interactive chat. Default model: google/gemini-3.1-flash-lite-preview."""
    tool_provider = DictToolProvider()
    tool_provider.add_tool(GET_TIME_TOOL, get_current_time)
    log_cb = None if getattr(args, "no_log", False) else log_chat_completion
    tool_provider.add_tool(
        EXTRACT_LANGUAGE_DETAILS_TOOL,
        make_extract_language_details_handler(
            client, args.model, log_chat_completion=log_cb
        ),
    )
    messages: list[dict] = [
        {
            "role": "system",
            "content": "You are a useful assistant. When asked for the time or date, call get_current_time. Pass timezone (IANA e.g. Europe/Paris) when a location is mentioned; omit for local. Always report the timezone with the date in your response. When asked to identify a language from a phrase, call extract_language_details with example set to that phrase. When asked about a language variant described in words (e.g. Quebec French for children), use reference instead. Then explain the result clearly.",
        }
    ]

    print(f"Model: {args.model}")
    print("Tools: get_current_time, extract_language_details")
    if getattr(args, "no_log", False):
        print("Logging disabled")
    print('Type "done" to exit.\n')

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() == "done":
            break

        messages.append({"role": "user", "content": prompt})

        req = ChatRequest(
            model=args.model,
            messages=messages,
            tool_provider=tool_provider,
            log_chat_completion=log_cb,
            idempotency_key=f"mw2-ai-llm-{uuid.uuid4()}",
            temperature=getattr(args, "temperature", None),
            max_tokens=getattr(args, "max_tokens", None),
            top_p=getattr(args, "top_p", None),
        )
        result, _ = client.run_turn(req)

        if result:
            print()
            print_indented("Assistant", str(result))
            print()
            messages.append({"role": "assistant", "content": result})
        else:
            print("\nAssistant: (no text)\n")

    print("Goodbye.")


def cmd_lang_example(client: AIClient, args: argparse.Namespace) -> None:
    """Interactive loop: extract language details from phrases."""
    log_cb = None if getattr(args, "no_log", False) else log_chat_completion
    print(f"Model: {args.model}")
    print('Type "done" to exit.\n')
    while True:
        try:
            phrase = input("language example > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not phrase:
            continue
        if phrase.lower() == "done":
            break
        result = _run_extract_language_details(
            client, args.model, example=phrase, log_chat_completion=log_cb
        )
        if "error" in result:
            print(result["error"])
        else:
            print_indented("Language details", _format_language_details(result))
        print()
    print("Goodbye.")


def cmd_lang_analyze(client: AIClient, args: argparse.Namespace) -> None:
    """Interactive loop: extract language details from references."""
    log_cb = None if getattr(args, "no_log", False) else log_chat_completion
    print(f"Model: {args.model}")
    print('Type "done" to exit.\n')
    while True:
        try:
            reference = input("language to analyze > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not reference:
            continue
        if reference.lower() == "done":
            break
        result = _run_extract_language_details(
            client, args.model, reference=reference, log_chat_completion=log_cb
        )
        if "error" in result:
            print(result["error"])
        else:
            print_indented("Language details", _format_language_details(result))
        print()
    print("Goodbye.")


def _format_language_details(d: dict) -> str:
    """Format LanguageDetails dict for user display."""
    lines = []
    if d.get("language_name"):
        code = d.get("language_code", "")
        lines.append(f"Language: {d['language_name']}" + (f" ({code})" if code else ""))
    if d.get("dialect_name"):
        lines.append(f"Dialect: {d['dialect_name']}")
    if d.get("region_code"):
        lines.append(f"Region: {d['region_code']}")
    if d.get("audience_name"):
        lines.append(f"Audience: {d['audience_name']}")
    if d.get("canonical_name"):
        lines.append(f"Canonical: {d['canonical_name']}")
    if d.get("language_notes"):
        lines.append(f"Notes: {d['language_notes']}")
    return "\n".join(lines)


def _language_details_to_system_prompt(d: dict) -> str:
    """Build translator system prompt from LanguageDetails dict."""
    parts = [
        "You are a translator. Translate into the following target language. "
        "Provide the translation, an ARPABET phonetic transcription, and brief cultural/contextual notes."
    ]
    parts.append(
        f"Language: {d.get('language_name', '')} ({d.get('language_code', '')})"
    )
    if d.get("dialect_name"):
        parts.append(f"Dialect: {d['dialect_name']}")
    if d.get("region_code"):
        parts.append(f"Region: {d['region_code']}")
    if d.get("audience_name"):
        parts.append(f"Audience: {d['audience_name']}")
    if d.get("canonical_name"):
        parts.append(f"Canonical target: {d['canonical_name']}")
    if d.get("language_notes"):
        parts.append(f"Notes: {d['language_notes']}")
    return "\n".join(parts)


def cmd_translate(client: AIClient, args: argparse.Namespace) -> None:
    """Interactive translation: first set target language, then translate phrases."""
    log_cb = None if getattr(args, "no_log", False) else log_chat_completion
    print(f"Model: {args.model}")
    try:
        lang_input = input("language to translate to > ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not lang_input:
        print("No language specified.")
        return

    lang_result = _run_extract_language_details(
        client, args.model, reference=lang_input, log_chat_completion=log_cb
    )
    if "error" in lang_result:
        print(lang_result["error"])
        return

    print_indented("Language details", _format_language_details(lang_result))
    print()
    canonical = lang_result.get("canonical_name") or lang_result.get(
        "language_name", "target language"
    )
    system_prompt = _language_details_to_system_prompt(lang_result)
    print(f"Target: {canonical}\n")
    print('Type "done" to exit.\n')

    while True:
        try:
            phrase = input("phrase to translate > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not phrase:
            continue
        if phrase.lower() == "done":
            break

        req = ChatRequest(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate to {canonical}: {phrase}"},
            ],
            response_format=TranslationWithNotes,
            log_chat_completion=log_cb,
            temperature=0,
        )
        result, _ = client.run_turn(req)
        if result:
            r = result.model_dump() if hasattr(result, "model_dump") else result
            print_indented("Translation", r.get("translated_text", ""))
            if r.get("phonetic_spelling"):
                print_indented("Phonetics", r["phonetic_spelling"])
            if r.get("notes"):
                print_indented("Notes", r["notes"])
            print()
        else:
            print("(no translation)\n")

    print("Goodbye.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI LLM CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--client-retries",
        type=int,
        default=None,
        help="Override client max_retries",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="List models and save to JSON")
    models_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/models.json)",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--model", default="google/gemini-3.1-flash-lite-preview", help="Model"
    )
    chat_parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature"
    )
    chat_parser.add_argument(
        "--max-tokens", type=int, default=None, dest="max_tokens", help="Max tokens"
    )
    chat_parser.add_argument(
        "--top-p", type=float, default=None, dest="top_p", help="Top-p sampling"
    )
    chat_parser.add_argument(
        "--no-log",
        action="store_true",
        dest="no_log",
        help="Disable logging to data/ai_llm/",
    )

    lang_example_parser = subparsers.add_parser(
        "lang_example", help="Interactive: extract language details from phrases"
    )
    lang_example_parser.add_argument(
        "--model", default="google/gemini-3.1-flash-lite-preview", help="Model"
    )
    lang_example_parser.add_argument(
        "--no-log",
        action="store_true",
        dest="no_log",
        help="Disable logging to data/ai_llm/",
    )

    lang_analyze_parser = subparsers.add_parser(
        "lang_analyze",
        help="Interactive: extract language details from references",
    )
    lang_analyze_parser.add_argument(
        "--model", default="google/gemini-3.1-flash-lite-preview", help="Model"
    )
    lang_analyze_parser.add_argument(
        "--no-log",
        action="store_true",
        dest="no_log",
        help="Disable logging to data/ai_llm/",
    )

    translate_parser = subparsers.add_parser(
        "translate",
        help="Interactive: translate phrases to a target language",
    )
    translate_parser.add_argument(
        "--model", default="google/gemini-3.1-flash-lite-preview", help="Model"
    )
    translate_parser.add_argument(
        "--no-log",
        action="store_true",
        dest="no_log",
        help="Disable logging to data/ai_llm/",
    )

    args = parser.parse_args()
    kwargs = {}
    if args.client_retries is not None:
        kwargs["max_retries"] = args.client_retries
    client = AIClient(**kwargs)

    if args.command == "models":
        cmd_models(client, args)
    elif args.command == "chat":
        cmd_chat(client, args)
    elif args.command == "lang_example":
        cmd_lang_example(client, args)
    elif args.command == "lang_analyze":
        cmd_lang_analyze(client, args)
    elif args.command == "translate":
        cmd_translate(client, args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
