"""
Interactive demo: Reasoning models (o1, o3, etc.) with tool use

Use case: Step through a conversation with a reasoning model that can show
chain-of-thought and call a fetch_url tool. Type "done" to exit.

Patterns shown:
  - **Reasoning models (primary)**: Model may return both content (reasoning)
    and tool_calls in the same response; we print content first, then tool
    calls and results. System prompt encourages "Think step by step."
  - **Chat completions with tools**: Same tool-execution loop as chat models;
    one OpenAILog for the session.
  - **Tuning**: Many reasoning models ignore or fix temperature; max_tokens
    is the main lever for length and cost.

Details:
  - Uses a simple fetch_url tool. Reasoning (when present) appears before
    tool call blocks. Session is stateful until you type "done".

Example settings (omit any flag to use API default):

  # Full reasoning chains (complex problems)
  python -m src.understand_reasoning_model --max-tokens 8192

  # Shorter (faster, cheaper)
  python -m src.understand_reasoning_model --max-tokens 2048
  python -m src.understand_reasoning_model --max-tokens 16384

  # Lower temperature if supported
  python -m src.understand_reasoning_model --temperature 0.2 --max-tokens 4096

Usage:
    python -m src.understand_reasoning_model [--model MODEL] [--temperature T] [--max-tokens N]
"""

import argparse
import json
import os
import urllib.request
from urllib.error import URLError

from openai import OpenAI
from utils import create_client, OpenAILog, print_indented

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch the content of a URL. Returns the raw text of the page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"],
        },
    },
}


def fetch_url(url: str) -> dict:
    """Fetch a URL and return its content."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-learn/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"url": url, "status": resp.status, "content": body[:2000]}
    except URLError as e:
        return {"url": url, "error": str(e)}


def run_turn(
    client: OpenAI,
    model: str,
    messages: list,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[list, bool, int, int, object, list]:
    """Run one API turn. Returns (messages, done, prompt_tokens, completion_tokens, response, messages_sent)."""
    messages_sent = list(messages)
    api_kwargs = {
        "model": model,
        "messages": messages,
        "tools": [FETCH_URL_TOOL],
        "tool_choice": "auto",
    }
    if temperature is not None:
        api_kwargs["temperature"] = temperature
    if max_tokens is not None:
        api_kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**api_kwargs)
    msg = response.choices[0].message
    messages.append(msg)

    # Reasoning models may put reasoning in msg.content before tool calls
    if msg.content:
        print("\n  [Assistant]")
        print_indented("  Response", msg.content, indent=4, max_length=4000)

    if msg.tool_calls:
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"\n  [Tool call] {name}")
            print_indented("  Arguments", json.dumps(args, indent=2), indent=4)
            if name == "fetch_url":
                result = fetch_url(args["url"])
            else:
                result = {"error": f"Unknown tool: {name}"}
            result_str = json.dumps(result, indent=2)
            print("  [Tool result]")
            print_indented("  Result", result_str, indent=4, max_length=2000)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        return messages, False, in_tok, out_tok, response, messages_sent

    if msg.content:
        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        return messages, True, in_tok, out_tok, response, messages_sent

    usage = response.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    return messages, False, in_tok, out_tok, response, messages_sent


def main():
    parser = argparse.ArgumentParser(description="Interactive reasoning model chat")
    parser.add_argument("--model", default=None, help="Model to use")
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
    default = "openai/o3-mini" if os.getenv("OPENROUTER_API_KEY") else "o3-mini"
    model = args.model or os.getenv("MODEL", default)
    print(f"Model: {model}")
    if args.temperature is not None or args.max_tokens is not None:
        print(
            f"Overrides: temperature={args.temperature}, max_tokens={args.max_tokens}"
        )
    print('Type "done" to exit.\n')

    system_content = "You are a helpful assistant. You can use fetch_url to get web page content. Think step by step and keep responses concise."
    messages = [{"role": "system", "content": system_content}]
    first_turn = True
    total_in = 0
    total_out = 0
    log = OpenAILog()

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
        print("\n--- New messages this turn ---")
        if first_turn:
            print("[System]")
            print_indented("  ", system_content, indent=4, max_length=2000)
        print("[User]")
        print_indented("  ", prompt, indent=4, max_length=2000)

        done = False
        while not done:
            log.start_call()
            messages, done, in_tok, out_tok, response, messages_sent = run_turn(
                client,
                model,
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            total_in += in_tok
            total_out += out_tok
            log.register(
                "chat.completions.create",
                messages_sent,
                response,
            )
            print(
                f"  [in: {in_tok} out: {out_tok} "
                f"total in: {total_in} total out: {total_out}]"
            )
        first_turn = False

    log.print_summary()
    print("Goodbye.")


if __name__ == "__main__":
    main()
