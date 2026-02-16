# Agent guide: learning example scripts

This directory contains example scripts that demonstrate LLM/reasoning patterns (prompts, tool use, structured output, etc.). Follow these practices so examples stay consistent and easy to compare.

## Shared utilities (`utils.py`)

- **`create_client()`** – Builds an `OpenAI` client. Uses OpenRouter when `OPENROUTER_API_KEY` is set, otherwise OpenAI. Do not construct `OpenAI()` directly in examples.
- **`OpenAILog`** – Collects and formats every OpenAI request/response for the run. See “OpenAILog” below. By default, each raw request/response is written to `data/YYYYMMDD/<timestamp>-openai-response.json` (one JSON file per registered call). Pass `log_raw_dir=None` to disable.
- **`print_indented(label, content, indent=2, width=120, max_length=None)`** – Use for user-facing content (phrases, translations, assessments) so output is wrapped and indented consistently.
- **`format_response(content)`** – Prefixes each line with `>> ` for simple response display.

Use these instead of ad-hoc printing or client creation.

## OpenAILog

### One log per message train

- **One `OpenAILog()` per logical “run”** (one conversation, one example, or one mode). Do not reuse a single log across separate message trains or mixed models.
- Create the log at the start of that run, call `log.register(...)` for every API call in that run, then call **`log.print_summary()`** at the end of that run.
- Examples: `system_prompt_example` uses one log for “Example A” (stateless) and another for “Example B” (stateful). Interactive scripts (`understand_llm_models`, `understand_reasoning_model`) use one log for the whole session.

### Register every API call

After each `client.chat.completions.create(...)` or `client.beta.chat.completions.parse(...)`:

1. Call **`log.start_call()`** immediately before the API call (it sets an internal timestamp).
2. After the call, call:
   ```python
   log.register(
       "chat.completions.create",   # or "beta.chat.completions.parse"
       messages,                    # exact list passed to the API (for this request)
       response,
       label="Short label",         # optional, e.g. "Initial (with tools)", "Final"
   )
   ```
   Elapsed time is computed inside the logger when `register()` runs.
3. Use the **exact `messages` list** that was sent in that request (e.g. the list before you append the assistant message).

Model is read from `response.model`; do not pass it into `register`.

### Nested or helper API calls

If a helper (e.g. `assess_lang`) performs an API call as part of the same logical run (e.g. inside a tool), pass the **same** `log` into that helper so all requests in that run appear in one summary. Make the log parameter optional (`log: OpenAILog | None = None`) if the helper is sometimes used without logging.

## Request types

- **`chat.completions.create`** – Standard chat completion (and tool-calling flows that use `create`).
- **`beta.chat.completions.parse`** – When using `client.beta.chat.completions.parse(...)` with a Pydantic `response_format`.

Use the string that matches how you called the API.

## Raw response logging to file

**`OpenAILog`** writes each registered request/response to a JSON file by default under **`data/`**. Pass **`log_raw_dir=None`** to disable, or another path to change the directory. Files are written as:

- Path: `log_raw_dir/YYYYMMDD/<unix_ms>-openai-response.json` (e.g. `data/20250207/1736265318220-openai-response.json`). Unix milliseconds so filenames sort alphabetically by time.
- Each file has two top-level keys: **`meta`** (added by our logger: `request_type`, `label`, `elapsed_time`, `messages`) and **`response`** (raw SDK/API response). Use `meta` to distinguish logger fields from API data. The `data/` directory is in `.gitignore`.

## CLI and environment

- Use **argparse** for flags. Common flags:
  - `--model` – Override model (otherwise use `MODEL` env or a default).
  - `--temperature`, `--max-tokens` – Pass through when relevant; use `None` to mean “API default”.
- **Model resolution order:** `args.model` → `os.getenv("MODEL")` → provider-specific default (e.g. OpenRouter vs OpenAI).
- **Client:** `client = create_client()` (no API key in code; use `OPENAI_API_KEY` or `OPENROUTER_API_KEY`).
- In `main()`, print which API is used (“Using OpenRouter API” / “Using OpenAI API”) and the chosen model.

## Output consistency

- **API details** – Always go through `OpenAILog`: request type, model (from response), response type (text/schema/tool_calls), message summary, response summary (with tool-call details), token usage and running total, response time. Then `log.print_summary()` for that run.
- **User-facing content** – Use `print_indented()` for phrases, translations, assessment fields, tool results, etc., so layout and width match across examples.
- Avoid duplicate token/timing prints; the log already shows them per request and in the summary.

## File and structure

- **Module docstring** – State which patterns the example demonstrates (examples may show more than one); list them explicitly. Include use case. Include a short “Usage” or “Example settings” section if helpful.
- **Run as module** – Scripts are run as `python -m src.examples.<script_name>`. Imports from this directory use `from utils import ...` (no `src.examples.` prefix when inside `src/examples/`).
- **main()** – Parse args, create client, resolve model, then call one or more example functions. Each example that is a separate message train should create its own `OpenAILog`, register its calls, and call `print_summary()`.

## Example skeleton

```python
"""Pattern N: <Name>. Use case: <short description>."""

import argparse
import os
import time
from openai import OpenAI
from utils import create_client, OpenAILog, print_indented

def run_example(client: OpenAI, model: str, *, temperature=None, max_tokens=None):
    log = OpenAILog()
    messages = [{"role": "user", "content": "..."}]
    log.start_call()
    response = client.chat.completions.create(model=model, messages=messages, ...)
    log.register("chat.completions.create", messages, response, label="...")
    print_indented("Result", response.choices[0].message.content)
    log.print_summary()

def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--model", ...)
    args = parser.parse_args()
    client = create_client()
    model = args.model or os.getenv("MODEL", "default-for-provider")
    run_example(client, model, ...)

if __name__ == "__main__":
    main()
```

Clearly document which patterns each example shows (they may combine several). Use the shared utils and one log per message train, and register every API call so output stays consistent across the repo.
