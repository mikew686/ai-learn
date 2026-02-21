# Theory: understand_llm_models.py

This document walks through **what is happening** in the `understand_llm_models` example: chat completions, tool use, and the turn loop. Run the example with:

```bash
python -m src.scripts.examples.understand_llm_models [--model MODEL] [--temperature T] [--max-tokens N]
```

---

## 1. What the example does

- **Interactive chat** with a single tool: `fetch_url`. You type at "You:", the model can reply with text or call the tool to fetch a URL; you see assistant text, tool calls, and tool results until the model gives a final answer.
- **Stateful session**: every user message and every assistant/tool message is appended to a single `messages` list and sent on each API call.
- **Logging**: each chat completion is recorded with `OpenAILog` (request messages and response) and a summary is printed at the end.

---

## 2. Message list and roles

The API expects a list of **messages** with roles and content. The example builds:

| Role       | Who adds it        | Meaning |
|-----------|--------------------|--------|
| `system`  | You (once at start)| Instructions and persona; the model follows this. |
| `user`    | You (each turn)    | The human’s latest message. |
| `assistant` | API (each turn)  | Model’s reply: either **content** (text) or **tool_calls**. |
| `tool`    | You (after running a tool) | Result of a tool call; must include `tool_call_id` so the API can match it. |

**In the code:**  
`messages` starts as `[system]`. Each turn you append one `user` message, then the loop keeps calling the API and appending `assistant` (and possibly `tool`) messages until the assistant returns **content** instead of (or in addition to) tool calls. So the list grows like:

```
[system] → [system, user] → [system, user, assistant(tool_calls), tool, tool, ...] → [..., assistant(content)] → done
```

The model only ever sees this linear history; it has no separate “memory” of past turns except what’s in `messages`.

---

## 3. Tool definition and tool_choice

**Theory:** The model doesn’t know your code. You describe **what** the tool does (name, description, parameters) in a schema; the model decides **whether** to call it and **with what arguments**.

The example defines one tool:

```python
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
```

- **name**: used in the response as `tool_calls[].function.name`.
- **description**: tells the model when to use the tool; better descriptions lead to better tool use.
- **parameters**: JSON Schema; the model fills in arguments (e.g. `{"url": "https://example.com"}`).

On each request the example passes `tools=[FETCH_URL_TOOL]` and `tool_choice="auto"`:

- **`tool_choice="auto"`**: the model may return text, tool calls, or both; it’s up to the model.
- Other options (e.g. `"none"` or a specific tool name) force or forbid tool use; the example keeps it open.

So **what’s happening**: every completion request includes the full message history **plus** the tool schema and `tool_choice`. The model’s next message is conditioned on that history and on the fact that it *can* call `fetch_url`.

---

## 4. One “turn” from the API’s point of view

A single **chat completion** request is stateless: you send `model`, `messages`, `tools`, `tool_choice`, and optional `temperature` / `max_tokens`. The API returns one **message** (and usage, etc.). That message can contain:

1. **`content`** (text) — the model’s reply to the user.
2. **`tool_calls`** — one or more calls (each has `id`, `function.name`, `function.arguments`).

Chat models usually return **either** content **or** tool_calls in a given turn, but the API allows both. The example checks **tool_calls first**, then content.

**In the code:**  
`run_turn()` sends the current `messages` (and tools, tool_choice) to `client.chat.completions.create()`, then:

- Appends the assistant message to `messages` (so the next request includes it).
- If there are **tool_calls**: for each call it runs the tool (e.g. `fetch_url`), prints the call and result, appends one `tool` message per call (with `tool_call_id` and stringified result), and returns `done=False`.
- If there is **content**: prints it and returns `done=True`.
- If there are neither (edge case): returns `done=False` so the loop can retry or you can handle it.

So one “user turn” in the UI can trigger **multiple** API calls: first call returns tool_calls, you append tool results and call again; repeat until the model returns content.

---

## 5. Tool execution loop (same turn, multiple API calls)

Conceptually:

```
User types: "What's on https://example.com?"
  → messages = [system, user]
  → API call 1 → assistant message with tool_calls: fetch_url("https://example.com")
  → You run fetch_url, append tool result to messages
  → messages = [system, user, assistant(tool_calls), tool]
  → API call 2 → assistant message with content: "The page contains..."
  → done = True, print content, wait for next user input
```

So the **same** user message leads to two (or more) completions. The model’s first reply is “I’ll fetch that”; your code does the fetch and adds a `tool` message; the second reply is the actual answer based on the fetched content. The model never “runs” the tool—your code does. The model only sees the tool **result** in the next request.

**Why a loop:** In general the model might call several tools, or call one tool then decide to call another. The example handles one tool at a time per assistant message: it runs all `tool_calls` in that message, appends all results, then sends one more request. If the model again returns tool_calls, the loop continues.

---

## 6. Logging (OpenAILog)

Each completion is logged so you can inspect requests and responses (e.g. for debugging or cost):

- **`log.start_call()`** — start of one API call.
- **`log.register("chat.completions.create", messages_sent, response)`** — record the exact `messages` sent and the raw `response`.

`messages_sent` is the list passed to the API for *that* call (including system, user, and any prior assistant/tool messages). The summary at exit shows how many calls were made and where the logs were written (e.g. under `data/`). So you can see how the message list grew and what the model returned at each step.

---

## 7. Temperature and max_tokens

- **temperature**: Controls randomness. The example allows `--temperature 0` for deterministic behavior (good for tool use and extraction) or higher values (e.g. 0.7, 1.0) for more variation. If you omit it, the API default is used.
- **max_tokens**: Cap on tokens generated in one completion. The example can override with `--max-tokens`; otherwise the API default applies. Prevents runaway tool-call chains or very long replies.

These are passed through to every `chat.completions.create` call in the turn loop.

---

## 8. Summary: flow of one user message

| Step | Who        | What |
|------|------------|------|
| 1    | User       | Types a message. |
| 2    | Your code  | Appends `{"role": "user", "content": "..."}` to `messages`. |
| 3    | Your code  | Enters inner loop; calls `run_turn(client, model, messages, ...)`. |
| 4    | Your code  | Sends `messages` + `tools` + `tool_choice` to the API. |
| 5    | API        | Returns one message: `content` and/or `tool_calls`. |
| 6    | Your code  | Appends assistant message to `messages`. |
| 7a   | If tool_calls | Run each tool, append each result as `role: "tool"`; go to step 4 (same user turn, next API call). |
| 7b   | If content | Print content, log the call, exit inner loop; wait for next user input (step 1). |

So “one conversation turn” in the UI = one user message plus **one or more** chat completions, depending on how many tool-call rounds the model uses before answering with text.
