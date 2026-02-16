# Theory: understand_reasoning_model.py

This document walks through **what is happening** in the `understand_reasoning_model` example: chat completions with a **reasoning model** (e.g. o1, o3-mini) that can return both **reasoning text** and **tool calls** in the same response. Run the example with:

```bash
python -m src.examples.understand_reasoning_model [--model MODEL] [--temperature T] [--max-tokens N]
```

---

## 1. What the example does

- **Interactive chat** with a reasoning model and a single tool: `fetch_url`. Same "You:" / "done" loop as `understand_llm_models`.
- **Reasoning in the reply**: The model may return **both** `content` (its step-by-step thinking) **and** `tool_calls` in one message. The script prints the **content first** (so you see the reasoning), then runs the tool(s) and appends results. So you see: reasoning block → tool call(s) → tool result(s) → then the next API call uses that context.
- **Tool loop**: Same pattern as chat models: if the response has `tool_calls`, we run the tools, append `tool` messages, and call the API again until the model returns a turn with no tool calls (final answer or reasoning-only).
- **System prompt**: Encourages "Think step by step" so the model uses its reasoning capability.
- **Logging**: One `OpenAILog` for the session; each chat completion is registered (request messages and response).

---

## 2. How it differs from understand_llm_models

| Aspect | understand_llm_models | understand_reasoning_model |
|--------|------------------------|----------------------------|
| Model type | Chat (e.g. gpt-4o-mini) | Reasoning (e.g. o3-mini, o1) |
| Typical response | Either `content` **or** `tool_calls` | Often **both** `content` and `tool_calls` in one message |
| Order of handling | Script checks `tool_calls` first, then `content` | Script handles **content first** (print reasoning), then `tool_calls` |
| System prompt | "You can use fetch_url..." | "Think step by step and keep responses concise" + fetch_url |
| Default model | gpt-4o-mini | o3-mini |
| Temperature | Often used | Many reasoning models ignore or fix it; `max_tokens` is the main lever |

The **tool schema and execution loop** are the same (same `fetch_url` tool, same append-tool-results-and-call-again loop). The important difference is that reasoning models are built to show intermediate steps, so the API often puts that reasoning in `content` and may add `tool_calls` in the **same** message.

---

## 3. Message list and roles

Same as the chat example: **system**, **user**, **assistant**, **tool**. The assistant message can contain:

- **`content`** — reasoning and/or final answer (the model’s “chain of thought”).
- **`tool_calls`** — one or more tool invocations (same structure as chat models).

The script appends the full assistant message (including both content and tool_calls) to `messages`, then appends each tool result as a `tool` message. So the conversation history stays a single linear list.

---

## 4. Tool definition

Identical to `understand_llm_models`: one tool, `fetch_url`, with a URL parameter. The request includes `tools=[FETCH_URL_TOOL]` and `tool_choice="auto"`. The reasoning model decides when to call it (e.g. after reasoning that it needs to fetch a page).

---

## 5. One turn from the API’s point of view

A single **chat completion** request is stateless. The API returns one **message**, which can contain:

1. **`content`** — text (reasoning and/or answer).
2. **`tool_calls`** — one or more tool calls.

**Reasoning models often return both in the same message**: e.g. a block of reasoning in `content` (“I need to check the current weather, so I’ll call…”) and one or more entries in `tool_calls`. So one “turn” from the API can be: reasoning text + tool calls.

**In the code:**  
`run_turn()`:

- Appends the assistant message to `messages`.
- **If there is `content`**: prints it first (under `[Assistant]`). That way you see the reasoning before any tool output.
- **If there are `tool_calls`**: for each call, runs the tool, prints call and result, appends a `tool` message; then returns `done=False` so the loop sends another request.
- **If there is `content` and no tool calls**: returns `done=True` (final answer).
- **If there are neither**: returns `done=False` (edge case).

So the **order** of handling is: print content (reasoning), then process tool_calls. That matches the idea that the model “thinks” then “acts.”

---

## 6. Tool execution loop (same turn, multiple API calls)

Conceptually the same as the chat example:

```
User: "What's on https://example.com? Summarize in one sentence."
  → API call 1 → assistant message: content = "I'll fetch the page first.", tool_calls = [fetch_url(...)]
  → Script prints the reasoning (content), runs fetch_url, appends tool result
  → API call 2 → assistant message: content = "The page contains...", no tool_calls
  → done = True
```

So one user message can trigger multiple API calls. The first response might be reasoning + tool call; the second might be the final summary. The model never runs the tool—your code does; the model only sees the tool result in the next request.

---

## 7. System prompt and tuning

- **System prompt**: “You are a helpful assistant. You can use fetch_url to get web page content. Think step by step and keep responses concise.” So the model is nudged to show reasoning and to use the tool when needed.
- **Temperature**: Many reasoning models (e.g. o1-series) ignore or fix temperature. The script still passes `--temperature` if you set it; behavior depends on the model.
- **max_tokens**: This is the main lever for length and cost. Reasoning models often produce long chains of thought; `--max-tokens` caps the size of each completion. Example: `--max-tokens 8192` for long reasoning, `--max-tokens 2048` for shorter, faster replies.

---

## 8. Logging

Same pattern as the other understand_* demos:

- **`log.start_call()`** before each chat completion.
- **`log.register("chat.completions.create", messages_sent, response)`** with the exact `messages` sent and the raw response.

So you can inspect what context was sent and what the model returned (including reasoning in `content` and any `tool_calls`). Logs are written under `data/` when `OpenAILog` is configured with a log directory.

---

## 9. Summary: flow of one user message

| Step | Who        | What |
|------|------------|------|
| 1    | User       | Types a message. |
| 2    | Your code  | Appends `{"role": "user", "content": "..."}` to `messages`. |
| 3    | Your code  | Enters inner loop; calls `run_turn(...)`. |
| 4    | Your code  | Sends `messages` + `tools` + `tool_choice` to the API. |
| 5    | API        | Returns one message: may have `content`, `tool_calls`, or both. |
| 6    | Your code  | Appends assistant message to `messages`. |
| 7    | Your code  | If there is `content`, prints it first (reasoning). |
| 8a   | If tool_calls | Run each tool, append each result as `role: "tool"`; go to step 4 (same user turn, next API call). |
| 8b   | If no tool_calls | Log the call, exit inner loop; wait for next user input (step 1). |

So “one conversation turn” in the UI = one user message plus **one or more** chat completions. With reasoning models you often see **content then tool_calls** in a single response, then another response with the final answer after tool results are added.
