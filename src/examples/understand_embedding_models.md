# Theory: understand_embedding_models.py

This document describes **what understand_embedding_models.py does** and then walks through an example run (sonnet conversation). Run the example with:

```bash
python -m src.examples.understand_embedding_models [--model CHAT_MODEL] [--embedding-model EMB_MODEL] [--retrieve-after N] [--retrieve-k K] [--temperature T] [--max-tokens N]
```

---

## 1. What the example does

- **Interactive chat** with the same "You:" / "done" loop as the other `understand_*` demos. No tools; the model only returns text.
- **Efficient context**: When the conversation is long enough, the script **does not** send the full history to the chat API. Instead it uses an **embedding model** to pick the **most relevant** past turns and sends only those (plus the current user message). That reduces tokens and keeps the context focused on what matters for the current query.
- **Two models**: An **embedding model** (e.g. `text-embedding-3-small`) produces vectors for similarity; a **chat model** (e.g. `gpt-4o-mini`) does the actual reply. They are separate API calls.
- **Logging**: Each **chat** completion is logged with `OpenAILog` (request messages and response). Embedding calls are not logged.

---

## 2. When we use full history vs retrieval

The script keeps a **full** list of past turns in memory (`full_turns`: user and assistant messages in order). For each new user message it decides:

- **Full history**: If there are fewer than `--retrieve-after` user messages so far, or too few past turns to select from, we send `[system, ...full_turns, current_user]` to the chat API.
- **Retrieval**: Once we have at least `--retrieve-after` user messages and at least one complete exchange (2 messages) in `full_turns`, we build the context via **embedding-based retrieval**: embed the current user message and each past turn, keep the top `--retrieve-k` turns by similarity, and send `[system, ...selected_turns, current_user]`.

Default is `--retrieve-after 2` and `--retrieve-k 6`, so from your **2nd** user message onward the context is built by retrieval (up to 6 most relevant past turns). The system prompt tells the model it "receives only part of the conversation history (the most relevant past turns)."

---

## 3. Embeddings and cosine similarity

**Embedding API:**  
`client.embeddings.create(model=embedding_model, input=text)` or `input=list_of_texts` for a batch. The response gives one vector (list of floats) per input. Same interface as in `eng-dev-patterns/understanding_models.md`.

**Cosine similarity:**  
To rank past turns by relevance to the current message, we compare vectors. The script implements cosine similarity without numpy:

- `cosine_similarity(a, b) = dot(a,b) / (||a|| * ||b||)`
- Values in `[-1, 1]`; higher means more similar. We take the turns whose vectors have the **highest** similarity to the current user message’s embedding.

**Turn representation for embedding:**  
Each "turn" sent to the chat API is a user message plus the following assistant message. For embedding we serialize that pair as one text, e.g. `"User: ...\nAssistant: ..."`, so the vector captures the meaning of the whole exchange.

---

## 4. Building the message list with retrieval

`build_messages_with_retrieval(...)` does the following:

1. **Turn texts**: From `full_turns` (flat list of user/assistant messages), build one string per exchange: `turn_to_text(turns[i]) + "\n" + turn_to_text(turns[i+1])` for each pair. If there’s an odd last message (user without reply), it’s added as a single turn text.
2. **Embed**: Get one embedding for the **current user message** (the "query") and a **batch** embedding for all turn texts.
3. **Similarity**: For each past turn text, compute cosine similarity with the query embedding.
4. **Top-K**: Take the indices of the `retrieve_k` highest similarities (or fewer if there aren’t that many turns).
5. **Chronological order**: Sort those indices so the selected turns stay in **original order** in the prompt (the model sees a coherent timeline).
6. **Message list**: Return `[system, selected_turn_1_user, selected_turn_1_assistant, ..., selected_turn_K_user, selected_turn_K_assistant, current_user]`.

So the chat model never sees the full history when retrieval is used; it only sees the system prompt, the retrieved exchanges, and the latest user message.

---

## 5. One chat turn (no tools)

Each user input triggers **one** call to the chat API. There are no tool definitions and no tool calls. `run_turn()`:

- Sends `messages` (the list built either from full history or from retrieval) to `client.chat.completions.create(model=chat_model, messages=messages, ...)`.
- Appends the assistant’s reply to the list and returns it plus the content and the raw response. The exact list **sent** to the API is what gets passed to `log.register()` for inspection.

After each turn, the script appends the user message and the assistant message to `full_turns`, so the next turn can use them for retrieval if needed.

---

## 6. Logging

- **`log.start_call()`** before each chat completion.
- **`log.register("chat.completions.create", messages_sent, response)`** with the **messages** that were sent (system + selected or full turns + current user), and the API response.

If logging is enabled, each request and response can be inspected (e.g. to see what context was sent). Embedding requests are not logged.

---

## 7. CLI and defaults

| Option | Meaning | Default |
|--------|---------|--------|
| `--model` | Chat model | `gpt-4o-mini` or `openai/gpt-4o-mini` (OpenRouter) |
| `--embedding-model` | Embedding model | `text-embedding-3-small` or `openai/text-embedding-3-small` |
| `--retrieve-after` | Use retrieval once user message count ≥ this | 2 |
| `--retrieve-k` | Max number of past turns to include when using retrieval | 6 |
| `--temperature` / `--max-tokens` | Passed to chat API | API default |

Example: `--retrieve-after 999` effectively disables retrieval (full history every time).

---

## 8. Summary: flow of one user message

| Step | What happens |
|------|----------------|
| 1 | User types a message. |
| 2 | Append it to the in-memory "current turn"; increment user message count. |
| 3 | If retrieval is active: embed current message and all past turn texts; compute similarities; select top `retrieve_k` turns in chronological order; build `messages = [system, ...selected, current_user]`. Else: `messages = [system, ...full_turns, current_user]`. |
| 4 | Call chat API with `messages`; log request and response. |
| 5 | Append user message and assistant reply to `full_turns`. Print reply and token usage. |

---

## 9. Walkthrough: sonnet conversation from logs

This section walks through an example run of **understand_embedding_models**. The conversation was:

1. Ask for rules to write sonnets to win competitions  
2. Ask for an example sonnet per rule  
3. Ask for a sonnet "for the first" (rule)  
4. "write me one for the second"  
5. "for the third"  
6. "for the fourth"  
7. "fifth"  
8. "sixth"  
9. "seventh"  
10. "eigth and 9th"  
11. **"Now give me the rules again. This time, explain them to me as if I was 8 years old"**

Each step below is one **API call** (one request/response pair).

### Step 1 — First user message (no retrieval)

**Exact prompt (only user message):**
```text
Tell me the rules to write sonnets to win competitions
```

**What was sent:**  
`[system, user]` — system prompt plus this single user message. No prior turns, so no retrieval.

**What happened:**  
Model returns the 9 competition sonnet rules (structure, rhyme scheme, meter, theme, volta, imagery, originality, editing, presentation). That reply is appended to conversation history for the next turn.

### Step 2 — Ask for examples per rule

**Exact prompt (current user message):**
```text
For each rule give me an example of a sonnet which exemplifies
```

**What was sent:**  
`[system, user₁, assistant₁, user₂]` — full history so far (one complete exchange plus this new user message). Still below the retrieval threshold, so the whole history is sent.

**What happened:**  
Model returns one example per rule (structure, rhyme scheme, meter, etc.). Response is appended; next turn will have two full exchanges in history.

### Step 3 — Sonnet "for the first"

**Exact prompt:**
```text
using the rules, write me a sonnet for the first
```

**What was sent:**  
Full history: system + turns 1–2 (rules + examples) + this user message.

**What happened:**  
Model writes a full Shakespearean sonnet illustrating the first rule (structure). Same pattern: full context, no retrieval.

### Step 4 — "write me one for the second"

**Exact prompt:**
```text
write me one for the second
```

**What was sent:**  
Full history: system + turns 1–3 + this user message.

**What happened:**  
Model writes a sonnet focused on the rhyme scheme (ABABCDCDEFEFGG).

### Step 5 — "for the third"

**Exact prompt:**
```text
for the third
```

**What was sent:**  
Full history: system + turns 1–4 + this user message.

**What happened:**  
Model writes a sonnet that showcases iambic pentameter (meter).

### Step 6 — "for the fourth"

**Exact prompt:**
```text
for the fourth
```

**What was sent:**  
Full history: system + turns 1–5 + this user message.

**What happened:**  
Model writes a sonnet centered on the theme of love.

### Step 7 — "fifth"

**Exact prompt:**
```text
fifth
```

**What was sent:**  
Full history: system + turns 1–6 + this user message.

**What happened:**  
Model writes a sonnet that highlights the volta (turn). So far every request has sent the full conversation; no retrieval yet.

### Step 8 — "sixth" (retrieval used)

**Exact prompt:**
```text
sixth
```

**What was sent:**  
**Retrieval is used.** The request does **not** contain the first turn (initial "rules" question and answer). The messages sent start with system, then:

- User: "For each rule give me an example of a sonnet which exemplifies"  
- Assistant: (long examples reply)  
- User: "using the rules, write me a sonnet for the first"  
- Assistant: (sonnet)  
- … through the "fifth" exchange …  
- User: "sixth"

So the API sees **6 selected past turns** (by embedding similarity to "sixth") plus the current user message. The **first** exchange (the original rules list) was ranked less relevant and dropped from this request.

**What happened:**  
Model still answers correctly: it returns a sonnet that emphasizes imagery and language (rule 6), using the retrieved context (examples and previous sonnets by rule).

### Step 9 — "seventh"

**Exact prompt:**
```text
seventh
```

**What was sent:**  
Again **retrieval**: 6 most relevant past turns (same kind of subset as in step 8) + this user message. The full 8-turn history is not sent.

**What happened:**  
Model writes a sonnet that demonstrates originality (rule 7).

### Step 10 — "eigth and 9th"

**Exact prompt:**
```text
eigth and 9th
```

**What was sent:**  
**Retrieval**: 6 selected past turns + this user message.

**What happened:**  
Model writes a sonnet that highlights both editing (rule 8) and presentation (rule 9), with a short explanation.

### Step 11 — "Rules again, as if I was 8 years old"

**Exact prompt:**
```text
Now give me the rules again. This time, explain them to me as if I was 8 years old
```

**What was sent:**  
**Retrieval**: 6 past turns (including the original "rules" Q&A and several example/sonnet turns) + this user message. So for this question, the embedding search did include the turn that states the rules, plus other relevant context.

**What happened:**  
Model returns the same 9 rules, rephrased in simple, child-friendly language (e.g. "14 lines," "like a heartbeat," "painting with words," "a fancy word for a turn"). It never saw the full 10-turn history in this call—only the retrieved slice—but the request was specific enough and the retrieved context sufficient to produce a correct, age-appropriate summary.

### Walkthrough summary

| Step | Your exact prompt | Context sent |
|------|--------------------|--------------|
| 1 | Tell me the rules to write sonnets to win competitions | Full (system + user) |
| 2 | For each rule give me an example of a sonnet which exemplifies | Full |
| 3 | using the rules, write me a sonnet for the first | Full |
| 4 | write me one for the second | Full |
| 5 | for the third | Full |
| 6 | for the fourth | Full |
| 7 | fifth | Full |
| 8 | sixth | **Retrieval** (6 turns) |
| 9 | seventh | **Retrieval** (6 turns) |
| 10 | eigth and 9th | **Retrieval** (6 turns) |
| 11 | Now give me the rules again. This time, explain them to me as if I was 8 years old | **Retrieval** (6 turns) |

**Takeaway:**  
From step 8 onward, the app uses embedding-based retrieval: it embeds the current user message and each past turn (user+assistant pair), keeps the top 6 by similarity, and sends only those plus the current message. That keeps prompt size bounded while still giving the model enough context to continue the sonnet-by-rule thread and, at the end, to re-explain the rules in simple terms.
