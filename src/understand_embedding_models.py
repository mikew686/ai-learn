"""
Interactive demo: Embedding models for efficient conversation context

Use case: Same interactive chat as the other understand_* demos, but when the
conversation gets long we use an embedding model to select the most relevant
past turns instead of sending the full history. Reduces tokens and keeps
context focused on what matters for the current query (see understanding_models.md).

Patterns shown:
  - **Embedding models (primary)**: Generate embeddings for the current user
    message and for each past turn; use cosine similarity to retrieve the
    most relevant turns. Send only those + current message to the chat model.
  - **RAG over conversation history**: Conversation turns are the "documents";
    the current user message is the query; we retrieve top-K turns and use
    them as context for the next chat completion.
  - **Chat completions**: Same interactive loop as understand_llm_models but
    with a context window built via retrieval instead of full history.
  - **Logging**: One OpenAILog for the session (chat only); embedding calls
    are not logged.

Details:
  - No tools; simple multi-turn chat. After --retrieve-after turns we switch
    to retrieval: only the top --retrieve-k turns by similarity (plus current)
    are sent. First N turns always sent in full.
  - Embedding model and chat model are separate (e.g. text-embedding-3-small
    + gpt-4o-mini). See eng-dev-patterns/understanding_models.md (Embedding Models).

Example settings:

  # Default: retrieval after 8 turns, keep 6 most relevant
  python -m src.understand_embedding_models

  # Use retrieval earlier (after 4 turns)
  python -m src.understand_embedding_models --retrieve-after 4 --retrieve-k 4

  # Disable retrieval (send full history every time, like other understand_ demos)
  python -m src.understand_embedding_models --retrieve-after 999

Usage:
    python -m src.understand_embedding_models [--model CHAT_MODEL] [--embedding-model EMB_MODEL] [--retrieve-after N] [--retrieve-k K] [--temperature T] [--max-tokens N]
"""

import argparse
import math
import os
from openai import OpenAI
from utils import create_client, OpenAILog, print_indented


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Uses list, no numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embedding(client: OpenAI, model: str, text: str) -> list[float]:
    """Return embedding vector for a single text."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def get_embeddings_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a list of texts (batch)."""
    if not texts:
        return []
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def turn_to_text(turn: dict) -> str:
    """Serialize a user/assistant turn for embedding."""
    role = turn.get("role", "?")
    content = (turn.get("content") or "").strip()
    return f"{role.capitalize()}: {content}"


def build_messages_with_retrieval(
    client: OpenAI,
    embedding_model: str,
    system_content: str,
    turns: list[dict],
    current_user_message: str,
    retrieve_k: int,
) -> list[dict]:
    """
    Build message list for chat: system + retrieved past turns + current user.
    Past turns are embedded; current user message is the query; we keep the
    retrieve_k most similar turns (in chronological order).
    """
    if not turns:
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": current_user_message},
        ]
    turn_texts = [turn_to_text(turns[i]) + "\n" + turn_to_text(turns[i + 1]) for i in range(0, len(turns) - 1, 2)]
    if len(turns) % 2 == 1:
        turn_texts.append(turn_to_text(turns[-1]))
    query_emb = get_embedding(client, embedding_model, current_user_message)
    turn_embs = get_embeddings_batch(client, embedding_model, turn_texts)
    similarities = [cosine_similarity(query_emb, te) for te in turn_embs]
    num_to_take = min(retrieve_k, len(turn_texts))
    top_indices = sorted(range(len(similarities)), key=lambda i: -similarities[i])[:num_to_take]
    top_indices_sorted = sorted(top_indices)
    selected_turns = []
    for i in top_indices_sorted:
        idx = i * 2
        selected_turns.append(turns[idx])
        if idx + 1 < len(turns):
            selected_turns.append(turns[idx + 1])
    return [
        {"role": "system", "content": system_content},
        *selected_turns,
        {"role": "user", "content": current_user_message},
    ]


def run_turn(
    client: OpenAI,
    chat_model: str,
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[list[dict], str | None, object, list[dict]]:
    """
    One chat completion. Returns (updated_messages, content, response, messages_sent).
    """
    messages_sent = list(messages)
    api_kwargs = {"model": chat_model, "messages": messages_sent}
    if temperature is not None:
        api_kwargs["temperature"] = temperature
    if max_tokens is not None:
        api_kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**api_kwargs)
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    sent_for_log = list(messages_sent)
    messages_sent.append({"role": "assistant", "content": content or ""})
    return messages_sent, content or None, response, sent_for_log


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with embedding-based context selection"
    )
    parser.add_argument("--model", default=None, help="Chat model (e.g. gpt-4o-mini)")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model (e.g. text-embedding-3-small); default depends on API",
    )
    parser.add_argument(
        "--retrieve-after",
        type=int,
        default=2,
        help="Use retrieval once history has this many user messages (default 2)",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=6,
        help="Number of past turns to retrieve when using retrieval (default 6)",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    client = create_client()
    is_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    default_chat = "openai/gpt-4o-mini" if is_openrouter else "gpt-4o-mini"
    default_emb = "openai/text-embedding-3-small" if is_openrouter else "text-embedding-3-small"
    chat_model = args.model or os.getenv("MODEL", default_chat)
    embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", default_emb)

    print(f"Chat model: {chat_model}")
    print(f"Embedding model: {embedding_model}")
    print(f"Retrieval: after {args.retrieve_after} user messages, keep {args.retrieve_k} most relevant turns")
    print("  → With default (2), embedding retrieval is used from your 2nd message so you can see it in action.")
    if args.temperature is not None or args.max_tokens is not None:
        print(f"Overrides: temperature={args.temperature}, max_tokens={args.max_tokens}")
    print('Type "done" to exit.\n')

    system_content = (
        "You are a helpful assistant. Keep responses concise. "
        "You receive only part of the conversation history (the most relevant past turns); answer based on that and the current message."
    )
    full_turns: list[dict] = []
    user_message_count = 0
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

        user_message_count += 1
        print("\n--- New messages this turn ---")
        print("[User]")
        print_indented("  ", prompt, indent=4, max_length=2000)

        use_retrieval = len(full_turns) >= 2 and user_message_count >= args.retrieve_after
        if use_retrieval:
            messages = build_messages_with_retrieval(
                client,
                embedding_model,
                system_content,
                full_turns,
                prompt,
                args.retrieve_k,
            )
            print("  [Context: retrieval used — most relevant past turns included]")
        else:
            messages = [
                {"role": "system", "content": system_content},
                *full_turns,
                {"role": "user", "content": prompt},
            ]

        log.start_call()
        updated, content, response, messages_sent = run_turn(
            client,
            chat_model,
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        log.register("chat.completions.create", messages_sent, response)

        full_turns.append({"role": "user", "content": prompt})
        if content:
            print("\n  [Assistant]")
            print_indented("  Response", content, indent=4, max_length=4000)
            full_turns.append({"role": "assistant", "content": content})
        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        print(f"  [in: {in_tok} out: {out_tok}]")

    log.print_summary()
    print("Goodbye.")


if __name__ == "__main__":
    main()
