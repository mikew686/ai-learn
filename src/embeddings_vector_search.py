"""
Pattern 7: Embeddings / Vector Search

Use case: Interactive translator that asks once for target language (with optional
region or dialect), normalizes it via one LLM call to canonical language + region,
then translates phrases with notes. Uses a local SQLite-backed vector store to
retrieve similar past translations for few-shot prompting (semantic search).

Patterns shown:
  - **Embedding generation**: OpenAI embeddings API for source phrases.
  - **Vector similarity**: Cosine similarity (numpy) to find similar stored examples.
  - **Local vector storage**: SQLite table stores (source, translation, notes, embedding blob).
  - **Dynamic few-shot**: Top-K similar translations for the current target/dialect
    are injected into the prompt so the model can mimic style and reuse phrasing.
  - **Translation with notes**: Each result includes translation and brief
    cultural/contextual notes (like schema_driven_translation).

Details:
  - DB path: data/embeddings_vector_search.db (ephemeral; create data/ if missing).
  - Embeddings: text-embedding-3-small (or OPENROUTER default). Dimension stored per row.
  - No tools; single chat completion per phrase with optional structured output.

See eng-dev-patterns/learning_progression.md (Pattern 7) and understanding_models.md.

Usage:
    python -m src.embeddings_vector_search [--model MODEL] [--embedding-model EMB_MODEL] [--db PATH] [--top-k N] [--temperature T] [--max-tokens N]
"""

import argparse
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field

from utils import create_client, OpenAILog, print_indented

DEFAULT_DB_PATH = "data/embeddings_vector_search.db"


class TranslationWithNotes(BaseModel):
    """Translation plus brief notes (cultural/contextual)."""

    translated_text: str = Field(description="The translated text in the target language and dialect")
    notes: str = Field(
        description="Brief cultural or contextual notes about the translation (in English)"
    )


class CanonicalLanguageRegion(BaseModel):
    """
    Canonical language and optional region from a single user phrase.
    Uses standard identifiers (ISO 639-1, ISO 3166-1) like schema_driven_translation
    and language_assessment_structured_output.
    """

    language_name: str = Field(
        description="Canonical language name in English (e.g. French, Spanish, German)"
    )
    language_code: str = Field(
        description="ISO 639-1 two-letter language code (e.g. en, fr, es, de)"
    )
    region_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 region code when a region or dialect is specified (e.g. US, CA, MX, FR, GB); empty string if none",
    )


def parse_language_region(
    client: OpenAI,
    chat_model: str,
    user_input: str,
    *,
    log: OpenAILog | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str, str]:
    """
    Parse a single user phrase (e.g. "french", "French (Quebec)", "spanish mexico")
    into canonical language name, ISO 639-1 language code, and optional ISO 3166-1
    region code via one LLM request.
    Returns (language_name, language_code, region_code); region_code is "" if none.
    """
    if not (user_input or "").strip():
        return "English", "en", ""
    messages = [
        {
            "role": "system",
            "content": (
                "Normalize the user's translation target into standard identifiers. "
                "Return: (1) canonical language name in English (e.g. French, Spanish), "
                "(2) ISO 639-1 two-letter language code (e.g. fr, es, en), "
                "(3) ISO 3166-1 alpha-2 region code if a region or dialect was specified (e.g. CA for Quebec/Canada, MX for Mexico, FR for France, GB for UK); otherwise empty string."
            ),
        },
        {
            "role": "user",
            "content": f"Normalize this translation target: {user_input.strip()}",
        },
    ]
    kwargs = {"model": chat_model, "messages": messages, "response_format": CanonicalLanguageRegion}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if log:
        log.start_call()
    response = client.beta.chat.completions.parse(**kwargs)
    if log:
        log.register("beta.chat.completions.parse", messages, response)
    parsed = response.choices[0].message.parsed
    name = (parsed.language_name or "English").strip()
    code = (parsed.language_code or "en").strip().lower()
    region = (parsed.region_code or "").strip().upper()
    return name, code, region


def get_embedding(client: OpenAI, model: str, text: str) -> list[float]:
    """Return embedding vector for a single text."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def cosine_similarities(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Return cosine similarity between query (1D) and each row of vectors (2D)."""
    if vectors.size == 0:
        return np.array([])
    q = query.astype(np.float32).reshape(-1)
    norms_q = np.linalg.norm(q)
    if norms_q == 0:
        return np.zeros(len(vectors))
    norms_v = np.linalg.norm(vectors, axis=1)
    norms_v = np.where(norms_v == 0, 1e-10, norms_v)
    return (vectors @ q) / (norms_v * norms_q)


def _print_db_summary(conn: sqlite3.Connection) -> None:
    """Print summary of current database state (record counts by language)."""
    total = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    print()
    print("--- Vector store summary ---")
    print(f"  Total records: {total}")
    if total > 0:
        rows = conn.execute(
            """
            SELECT target_language, COUNT(*) AS n
            FROM translations
            GROUP BY target_language
            ORDER BY target_language
            """
        ).fetchall()
        for lang, n in rows:
            print(f"    {lang}: {n}")
    print("-----------------------------")
    print()


def init_db(conn: sqlite3.Connection) -> None:
    """Create translations table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_language TEXT NOT NULL,
            target_dialect TEXT NOT NULL,
            source_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            notes TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


def store_translation(
    conn: sqlite3.Connection,
    target_language: str,
    target_dialect: str,
    source_text: str,
    translated_text: str,
    notes: str,
    embedding: list[float],
) -> None:
    """Insert one translation and its embedding into the DB."""
    arr = np.array(embedding, dtype=np.float32)
    blob = arr.tobytes()
    dim = len(embedding)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO translations
        (target_language, target_dialect, source_text, translated_text, notes, embedding, embedding_dim, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (target_language, target_dialect, source_text, translated_text, notes, blob, dim, now),
    )
    conn.commit()


def _normalize_lang(s: str) -> str:
    """Normalize language/dialect for comparison (lowercase, strip)."""
    return (s or "").strip().lower()


def language_dialect_weight(
    row_lang: str,
    row_dialect: str,
    user_lang: str,
    user_dialect: str,
) -> float:
    """
    Weight for how well a stored (language, dialect) matches the user's choice.
    - Exact match (same language, same dialect): 1.0
    - Same language, dialect differs or one empty: 0.6 (e.g. French vs French Quebec)
    - Different language: 0.15 (so French ranks much higher than German)
    """
    r_l = _normalize_lang(row_lang)
    r_d = _normalize_lang(row_dialect)
    u_l = _normalize_lang(user_lang)
    u_d = _normalize_lang(user_dialect)
    if r_l != u_l:
        return 0.15
    if r_d == u_d:
        return 1.0
    if not r_d or not u_d:
        return 0.6
    return 0.5


def get_similar_translations(
    conn: sqlite3.Connection,
    target_language: str,
    target_dialect: str,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[str, str, str]]:
    """
    Return up to top_k past translations (source_text, translated_text, notes)
    ranked by phrase similarity weighted by language/dialect match. Same language
    (e.g. French) ranks higher than another language (e.g. German); exact
    language+dialect match ranks highest.
    """
    cursor = conn.execute(
        """
        SELECT target_language, target_dialect, source_text, translated_text, notes, embedding, embedding_dim
        FROM translations
        """
    )
    rows = cursor.fetchall()
    if not rows:
        return []

    query = np.array(query_embedding, dtype=np.float32)
    dim = len(query)
    # Keep only rows with matching embedding dimension
    rows = [r for r in rows if r[6] == dim]
    if not rows:
        return []

    blobs = [r[5] for r in rows]
    vectors = np.array([np.frombuffer(b, dtype=np.float32) for b in blobs])
    phrase_sims = cosine_similarities(query, vectors)

    user_lang = target_language
    user_dialect = target_dialect
    scores = []
    for i, r in enumerate(rows):
        w = language_dialect_weight(r[0], r[1], user_lang, user_dialect)
        scores.append(phrase_sims[i] * w)

    scores = np.array(scores)
    indices = np.argsort(-scores)[:top_k]
    return [(rows[i][2], rows[i][3], rows[i][4]) for i in indices]


def translate_phrase(
    client: OpenAI,
    chat_model: str,
    target_language: str,
    target_dialect: str,
    source_text: str,
    few_shot_examples: list[tuple[str, str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[TranslationWithNotes, object, list]:
    """
    Call chat API to translate source_text with optional few-shot examples.
    Returns (parsed result, raw response, messages sent) for logging.
    """
    system = (
        f"You are a translator into {target_language}"
        + (f" ({target_dialect})" if target_dialect else "")
        + ". For each phrase, provide the translation and brief cultural or contextual notes in English."
    )
    messages = [{"role": "system", "content": system}]

    tgt_label = f"Translate to {target_language}" + (f" ({target_dialect})" if target_dialect else "")

    for src, trans, notes in few_shot_examples:
        messages.append({"role": "user", "content": f"{tgt_label}: {src}"})
        messages.append({"role": "assistant", "content": f"Translation: {trans}\nNotes: {notes}"})

    messages.append({"role": "user", "content": f"{tgt_label}: {source_text}"})

    kwargs = {"model": chat_model, "messages": messages, "response_format": TranslationWithNotes}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.beta.chat.completions.parse(**kwargs)
    parsed = response.choices[0].message.parsed
    return parsed, response, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive translator with embedding-backed few-shot (vector search)"
    )
    parser.add_argument("--model", default=None, help="Chat model (e.g. gpt-4o-mini)")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model (e.g. text-embedding-3-small); default from API",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"SQLite DB path for vector store (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of similar past translations to use as few-shot (default: 3)",
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

    db_path = args.db
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)

    print("Translator with vector-backed few-shot (Pattern 7: Embeddings / Vector Search)")
    print(f"Chat model: {chat_model}")
    print(f"Embedding model: {embedding_model}")
    print(f"Vector store: {db_path}")
    print(f"Few-shot: up to {args.top_k} similar past translations per phrase")
    if args.temperature is not None or args.max_tokens is not None:
        print(f"Overrides: temperature={args.temperature}, max_tokens={args.max_tokens}")
    print()

    log = OpenAILog()
    raw_target = input(
        "Target language (with optional region or dialect, e.g. Spanish, French (Quebec), German): "
    ).strip()
    target_language, target_language_code, target_dialect = parse_language_region(
        client,
        chat_model,
        raw_target or "Spanish",
        log=log,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    # target_dialect holds ISO 3166-1 region code when present (e.g. CA, MX)
    # target_language_code is ISO 639-1 (e.g. fr, es) for reference; DB stores name + region_code

    print()
    print(
        f"Translating into: {target_language} ({target_language_code})"
        + (f" [{target_dialect}]" if target_dialect else "")
    )
    print('Enter a phrase to translate, or "done" to exit.')
    print()

    while True:
        try:
            phrase = input("Phrase: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not phrase:
            continue
        if phrase.lower() == "done":
            break

        # Embed and retrieve similar examples
        query_embedding = get_embedding(client, embedding_model, phrase)
        similar = get_similar_translations(
            conn, target_language, target_dialect, query_embedding, args.top_k
        )

        log.start_call()
        try:
            result, response, messages_sent = translate_phrase(
                client,
                chat_model,
                target_language,
                target_dialect,
                phrase,
                similar,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            log.print_summary()
            raise
        log.register("beta.chat.completions.parse", messages_sent, response)

        store_translation(
            conn,
            target_language,
            target_dialect,
            phrase,
            result.translated_text,
            result.notes,
            query_embedding,
        )

        if similar:
            print("  [Few-shot: used", len(similar), "similar past translation(s)]")
        print_indented("  Translation", result.translated_text, indent=4, max_length=2000)
        print_indented("  Notes", result.notes, indent=4, max_length=1000)
        print()

    _print_db_summary(conn)
    conn.close()
    log.print_summary()
    print("Goodbye.")


if __name__ == "__main__":
    main()
