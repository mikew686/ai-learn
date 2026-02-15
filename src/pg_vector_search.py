"""
Pattern 6: Embeddings / Vector Search (Postgres + pgvector)

Same approach as embeddings_vector_search but using a Postgres database with
the pgvector extension for vector similarity (cosine distance in-database).

Uses db.utils.PgVectorHelper for database creation and connections.
See pg_user.md and postgres_pgvector_notes.md for setup.

Embedding approach (dialect-in-embedding):
  - We embed a single string that includes both the source phrase and the target
    language/dialect, using a BCP-47-style tag: "[<lang>-<region>] <phrase>" when
    a dialect is set (e.g. "[fr-CA] comment ça va"), or "[<lang>] <phrase>" when no
    dialect (e.g. "[fr] comment ça va"). This is used for both the query and the
    stored embedding.
  - Retrieval filters by target_language only (no dialect filter). Ranking is
    purely by cosine distance. Same-dialect examples naturally rank higher because
    their vectors encode the same dialect tag; no separate dialect weighting is
    applied. The database schema is unchanged (target_dialect remains a column
    for display and analytics).

Usage:
    python -m src.pg_vector_search [--model MODEL] [--embedding-model EMB_MODEL] [--db-url URL] [--top-k N] [--temperature T] [--max-tokens N]
"""

import argparse
import os

from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import text

from db.utils import PgVectorHelper
from pgvector import Vector
from utils import create_client, OpenAILog, print_indented

# Default embedding dimension (text-embedding-3-small)
DEFAULT_EMBEDDING_DIM = 1536


class TranslationWithNotes(BaseModel):
    """Translation plus brief notes (cultural/contextual)."""

    translated_text: str = Field(
        description="The translated text in the target language and dialect"
    )
    notes: str = Field(
        description="Brief cultural or contextual notes about the translation (in English)"
    )


class CanonicalLanguageRegion(BaseModel):
    """
    Canonical language and optional region from a single user phrase.
    Uses standard identifiers (ISO 639-1, ISO 3166-1) like schema_driven_translation.
    """

    language_name: str = Field(
        description="Canonical language name in English (e.g. French, Spanish, German)"
    )
    language_code: str = Field(
        description="ISO 639-1 two-letter language code (e.g. en, fr, es, de)"
    )
    region_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 region code when a region or dialect is specified; empty string if none",
    )
    target_description: str = Field(
        default="",
        description="Short human-readable description of the language and region/dialect for prompts and display",
    )


def parse_language_region(
    client: OpenAI,
    chat_model: str,
    user_input: str,
    *,
    log: OpenAILog | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str, str, str]:
    """Parse user phrase into (language_name, language_code, region_code, target_description)."""
    if not (user_input or "").strip():
        return "English", "en", "", "English"
    messages = [
        {
            "role": "system",
            "content": (
                "Normalize the user's translation target into standard identifiers and a human-readable description. "
                "Return: (1) canonical language name in English, "
                "(2) ISO 639-1 two-letter language code, "
                "(3) ISO 3166-1 alpha-2 region code if a region or dialect was specified; otherwise empty string. "
                "(4) target_description: a short human-readable description of the language and region/dialect."
            ),
        },
        {"role": "user", "content": f"Normalize this translation target: {user_input.strip()}"},
    ]
    kwargs = {
        "model": chat_model,
        "messages": messages,
        "response_format": CanonicalLanguageRegion,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if log:
        log.start_call()
    response = client.beta.chat.completions.parse(**kwargs)
    parsed = response.choices[0].message.parsed
    name = (parsed.language_name or "English").strip()
    code = (parsed.language_code or "en").strip().lower()
    region = (parsed.region_code or "").strip().upper()
    description = (parsed.target_description or "").strip() or name
    return name, code, region, description


def get_embedding(
    client: OpenAI,
    model: str,
    text_input: str,
    *,
    log: OpenAILog | None = None,
) -> list[float]:
    """Return embedding vector for a single text."""
    if log:
        log.start_call()
    response = client.embeddings.create(model=model, input=text_input)
    if log:
        log.register(
            "embeddings.create",
            [{"model": model, "input": text_input[:2000] + ("..." if len(text_input) > 2000 else "")}],
            response,
        )
    return response.data[0].embedding


def _embedding_text(phrase: str, language_code: str, dialect: str) -> str:
    """Build the string to embed: includes language/dialect tag so retrieval prefers same dialect."""
    tag = f"{language_code}-{dialect}" if dialect else language_code
    return f"[{tag}] {phrase}"


def _print_db_summary(helper: PgVectorHelper) -> None:
    """Print summary of current database state (record counts by language and dialect)."""
    with helper.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM translations")).scalar() or 0
        print()
        print("--- Vector store summary (Postgres + pgvector) ---")
        print(f"  Total records: {total}")
        if total and total > 0:
            rows = conn.execute(
                text(
                    """
                    SELECT target_language, target_dialect, COUNT(*) AS n
                    FROM translations
                    GROUP BY target_language, target_dialect
                    ORDER BY target_language, target_dialect
                    """
                )
            ).fetchall()
            prev_lang: str | None = None
            for lang, dialect, n in rows:
                dialect_label = dialect if dialect else "(no dialect)"
                if lang != prev_lang:
                    print(f"  {lang}:")
                    prev_lang = lang
                print(f"    {dialect_label}: {n}")
        print("--------------------------------------------------")
        print()


def store_translation(
    helper: PgVectorHelper,
    target_language: str,
    target_dialect: str,
    source_text: str,
    translated_text: str,
    notes: str,
    embedding: list[float],
) -> None:
    """Insert one translation and its embedding into the DB."""
    with helper.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO translations
                (target_language, target_dialect, source_text, translated_text, notes, embedding)
                VALUES (:target_language, :target_dialect, :source_text, :translated_text, :notes, :embedding)
                """
            ),
            {
                "target_language": target_language,
                "target_dialect": target_dialect,
                "source_text": source_text,
                "translated_text": translated_text,
                "notes": notes,
                "embedding": Vector(embedding),
            },
        )
        conn.commit()


def get_similar_translations(
    helper: PgVectorHelper,
    target_language: str,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[str, str, str]]:
    """
    Return up to top_k past translations (source_text, translated_text, notes)
    for the given target language, ranked by pgvector cosine distance only.
    Dialect is encoded in the embedding, so same-dialect examples naturally
    rank higher; no separate dialect filter or weight.
    """
    with helper.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT source_text, translated_text, notes
                FROM translations
                WHERE target_language = :lang
                ORDER BY embedding <=> :query
                LIMIT :limit
                """
            ),
            {"lang": target_language, "query": Vector(query_embedding), "limit": top_k},
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def translate_phrase(
    client: OpenAI,
    chat_model: str,
    target_language: str,
    target_dialect: str,
    source_text: str,
    few_shot_examples: list[tuple[str, str, str]],
    *,
    target_description: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[TranslationWithNotes, object, list]:
    """Call chat API to translate source_text with optional few-shot examples."""
    tgt_spec = (target_description or target_language).strip()
    if not tgt_spec:
        tgt_spec = target_language + (f" ({target_dialect})" if target_dialect else "")
    system = (
        f"You are a translator into {tgt_spec}. "
        "For each phrase, provide the translation and brief cultural or contextual notes in English."
    )
    messages = [{"role": "system", "content": system}]
    tgt_label = f"Translate to {tgt_spec}"
    for src, trans, notes in few_shot_examples:
        messages.append({"role": "user", "content": f"{tgt_label}: {src}"})
        messages.append({"role": "assistant", "content": f"Translation: {trans}\nNotes: {notes}"})
    messages.append({"role": "user", "content": f"{tgt_label}: {source_text}"})

    kwargs = {
        "model": chat_model,
        "messages": messages,
        "response_format": TranslationWithNotes,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.beta.chat.completions.parse(**kwargs)
    parsed = response.choices[0].message.parsed
    return parsed, response, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive translator with Postgres/pgvector-backed few-shot (vector search)"
    )
    parser.add_argument("--model", default=None, help="Chat model (e.g. gpt-4o-mini)")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model (e.g. text-embedding-3-small); default from API",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Postgres URL (default: DATABASE_URL or postgresql://$USER:vectorfun@localhost:5432/$USER)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of similar past translations to use as few-shot (default: 4)",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    client = create_client()
    is_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    default_chat = "openai/gpt-4.1-mini" if is_openrouter else "gpt-4.1-mini"
    default_emb = (
        "openai/text-embedding-3-small" if is_openrouter else "text-embedding-3-small"
    )
    chat_model = args.model or os.getenv("MODEL", default_chat)
    embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", default_emb)

    helper = PgVectorHelper(url=args.db_url)
    helper.init_schema(embedding_dim=DEFAULT_EMBEDDING_DIM)

    print("Translator with Postgres/pgvector-backed few-shot (Pattern 6)")
    print(f"Chat model: {chat_model}")
    print(f"Embedding model: {embedding_model}")
    print(f"Database: Postgres + pgvector")
    print(f"Few-shot: up to {args.top_k} similar past translations per phrase")
    if args.temperature is not None or args.max_tokens is not None:
        print(f"Overrides: temperature={args.temperature}, max_tokens={args.max_tokens}")
    print()

    log_language = OpenAILog(log_raw_dir="data", description="language")
    log_embeddings = OpenAILog(log_raw_dir="data", description="embeddings")
    log_translations = OpenAILog(log_raw_dir="data", description="translations")

    raw_target = input(
        "Target language (with optional region or dialect, e.g. Spanish, French (Quebec), German): "
    ).strip()
    target_language, target_language_code, target_dialect, target_description = (
        parse_language_region(
            client,
            chat_model,
            raw_target or "Spanish",
            log=log_language,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
    print()
    print(
        f"Translating into: {target_description or target_language} "
        f"(language: {target_language}, code: {target_language_code}"
        + (f", region: {target_dialect}" if target_dialect else "")
        + ")"
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

        text_to_embed = _embedding_text(phrase, target_language_code, target_dialect)
        query_embedding = get_embedding(
            client, embedding_model, text_to_embed, log=log_embeddings
        )
        similar = get_similar_translations(
            helper, target_language, query_embedding, args.top_k
        )

        log_translations.start_call()
        try:
            result, response, messages_sent = translate_phrase(
                client,
                chat_model,
                target_language,
                target_dialect,
                phrase,
                similar,
                target_description=target_description,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception:
            log_translations.print_summary()
            raise
        log_translations.register(
            "beta.chat.completions.parse", messages_sent, response
        )

        store_translation(
            helper,
            target_language,
            target_dialect,
            phrase,
            result.translated_text,
            result.notes,
            query_embedding,
        )

        if similar:
            print("  [Few-shot: used", len(similar), "similar past translation(s)]")
        print_indented(
            "  Translation", result.translated_text, indent=4, max_length=2000
        )
        print_indented("  Notes", result.notes, indent=4, max_length=1000)
        print()

    _print_db_summary(helper)
    log_language.print_summary()
    log_embeddings.print_summary()
    log_translations.print_summary()
    print("Goodbye.")


if __name__ == "__main__":
    main()
