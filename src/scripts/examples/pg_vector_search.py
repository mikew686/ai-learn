"""
Pattern 6: Embeddings / Vector Search (Postgres + pgvector)

Same approach as embeddings_vector_search but using a Postgres database with
the pgvector extension for vector similarity (cosine distance in-database).

Uses db.utils.PgVectorHelper for database creation and connections.
See pg_user.md and postgres_pgvector_notes.md for setup.

Embedding approach (exact-prompt embedding):
  - We embed the exact user-message prompt sent to the model, e.g.
    "Translate to French (Quebec French): <phrase>". Query and stored embeddings
    use the same wording (language_name + dialect_description), so retrieval
    matches "same task + similar phrase" and stays consistent with prompt wording.
  - Retrieval filters by language_name only (no dialect filter). Ranking is
    purely by cosine distance. The database schema uses language_name, region_code
    (and dialect_description) for display and analytics.

Target normalization (translation_targets table):
  - The user's raw target (e.g. "French (Quebec)") is embedded and compared to
    stored targets. If the nearest row has cosine distance below a threshold
    (TARGET_MATCH_DISTANCE_THRESHOLD, default 0.2), we use that row's
    (language_name, language_code, region_code, dialect_description) and skip
    the LLM parse. Otherwise we call the LLM (with similar targets as few-shot)
    and then store the new target for future lookups.

Usage:
    python -m src.scripts.examples.pg_vector_search [--model MODEL] [--embedding-model EMB_MODEL] [--db-url URL] [--top-k N] [--temperature T] [--max-tokens N]
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
    """Translation plus English phonetic spelling and brief notes (cultural/contextual)."""

    translated_text: str = Field(
        description="The translated text in the target language and dialect"
    )
    phonetic_spelling: str = Field(
        description="English phonetic spelling of the translation (how to pronounce it, e.g. IPA or respelling)"
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
    dialect_description: str = Field(
        default="",
        description="Short human-readable, canonical name for the region/dialect only when possible (e.g. Southern English, Louisiana English, Quebec French); empty if none",
    )


def get_similar_targets(
    helper: PgVectorHelper,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[float, str, str, str, str]]:
    """
    Return up to top_k rows from translation_targets by cosine distance.
    Each row is (distance, language_name, language_code, region_code, dialect_description).
    """
    with helper.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT embedding <=> :query AS dist,
                       language_name, language_code, region_code, dialect_description
                FROM translation_targets
                ORDER BY embedding <=> :query
                LIMIT :limit
                """),
            {"query": Vector(query_embedding), "limit": top_k},
        ).fetchall()
    return [(float(r[0]), r[1], r[2], r[3], r[4]) for r in rows]


def store_target(
    helper: PgVectorHelper,
    target_description: str,
    language_name: str,
    language_code: str,
    region_code: str,
    dialect_description: str,
    embedding: list[float],
    *,
    embedding_model: str = "",
    embedding_cost: float = 0.0,
    embedding_input_tokens: int = 0,
    embedding_output_tokens: int = 0,
    prompt_model: str = "",
    prompt_cost: float = 0.0,
    prompt_input_tokens: int = 0,
    prompt_output_tokens: int = 0,
) -> None:
    """Insert one translation target into translation_targets for future lookup. Includes model/cost/token fields when provided."""
    with helper.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO translation_targets
                (target_description, language_name, language_code, region_code, dialect_description, embedding, embedding_model, embedding_cost, embedding_input_tokens, embedding_output_tokens, prompt_model, prompt_cost, prompt_input_tokens, prompt_output_tokens)
                VALUES (:target_description, :language_name, :language_code, :region_code, :dialect_description, :embedding, :embedding_model, :embedding_cost, :embedding_input_tokens, :embedding_output_tokens, :prompt_model, :prompt_cost, :prompt_input_tokens, :prompt_output_tokens)
                """),
            {
                "target_description": target_description,
                "language_name": language_name,
                "language_code": language_code,
                "region_code": region_code,
                "dialect_description": dialect_description,
                "embedding": Vector(embedding),
                "embedding_model": embedding_model,
                "embedding_cost": embedding_cost,
                "embedding_input_tokens": embedding_input_tokens,
                "embedding_output_tokens": embedding_output_tokens,
                "prompt_model": prompt_model,
                "prompt_cost": prompt_cost,
                "prompt_input_tokens": prompt_input_tokens,
                "prompt_output_tokens": prompt_output_tokens,
            },
        )
        conn.commit()


# Cosine distance below this: use top-1 from translation_targets to fill fields without calling LLM.
TARGET_MATCH_DISTANCE_THRESHOLD = 0.2


def parse_language_region(
    client: OpenAI,
    chat_model: str,
    user_input: str,
    *,
    similar_targets: list[tuple[str, str, str, str]] | None = None,
    log: OpenAILog | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[tuple[str, str, str, str], object | None]:
    """Parse user phrase into ((language_name, language_code, region_code, dialect_description), response). response is the API response or None (for empty input or when no call made)."""
    if not (user_input or "").strip():
        return ("English", "en", "", ""), None
    system = (
        "Normalize the user's translation target into standard identifiers and a human-readable description. "
        "Return: (1) canonical language name in English, "
        "(2) ISO 639-1 two-letter language code, "
        "(3) ISO 3166-1 alpha-2 region code if a region or dialect was specified; otherwise empty string. "
        "(4) dialect_description: short human-readable, canonical name for the region/dialect only when possible (e.g. Southern English, Louisiana English, Quebec French); empty if none."
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    if similar_targets:
        examples = "\n".join(
            f"  language_name={ln}, language_code={lc}, region_code={rc or '(empty)'}, dialect_description={dd or '(empty)'}"
            for ln, lc, rc, dd in similar_targets
        )
        messages[0]["content"] = (
            system
            + " Here are some existing targets for reference (use the same style):\n"
            + examples
        )
    messages.append(
        {
            "role": "user",
            "content": f"Normalize this translation target: {user_input.strip()}",
        }
    )
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
    dialect_desc = (parsed.dialect_description or "").strip()
    return (name, code, region, dialect_desc), response


def _cost_from_usage(usage: object) -> float:
    """Return cost in USD from API response usage. All returns include usage.cost when available."""
    if usage is None:
        return 0.0
    cost = getattr(usage, "cost", None)
    if cost is not None:
        return float(cost)
    return 0.0


def _tokens_from_usage(usage: object) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from API response usage. Supports input_tokens/output_tokens or prompt_tokens/completion_tokens."""
    if usage is None:
        return 0, 0
    inp = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or 0
    )
    out = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )
    return int(inp), int(out)


def get_embedding(
    client: OpenAI,
    model: str,
    text_input: str,
    *,
    log: OpenAILog | None = None,
) -> tuple[list[float], object]:
    """Return (embedding vector, response) for a single text. Use response.usage for cost."""
    if log:
        log.start_call()
    response = client.embeddings.create(model=model, input=text_input)
    if log:
        log.register(
            "embeddings.create",
            [
                {
                    "model": model,
                    "input": text_input[:2000]
                    + ("..." if len(text_input) > 2000 else ""),
                }
            ],
            response,
        )
    return response.data[0].embedding, response


def _embedding_text(phrase: str, language_name: str, dialect_description: str) -> str:
    """Build the string to embed: use the exact prompt wording sent to the model so retrieval matches same task + phrase."""
    if (dialect_description or "").strip():
        return f"Translate to {language_name} ({dialect_description.strip()}): {phrase}"
    return f"Translate to {language_name}: {phrase}"


def _print_db_summary(helper: PgVectorHelper) -> None:
    """Print summary of both tables: totals by language/dialect, then total tokens and cost for entire DB per table."""
    with helper.connect() as conn:
        print()
        print("--- Vector store summary (Postgres + pgvector) ---")

        for table, label in [
            ("translations", "translations"),
            ("translation_targets", "translation_targets"),
        ]:
            total = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
            print(f"  {label}: {total} total")
            if total and total > 0:
                rows = conn.execute(text(f"""
                        SELECT language_name, language_code, region_code, dialect_description, COUNT(*) AS n
                        FROM {table}
                        GROUP BY language_name, language_code, region_code, dialect_description
                        ORDER BY language_name, dialect_description, region_code
                        """)).fetchall()
                for lang_name, lang_code, region_code, dialect_desc, n in rows:
                    dialect_label = (dialect_desc or "").strip() or "(no dialect)"
                    codes = f"[{lang_code}, {region_code or '-'}]"
                    print(f"    {lang_name} — {dialect_label} {codes}: {n}")
                # Whole-table totals: tokens and cost
                row = conn.execute(text(f"""
                        SELECT
                          COALESCE(SUM(embedding_input_tokens + embedding_output_tokens + prompt_input_tokens + prompt_output_tokens), 0) AS total_tokens,
                          COALESCE(SUM(embedding_cost + prompt_cost), 0) AS total_cost
                        FROM {table}
                        """)).fetchone()
                total_tokens = int(row[0]) if row else 0
                total_cost = float(row[1]) if row and row[1] is not None else 0.0
                print(
                    f"    Total tokens: {total_tokens:,}  Total cost: ${total_cost:.6f}"
                )
        print("--------------------------------------------------")
        print()


def store_translation(
    helper: PgVectorHelper,
    target_language: str,
    language_code: str,
    target_dialect: str,
    dialect_description: str,
    source_text: str,
    translated_text: str,
    phonetic_spelling: str,
    notes: str,
    embedding: list[float],
    *,
    embedding_model: str = "",
    embedding_cost: float = 0.0,
    embedding_input_tokens: int = 0,
    embedding_output_tokens: int = 0,
    prompt_model: str = "",
    prompt_cost: float = 0.0,
    prompt_input_tokens: int = 0,
    prompt_output_tokens: int = 0,
) -> None:
    """Insert one translation and its embedding into the DB. Includes phonetic_spelling, cost, and token fields from API usage."""
    with helper.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO translations
                (language_name, region_code, language_code, dialect_description, source_text, translated_text, phonetic_spelling, notes, embedding, embedding_model, embedding_cost, embedding_input_tokens, embedding_output_tokens, prompt_model, prompt_cost, prompt_input_tokens, prompt_output_tokens)
                VALUES (:language_name, :region_code, :language_code, :dialect_description, :source_text, :translated_text, :phonetic_spelling, :notes, :embedding, :embedding_model, :embedding_cost, :embedding_input_tokens, :embedding_output_tokens, :prompt_model, :prompt_cost, :prompt_input_tokens, :prompt_output_tokens)
                """),
            {
                "language_name": target_language,
                "region_code": target_dialect,
                "language_code": language_code,
                "dialect_description": dialect_description,
                "source_text": source_text,
                "translated_text": translated_text,
                "phonetic_spelling": phonetic_spelling,
                "notes": notes,
                "embedding": Vector(embedding),
                "embedding_model": embedding_model,
                "embedding_cost": embedding_cost,
                "embedding_input_tokens": embedding_input_tokens,
                "embedding_output_tokens": embedding_output_tokens,
                "prompt_model": prompt_model,
                "prompt_cost": prompt_cost,
                "prompt_input_tokens": prompt_input_tokens,
                "prompt_output_tokens": prompt_output_tokens,
            },
        )
        conn.commit()


def get_similar_translations(
    helper: PgVectorHelper,
    target_language: str,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[str, str, str, str, str, str]]:
    """
    Return up to top_k past translations for the given target language,
    ranked by pgvector cosine distance only. Each row is
    (source_text, translated_text, notes, phonetic_spelling, language_name, dialect_description)
    so few-shot prompts use the same wording as the original prompts (stored language_name and dialect_description, not codes).
    """
    with helper.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT source_text, translated_text, notes, phonetic_spelling, language_name, dialect_description
                FROM translations
                WHERE language_name = :lang
                ORDER BY embedding <=> :query
                LIMIT :limit
                """),
            {"lang": target_language, "query": Vector(query_embedding), "limit": top_k},
        ).fetchall()
    return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]


def _target_label(language_name: str, dialect_description: str) -> str:
    """Build 'Translate to X' label from stored language_name and dialect_description (same wording as original prompts, not codes)."""
    if dialect_description:
        return f"Translate to {language_name} ({dialect_description})"
    return f"Translate to {language_name}"


def translate_phrase(
    client: OpenAI,
    chat_model: str,
    target_language: str,
    source_text: str,
    few_shot_examples: list[tuple[str, str, str, str, str, str]],
    *,
    target_spec: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[TranslationWithNotes, object, list]:
    """Call chat API to translate source_text with optional few-shot examples.
    few_shot_examples: (source_text, translated_text, notes, phonetic_spelling, language_name, dialect_description) from DB.
    Few-shot labels use stored language_name and dialect_description so wording matches original prompts (no codes).
    target_spec: concatenated language_name + dialect_description for the current session (e.g. 'French (Quebec)').
    """
    tgt_spec = (target_spec or target_language).strip()
    system = (
        f"You are a translator into {tgt_spec}. "
        "For each phrase, provide: (1) the translation, (2) an English phonetic spelling (how to pronounce it, e.g. IPA or respelling), "
        "and (3) brief cultural or contextual notes in English."
    )
    messages = [{"role": "system", "content": system}]
    for (
        src,
        trans,
        notes,
        phonetic,
        ex_language_name,
        ex_dialect_description,
    ) in few_shot_examples:
        tgt_label = _target_label(ex_language_name, ex_dialect_description)
        messages.append({"role": "user", "content": f"{tgt_label}: {src}"})
        messages.append(
            {
                "role": "assistant",
                "content": f"Translation: {trans}\nPhonetic: {phonetic}\nNotes: {notes}",
            }
        )
    current_label = f"Translate to {tgt_spec}"
    messages.append({"role": "user", "content": f"{current_label}: {source_text}"})

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
        print(
            f"Overrides: temperature={args.temperature}, max_tokens={args.max_tokens}"
        )
    print()

    log_language = OpenAILog(log_raw_dir="data", description="language")
    log_embeddings = OpenAILog(log_raw_dir="data", description="embeddings")
    log_translations = OpenAILog(log_raw_dir="data", description="translations")

    raw_target = input(
        "Target language (with optional region or dialect, e.g. Spanish, French (Quebec), German): "
    ).strip()
    raw_target = raw_target or "Spanish"

    # Use translation_targets vector store to assist filling (language_name, language_code, region_code, dialect_description).
    target_query_embedding, target_emb_response = get_embedding(
        client, embedding_model, raw_target, log=log_embeddings
    )
    target_emb_usage = getattr(target_emb_response, "usage", None)
    target_embedding_cost = _cost_from_usage(target_emb_usage)
    target_emb_in, target_emb_out = _tokens_from_usage(target_emb_usage)
    similar_targets_with_dist = get_similar_targets(
        helper, target_query_embedding, top_k=5
    )
    similar_targets_tuples = [
        (ln, lc, rc or "", dd or "") for _, ln, lc, rc, dd in similar_targets_with_dist
    ]

    parse_response = None
    if (
        similar_targets_with_dist
        and similar_targets_with_dist[0][0] < TARGET_MATCH_DISTANCE_THRESHOLD
    ):
        (
            _dist,
            target_language,
            target_language_code,
            target_dialect,
            dialect_description,
        ) = similar_targets_with_dist[0]
        target_dialect = (target_dialect or "").strip().upper()
        dialect_description = (dialect_description or "").strip()
    else:
        (
            target_language,
            target_language_code,
            target_dialect,
            dialect_description,
        ), parse_response = parse_language_region(
            client,
            chat_model,
            raw_target,
            similar_targets=(
                similar_targets_tuples[:3] if similar_targets_tuples else None
            ),
            log=log_language,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        target_dialect = (target_dialect or "").strip().upper()
        dialect_description = (dialect_description or "").strip()

    target_prompt_usage = (
        getattr(parse_response, "usage", None) if parse_response else None
    )
    target_prompt_cost = (
        _cost_from_usage(target_prompt_usage) if parse_response else 0.0
    )
    target_prompt_model = chat_model if parse_response else ""
    target_prompt_in, target_prompt_out = _tokens_from_usage(target_prompt_usage)

    # Remember this target for future lookups.
    store_target(
        helper,
        raw_target,
        target_language,
        target_language_code,
        target_dialect,
        dialect_description,
        target_query_embedding,
        embedding_model=embedding_model,
        embedding_cost=target_embedding_cost,
        embedding_input_tokens=target_emb_in,
        embedding_output_tokens=target_emb_out,
        prompt_model=target_prompt_model,
        prompt_cost=target_prompt_cost,
        prompt_input_tokens=target_prompt_in,
        prompt_output_tokens=target_prompt_out,
    )
    target_spec = (
        f"{target_language} ({dialect_description})"
        if dialect_description
        else target_language
    )
    print()
    print(
        f"Translating into: {target_spec} "
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

        text_to_embed = _embedding_text(phrase, target_language, dialect_description)
        query_embedding, emb_response = get_embedding(
            client, embedding_model, text_to_embed, log=log_embeddings
        )
        emb_usage = getattr(emb_response, "usage", None)
        embedding_cost = _cost_from_usage(emb_usage)
        emb_in, emb_out = _tokens_from_usage(emb_usage)
        similar = get_similar_translations(
            helper, target_language, query_embedding, args.top_k
        )

        log_translations.start_call()
        try:
            result, response, messages_sent = translate_phrase(
                client,
                chat_model,
                target_language,
                phrase,
                similar,
                target_spec=target_spec,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception:
            log_translations.print_summary()
            raise
        log_translations.register(
            "beta.chat.completions.parse", messages_sent, response
        )
        prompt_usage = getattr(response, "usage", None)
        prompt_cost = _cost_from_usage(prompt_usage)
        prompt_in, prompt_out = _tokens_from_usage(prompt_usage)

        store_translation(
            helper,
            target_language,
            target_language_code,
            target_dialect,
            dialect_description,
            phrase,
            result.translated_text,
            result.phonetic_spelling,
            result.notes,
            query_embedding,
            embedding_model=embedding_model,
            embedding_cost=embedding_cost,
            embedding_input_tokens=emb_in,
            embedding_output_tokens=emb_out,
            prompt_model=chat_model,
            prompt_cost=prompt_cost,
            prompt_input_tokens=prompt_in,
            prompt_output_tokens=prompt_out,
        )

        if similar:
            print("  [Few-shot: used", len(similar), "similar past translation(s)]")
        print_indented(
            "  Translation", result.translated_text, indent=4, max_length=2000
        )
        print_indented("  Phonetic", result.phonetic_spelling, indent=4, max_length=500)
        print_indented("  Notes", result.notes, indent=4, max_length=1000)
        print()

    _print_db_summary(helper)
    log_language.print_summary()
    log_embeddings.print_summary()
    log_translations.print_summary()
    print("Goodbye.")


if __name__ == "__main__":
    main()
