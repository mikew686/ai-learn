"""
Postgres + pgvector helper: connection and schema setup.

Use for database creation and connections when running vector search
against a Postgres database with the pgvector extension.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine, create_engine

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None  # type: ignore[misc, assignment]


def _default_url() -> str:
    """Build default Postgres URL from local user (see pg_user.md)."""
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "postgres"
    return f"postgresql://{user}:vectorfun@localhost:5432/{user}"


class PgVectorHelper:
    """
    Helper for Postgres + pgvector: connections and schema (extension + table).

    Use connect() to get a connection that has the vector type registered
    and the translations table available.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.environ.get("DATABASE_URL") or _default_url()
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self._url)
        return self._engine

    def _register_vector(self, conn: Connection) -> None:
        """Register pgvector type on the underlying DBAPI connection."""
        if register_vector is None:
            return
        raw = getattr(conn.connection, "dbapi_connection", conn.connection)
        register_vector(raw)

    @contextmanager
    def connect(self) -> Generator[Connection, None, None]:
        """Yield a connection with vector extension registered. Caller commits as needed."""
        with self.engine.connect() as conn:
            self._register_vector(conn)
            yield conn

    def init_schema(self, embedding_dim: int = 1536) -> None:
        """
        Create pgvector extension and tables if they do not exist.
        Uses cosine distance (vector_cosine_ops) for embeddings.
        Drop the table(s) first if you need to change the schema.
        """
        with self.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Dimension is fixed per table; 1536 matches text-embedding-3-small
            conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS translations (
                        id bigserial PRIMARY KEY,
                        language_name text NOT NULL,
                        region_code text NOT NULL,
                        language_code text NOT NULL,
                        dialect_description text NOT NULL DEFAULT '',
                        source_text text NOT NULL,
                        translated_text text NOT NULL,
                        phonetic_spelling text NOT NULL DEFAULT '',
                        notes text NOT NULL,
                        embedding vector({embedding_dim}) NOT NULL,
                        embedding_model text NOT NULL DEFAULT '',
                        embedding_cost real NOT NULL DEFAULT 0,
                        embedding_input_tokens int NOT NULL DEFAULT 0,
                        embedding_output_tokens int NOT NULL DEFAULT 0,
                        prompt_model text NOT NULL DEFAULT '',
                        prompt_cost real NOT NULL DEFAULT 0,
                        prompt_input_tokens int NOT NULL DEFAULT 0,
                        prompt_output_tokens int NOT NULL DEFAULT 0,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """))
            conn.commit()
            # HNSW index for cosine similarity (get_similar_translations: ORDER BY embedding <=> :query)
            conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS translations_embedding_hnsw_idx
                    ON translations
                    USING hnsw (embedding vector_cosine_ops)
                    """))
            # B-tree for filter (get_similar_translations: WHERE language_name = :lang)
            conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS translations_language_name_idx
                    ON translations (language_name)
                    """))
            # B-tree for summary (GROUP BY / ORDER BY language_name, dialect_description, region_code)
            conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS translations_lang_dialect_region_idx
                    ON translations (language_name, dialect_description, region_code)
                    """))
            conn.commit()
            # Table of known translation targets: maps natural-language descriptions to
            # (language_name, language_code, region_code, dialect_description) for the initial query.
            conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS translation_targets (
                        id bigserial PRIMARY KEY,
                        target_description text NOT NULL,
                        language_name text NOT NULL,
                        language_code text NOT NULL,
                        region_code text NOT NULL DEFAULT '',
                        dialect_description text NOT NULL DEFAULT '',
                        embedding vector({embedding_dim}) NOT NULL,
                        embedding_model text NOT NULL DEFAULT '',
                        embedding_cost real NOT NULL DEFAULT 0,
                        embedding_input_tokens int NOT NULL DEFAULT 0,
                        embedding_output_tokens int NOT NULL DEFAULT 0,
                        prompt_model text NOT NULL DEFAULT '',
                        prompt_cost real NOT NULL DEFAULT 0,
                        prompt_input_tokens int NOT NULL DEFAULT 0,
                        prompt_output_tokens int NOT NULL DEFAULT 0,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """))
            conn.commit()
            # HNSW for similarity (get_similar_targets: ORDER BY embedding <=> :query)
            conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS translation_targets_embedding_hnsw_idx
                    ON translation_targets
                    USING hnsw (embedding vector_cosine_ops)
                    """))
            # B-tree for summary (GROUP BY / ORDER BY language_name, dialect_description, region_code)
            conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS translation_targets_lang_dialect_region_idx
                    ON translation_targets (language_name, dialect_description, region_code)
                    """))
            conn.commit()
