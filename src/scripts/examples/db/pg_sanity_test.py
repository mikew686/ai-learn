#!/usr/bin/env python3
"""Sanity check: connect to Postgres as local user and run a simple query.

Uses the local (Unix) username for user and database when DATABASE_URL is not set.
See pg_user.md for setup.
"""

import os
import sys

from sqlalchemy import create_engine, text


def main() -> None:
    local_user = os.environ.get("USER") or os.environ.get("USERNAME") or "postgres"
    default_url = f"postgresql://{local_user}:vectorfun@localhost:5432/{local_user}"
    url = os.environ.get("DATABASE_URL", default_url)

    print(f"Local user: {local_user}", file=sys.stderr)
    print(f"Connecting...", file=sys.stderr)

    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        result = conn.execute(text("SELECT current_database(), current_user"))
        row = result.fetchone()
        print(f"database={row[0]!r}, user={row[1]!r}")
        ext = conn.execute(
            text(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
            )
        ).fetchone()
        if ext:
            print(f"vector extension: {ext[1]!r}", file=sys.stderr)
        else:
            print("vector extension: not found", file=sys.stderr)


if __name__ == "__main__":
    main()
