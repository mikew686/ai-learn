# Setting Up a Local Postgres User and Database

Directions to create a dedicated **superuser** and database for local testing. No passwords; use with trust or peer auth only (not for production).

------------------------------------------------------------------------

## Prerequisites

- Postgres is running (e.g. via the [Docker setup](postgres_pgvector_notes.md#directions-for-running-it-docker) in this directory).
- You can connect as the superuser once to run the setup steps below.

On Linux, your login (Unix) user name:

``` bash
whoami
```

------------------------------------------------------------------------

## 1. Connect as default user (postgres)

Use one of the following depending on how Postgres is run.

**Local install (direct):**

``` bash
sudo -u postgres psql
```

**Docker (TCP to container):**

``` bash
psql -h localhost -p 5432 -U postgres -d postgres
```

------------------------------------------------------------------------

## 2. Create a superuser and database

Run the following in `psql`, replacing `myuser` with the name you want (database will have the same name as the user).

``` sql
-- Superuser for local testing
CREATE ROLE myuser WITH LOGIN SUPERUSER;

CREATE DATABASE myuser OWNER myuser;

-- Set password for TCP/Docker connections
ALTER USER myuser WITH PASSWORD 'vectorfun';
```

To enable pgvector in the new database, connect to that database and run:

``` sql
\c myuser
CREATE EXTENSION IF NOT EXISTS vector;
```

To see if the extension is installed (in `psql`):

``` sql
\dx vector
```

Or list all extensions: `\dx`.

You only need to run `CREATE EXTENSION vector` once per database; it persists across connections. If it seems missing after reconnecting, check which database you’re in (`SELECT current_database();` or `\conninfo`) — extensions are per-database, so connecting to a different database (e.g. `postgres`) won’t show the extension you enabled in `myuser`.

To remove the extension (connect to the database first):

``` sql
DROP EXTENSION IF EXISTS vector CASCADE;
```

`CASCADE` drops dependent objects (e.g. tables with `vector` columns, indexes). Omit it if you want the command to fail when such objects exist.

------------------------------------------------------------------------

## 3. Connect with psql

**Local (default Postgres):** `psql` uses local networking. With peer auth, the connection user is your login user and the default database has the same name, so `-d` is not required:

``` bash
psql
```

**Docker:** use host, port, and user explicitly (password `vectorfun` when prompted):

``` bash
psql -h localhost -p 5432 -U myuser -d myuser
```

------------------------------------------------------------------------

## 4. Connect with SQLAlchemy

**Connection URL format:** `postgresql://USER:PASSWORD@HOST:PORT/DATABASE`

**Minimal Python (SQLAlchemy 2.x):**

``` python
import os
from sqlalchemy import create_engine, text

url = os.environ.get("DATABASE_URL", "postgresql://myuser:vectorfun@localhost:5432/myuser")
engine = create_engine(url)

with engine.connect() as conn:
    result = conn.execute(text("SELECT current_database(), current_user"))
    print(result.fetchone())
```

**Using async (asyncpg):**

``` python
# pip install sqlalchemy[asyncio] asyncpg
url = os.environ.get("DATABASE_URL", "postgresql://myuser:vectorfun@localhost:5432/myuser")
engine = create_engine(url.replace("postgresql://", "postgresql+asyncpg://", 1), echo=True)
```

------------------------------------------------------------------------

## Quick reference

| Item     | Example value |
| -------- | ------------- |
| Host     | `localhost`   |
| Port     | `5432`        |
| User     | `myuser`      |
| Password | `vectorfun`   |
| Database | `myuser` (same as user) |
| URL      | `postgresql://myuser:vectorfun@localhost:5432/myuser` |
