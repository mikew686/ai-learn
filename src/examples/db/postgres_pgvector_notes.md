# PostgreSQL + pgvector for LLM Embeddings

## Overview

**pgvector** is a PostgreSQL extension that adds a native `vector` data
type and similarity search operators. It allows embeddings to be stored
and queried directly inside Postgres.

This enables:

-   Semantic search
-   Retrieval-Augmented Generation (RAG)
-   Hybrid relational + vector filtering
-   Similarity search across documents, users, events, etc.

pgvector is the de-facto standard vector solution for PostgreSQL today.

------------------------------------------------------------------------

## Architecture Model

Postgres remains the primary system of record.

    Application
        ↓
    PostgreSQL
        ├── Relational tables
        ├── JSONB metadata
        └── vector columns (pgvector)

Key advantage:

> Embeddings live alongside structured data --- no separate vector
> database required.

------------------------------------------------------------------------

## Installation

### Self-hosted Linux (APT)

``` bash
sudo apt install postgresql-16
sudo apt install postgresql-16-pgvector
```

Inside the database:

``` sql
CREATE EXTENSION vector;
```

### AWS RDS / Aurora PostgreSQL

pgvector is AWS-supported on modern PostgreSQL versions.

``` sql
CREATE EXTENSION vector;
```

(No OS-level installation required.)

------------------------------------------------------------------------

## Data Modeling

### Basic Table Pattern

``` sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    org_id bigint NOT NULL,
    content text NOT NULL,
    metadata jsonb,
    embedding vector(1536)
);
```

Notes:

-   `vector(1536)` should match your embedding model dimension
-   `jsonb` supports flexible metadata filtering
-   Foreign keys and joins work normally

------------------------------------------------------------------------

## Similarity Search

### Distance Operators

  Operator   Meaning
  ---------- -------------------------
  `<->`      Euclidean distance (L2)
  `<#>`      Inner product
  `<=>`      Cosine distance

Cosine distance is the most common choice for LLM embeddings.

Example:

``` sql
SELECT id, content
FROM documents
ORDER BY embedding <=> :query_embedding
LIMIT 5;
```

------------------------------------------------------------------------

## Indexing

Without an index, vector search performs a full table scan.

For production workloads, use approximate nearest neighbor (ANN)
indexes.

------------------------------------------------------------------------

### HNSW (Recommended)

``` sql
CREATE INDEX documents_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops);
```

Characteristics: - High recall - Fast query latency - Slower inserts -
Higher memory usage

Best for: - Read-heavy RAG - Semantic search - Production inference
workloads

------------------------------------------------------------------------

### IVFFLAT (Alternative)

``` sql
CREATE INDEX documents_embedding_idx
ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

After creation:

``` sql
ANALYZE documents;
```

Characteristics: - Faster inserts - Lower memory footprint - Slightly
lower recall

------------------------------------------------------------------------

## Hybrid Filtering (Major Strength)

pgvector integrates cleanly with relational filters.

Example:

``` sql
SELECT *
FROM documents
WHERE org_id = 42
  AND metadata->>'category' = 'math'
ORDER BY embedding <=> :query_embedding
LIMIT 10;
```

This enables: - Multi-tenant filtering - Permission-aware search - Rich
metadata constraints

All within a single SQL query.

------------------------------------------------------------------------

## Performance Expectations

  Scale             Suitability
  ----------------- ---------------------------------
  \< 100k vectors   Excellent
  \~1M vectors      Very good
  5--10M vectors    Strong with HNSW
  50M+ vectors      May require dedicated vector DB

Performance depends on: - RAM - CPU - Index choice - Query concurrency

------------------------------------------------------------------------

## Operational Considerations

### Memory

HNSW indexes are memory-intensive.

Recommendations: - Properly size `shared_buffers` - Increase
`maintenance_work_mem` for index builds - Avoid burstable instances for
production

------------------------------------------------------------------------

### Write Patterns

-   Vector inserts are slower than standard rows
-   HNSW index maintenance is costly
-   Prefer batch inserts when possible

------------------------------------------------------------------------

### Backup & Replication

pgvector works with standard Postgres tooling:

-   WAL replication
-   Logical replication
-   RDS snapshots
-   `pg_dump` / `pg_restore`

No special handling required.

------------------------------------------------------------------------

## Advantages

-   Single datastore
-   ACID guarantees
-   Mature Postgres ecosystem
-   Simple DevOps model
-   ORM compatibility (SQLAlchemy, Django, etc.)
-   Easy adoption for existing Postgres systems

------------------------------------------------------------------------

## Limitations

-   Not optimized for hundreds of millions of vectors
-   No native sharding
-   Index build times grow with dataset size
-   Fewer ANN tuning knobs than specialized vector DBs

------------------------------------------------------------------------

## Ideal Use Cases

-   Retrieval-Augmented Generation (RAG)
-   Semantic document search
-   Embedding-based recommendations
-   Multi-tenant SaaS search
-   Hybrid structured + semantic queries

------------------------------------------------------------------------

## When Not to Use pgvector

Consider a dedicated vector database if:

-   Dataset exceeds \~50--100M embeddings
-   Ultra-low latency (\<5ms) is required at scale
-   Heavy real-time ingestion
-   Very high concurrency search workloads

------------------------------------------------------------------------

## Production Checklist

-   [ ] Enable pgvector extension
-   [ ] Confirm embedding dimension
-   [ ] Create ANN index (HNSW preferred)
-   [ ] Tune memory parameters
-   [ ] Avoid burstable instances
-   [ ] Monitor CPU and RAM
-   [ ] Benchmark with realistic data volume

------------------------------------------------------------------------

## Directions for Running It (Docker)

This directory provides Postgres + pgvector using a **named volume** (`pgvector_data`). Data persists across container restarts.

### Prerequisites

- Docker and Docker Compose installed.

### Build and run

From this directory (`ai-learn/src/db/`):

``` bash
# Build the image
docker compose build

# Start the container (detached)
docker compose up -d

# Optional: stream logs
docker compose logs -f postgres
```

On first run, the entrypoint initializes the volume and runs `init-pgvector.sql`, which enables the `vector` extension. Subsequent starts reuse the same data.

### Connect

- **Host:** `localhost`
- **Port:** `5432`
- **User:** `postgres`
- **Password:** `postgres`

Example:

``` bash
psql -h localhost -p 5432 -U postgres -d postgres
```

Then in `psql`:

``` sql
CREATE EXTENSION vector;  -- already done by init script; safe to run again
```

### Stop and remove container (keep data)

``` bash
docker compose down
```

Data in the `pgvector_data` volume is preserved. Run `docker compose up -d` again to resume.

### Remove container and data

``` bash
docker compose down -v
```

The `-v` flag removes the named volume, so all database data is deleted.

### Named volume

- Data lives in the Docker volume `pgvector_data` (managed by Docker; not a directory in the repo).
- **Backup:** Use `pg_dump` from the host or from another container that has `postgresql-client` and can reach the container.

------------------------------------------------------------------------

## Conceptual Summary

pgvector turns PostgreSQL into a:

> **Hybrid relational + semantic database**

It is not a research-grade ANN engine, but it *is* a pragmatic,
production-ready vector solution tightly integrated with the most mature
open-source relational database available.
