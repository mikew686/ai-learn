# Embeddings / Vector Search Pattern

## Overview

**Embeddings / Vector Search** is the use of dense vector representations (embeddings) and similarity search to find relevant content. It is the foundation for semantic search, retrieval-augmented generation (RAG), and dynamic few-shot selection. This pattern covers embedding generation, vector storage, similarity metrics, and how to combine them in an application—for example, storing translation examples and retrieving the most similar ones to build few-shot prompts.

## What are embeddings? (intuitive description)

An **embedding** is a numerical representation of meaning. An embedding model takes a piece of text (a phrase, a sentence, a paragraph) and maps it to a fixed-length list of numbers—a **vector**. The crucial property: **texts that are similar in meaning end up close together in this number space**, while unrelated texts end up farther apart. You don’t interpret the individual numbers; you use the **distance** or **angle** between two vectors to measure how semantically similar two texts are.

Think of it like a coordinate system for “meaning”: the model has learned to place “How are you?” and “How’s it going?” near each other, and “The weather is nice” in a different region. That’s what makes **semantic search** possible—finding content that is alike in meaning, not just in keywords. Once you have vectors, you can **rank** stored items by similarity to a query (e.g. “which past translations are most like this phrase?”) and take the top-K. Embeddings are the bridge between raw text and this geometric view of meaning.

## Description

Embedding models convert text into fixed-length vectors that capture semantic meaning. Similar texts produce similar vectors; you can rank items by similarity (e.g. cosine similarity or dot product) to implement semantic search. Storing embeddings in a local or remote store (SQLite, FAISS, or a dedicated vector DB) lets you scale beyond in-memory lookups while keeping the same workflow: embed the query, compare to stored vectors, take the top-K.

**Key Concepts**:
- **Embedding generation**: Call an embeddings API (e.g. OpenAI `text-embedding-3-small`) with one or many texts; get one vector per text.
- **Vector similarity**: Cosine similarity or dot product (often with normalized vectors) to rank by relevance.
- **Local vector storage**: Start with something simple (e.g. SQLite with a BLOB column, or an in-memory structure); move to FAISS or a vector DB when needed.
- **Dynamic few-shot**: Use the current user input (or its embedding) to select the most relevant stored examples and inject them into the prompt.
- **Exact-prompt embedding**: When using embeddings to retrieve few-shot examples for a chat model, embed the **same text you send in the user message** (e.g. “Translate to French (Quebec): &lt;phrase&gt;”). Query and stored items then share the same surface form, so retrieval matches “same task + similar phrase” and stays consistent with how the model sees the prompt.
- **Target normalization (vector-backed)**: For free-form user input (e.g. “French Quebec”, “Spanish Mexico”), you need canonical fields (language name, codes, dialect). Instead of always calling an LLM to parse, maintain a small **target store**: embed the user’s raw target string and search for the nearest stored target. If the nearest row is **close enough** (cosine distance below a threshold), use that row’s canonical fields and **skip the LLM**. Otherwise call the LLM, then **store** the new (raw target, canonical fields, embedding) for future lookups. Over time, repeated phrasings hit the store and avoid extra LLM calls.

## How It Works

### 1. Generate embeddings

Use the provider’s embeddings API. One request can take a single string or a batch of strings.

```python
from openai import OpenAI

client = OpenAI()

# Single text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog"
)
embedding = response.data[0].embedding  # list of floats

# Batch (more efficient for many texts)
texts = ["First phrase", "Second phrase", "Third phrase"]
response = client.embeddings.create(model="text-embedding-3-small", input=texts)
embeddings = [item.embedding for item in response.data]
```

### 2. Store vectors with metadata

Store each vector with whatever you need to retrieve later (e.g. source text, translation, language, dialect). In a simple SQLite setup you might store the vector as a BLOB (e.g. `numpy` float32 array serialized to bytes) and keep dimension and metadata in columns.

```python
import sqlite3
import numpy as np

# Example: one row per translation with embedding blob
conn.execute("""
    CREATE TABLE IF NOT EXISTS translations (
        id INTEGER PRIMARY KEY,
        target_language TEXT,
        target_dialect TEXT,
        source_text TEXT,
        translated_text TEXT,
        notes TEXT,
        embedding BLOB,
        embedding_dim INTEGER
    )
""")
arr = np.array(embedding, dtype=np.float32)
conn.execute(
    "INSERT INTO translations (..., embedding, embedding_dim) VALUES (..., ?, ?)",
    (arr.tobytes(), len(embedding)),
)
```

#### Popular databases and stores for embedding data

Choosing where to store vectors depends on scale, latency, and operational needs. Common options:

| Store | Type | Typical use | Notes |
|-------|------|-------------|--------|
| **SQLite** (e.g. BLOB column) | Relational, file-based | Prototypes, small to medium datasets, single process | No native vector index; load rows and compute similarity in app (e.g. NumPy). Simple, portable, good for thousands to low millions of vectors. |
| **FAISS** (Facebook AI Similarity Search) | In-memory index | High-throughput similarity search on one machine | Library, not a server. Build an index from vectors; query for top-K nearest neighbors. Supports L2 and inner product. Best when the index fits in RAM and you can rebuild or update in batch. |
| **Pinecone** | Managed vector DB | Production RAG, semantic search, multi-tenant | Hosted; no ops. Namespaces for multi-tenancy. Scale to large vector counts; pay per index and queries. |
| **Weaviate** | Vector DB (self-hosted or cloud) | RAG, hybrid search (vector + keyword) | Open source; can run locally or managed. GraphQL and REST; supports multiple vectorizers and metadata filters. |
| **Qdrant** | Vector DB (self-hosted or cloud) | RAG, filtering by metadata, payload search | Open source; filter-then-rank (e.g. by language) then vector search. Good fit when you need metadata filters alongside similarity. |
| **Milvus** / **Zilliz** | Vector DB | Large-scale similarity search, ML pipelines | Open source (Milvus); Zilliz is managed. Built for billions of vectors; supports multiple index types and metrics. |
| **Chroma** | Embedded / client library | Local dev, small apps, quick RAG experiments | Lightweight; often used in Python scripts or notebooks. Persists to disk; no separate server. |
| **pgvector** (PostgreSQL) | Extension to Postgres | Apps already on Postgres; vector + SQL in one DB | Add a vector column and use `<=>` (or similar) for approximate nearest neighbor. Keeps vectors and relational data in one place. |
| **Redis** (e.g. RediSearch) | In-memory store with vector support | Low-latency search, caching, real-time apps | Vector similarity search in Redis; good when you already use Redis and need fast lookups. |

**Practical path**: Start with **SQLite + BLOB** (or Chroma) for development and small data. Move to **FAISS** for higher QPS on one machine with in-memory index, or to a **vector DB** (Pinecone, Qdrant, Weaviate, pgvector) when you need filtering, persistence, scale, or a shared service. Use **filter-then-rank** (metadata filter then vector similarity) when your store supports it.

### 3. Similarity search

Load the relevant rows (e.g. same language/dialect or all rows), reconstruct vectors from BLOBs, compute similarity between the query vector and each stored vector, then take the top-K. The choice of **similarity metric** determines how “closeness” is defined; see the next subsection for detail.

```python
import numpy as np

def cosine_similarities(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Query (1D) vs each row of vectors (2D)."""
    q = query.astype(np.float32).reshape(-1)
    norms_q = np.linalg.norm(q)
    if norms_q == 0:
        return np.zeros(len(vectors))
    norms_v = np.linalg.norm(vectors, axis=1)
    norms_v = np.where(norms_v == 0, 1e-10, norms_v)
    return (vectors @ q) / (norms_v * norms_q)

# query_embedding: list from API; rows: list of (embedding_blob, ...)
vectors = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
scores = cosine_similarities(np.array(query_embedding), vectors)
top_indices = np.argsort(-scores)[:top_k]
```

#### Vector similarity metrics: cosine and alternatives

**Cosine similarity**

Cosine similarity measures the angle between two vectors, ignoring their length. It is defined as:

\[
\text{cosine}(a, b) = \frac{a \cdot b}{\|a\| \,\|b\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \sqrt{\sum_i b_i^2}}
\]

- **Range**: \([-1, 1]\). Value 1 means same direction (most similar), 0 means orthogonal, \(-1\) means opposite direction.
- **Interpretation**: High values indicate that the two texts are semantically aligned; low or negative values indicate unrelated or opposing meaning. For typical embedding models, most pairwise similarities are positive.
- **Why it’s common for embeddings**: Many embedding APIs (including OpenAI) return **normalized** vectors (unit length). For normalized vectors, cosine similarity equals the **dot product** \(a \cdot b\), so you can skip the denominator and just use the dot product for both ranking and interpretation. Even when vectors are not normalized, cosine similarity focuses on **direction** (topic/semantics) rather than **magnitude**, which often makes it more stable across documents of different length or scale.
- **Implementation note**: If the API does not normalize, normalize once (e.g. divide by \(\|a\|\)) before storing or comparing, or compute the full formula above to avoid double work.

**Dot product**

\[
\text{dot}(a, b) = a \cdot b = \sum_i a_i b_i
\]

- **Range**: Unbounded; depends on vector length and dimension.
- **When to use**: When vectors are **already normalized** (unit length), dot product and cosine similarity are identical. Using the dot product alone is then a small optimization (no division). If vectors are not normalized, dot product is sensitive to magnitude: longer documents can score higher even when direction is less similar, which may or may not be desired.
- **Summary**: Prefer dot product when you normalize (or the API guarantees normalization); otherwise cosine is usually preferable for semantic ranking.

**Euclidean (L2) distance**

\[
d(a, b) = \|a - b\|_2 = \sqrt{\sum_i (a_i - b_i)^2}
\]

- **Range**: \([0, +\infty)\). Smaller distance = more similar.
- **Relationship to cosine**: For **normalized** vectors, \(d^2 = 2(1 - \cos(a,b))\), so ranking by increasing L2 distance is equivalent to ranking by decreasing cosine similarity. For unnormalized vectors, L2 distance is influenced by both direction and length.
- **When to use**: Many vector indexes (e.g. FAISS, Annoy) are built for “nearest neighbor” under L2. If your vectors are normalized, you can use L2 in the index and get the same ordering as cosine; otherwise L2 may emphasize magnitude differences. Some systems also support “cosine” natively by storing normalized vectors and using L2.

**Other options**

- **Manhattan (L1) distance** \(\sum_i |a_i - b_i|\): Less common for embeddings; more sensitive to coordinate-wise differences. Sometimes used in low-dimensional or sparse settings.
- **Negative distance as similarity**: If your search API returns “distance” (e.g. L2), you can use **negative distance** as a score so that “smaller distance” becomes “higher score” and you still take top-K by score. For normalized vectors, \(-\|a-b\|_2\) preserves the same order as cosine similarity.
- **Inner product (IP)** in vector DBs: Often synonymous with dot product; when vectors are normalized, IP and cosine rank identically.

**Cosine distance**

Some APIs and vector DBs expose **cosine distance** instead of similarity. For normalized vectors, cosine distance is often defined as \(1 - \cos(a,b)\), so it ranges from 0 (identical) to 2 (opposite). **Smaller distance = more similar.** A **threshold** on distance (e.g. “accept only if distance &lt; 0.2”) means “only treat as a match when the vector is at least this close.” That pattern is used in **target normalization**: if the nearest stored target is within the threshold, reuse it and skip the LLM.

**Practical recommendation**

For embedding-based retrieval: use **cosine similarity** (or dot product on normalized vectors) by default. It matches how embedding models are usually trained and how most APIs behave. Use **L2 distance** or **cosine distance** when you plug into a vector index that expects it (e.g. pgvector’s `<=>` operator); if you normalize vectors first, ranking stays consistent with cosine.

### 4. Combine with language/dialect weighting

When stored items have language or region (e.g. French vs German, or French vs French (Quebec)), you can weight similarity by how well the stored item matches the user’s chosen target: same language and same dialect = 1.0, same language different dialect = lower, different language = much lower. Final score = phrase_similarity × language_weight. That way “French” without dialect still ranks French examples above German.

## Translation example: few-shot from vector search

The example scripts implement this pattern for translation with two storage options (SQLite and Postgres + pgvector). The flow uses **target normalization** (optionally vector-backed) and **exact-prompt embedding** for phrase retrieval.

### Target normalization

The user types a free-form target such as “French (Quebec)”, “spanish mexico”, or “French for curriculum”. The app needs canonical fields: language name, ISO 639-1 language code, optional ISO 3166-1 region code, and a human-readable dialect description.

- **Vector-backed (optional)**: Embed the user’s raw target string and search a **translation_targets** table (or similar) for the nearest stored row. Each stored row has (raw target text, language_name, language_code, region_code, dialect_description, embedding). If the **cosine distance** to the nearest row is below a **threshold** (e.g. 0.2), use that row’s canonical fields and **skip the LLM**. Otherwise call the LLM to parse the target, then **insert** (raw target, parsed fields, embedding) into the target store. Next time the same or similar phrasing reuses the stored result. This reduces LLM calls and keeps parsing consistent.
- **LLM-only**: Alternatively, always call the LLM once per session to normalize the target; no target vector store.

### Per-phrase retrieval and few-shot

1. **Embed the exact prompt text**: Build the string that will appear in the user message, e.g. “Translate to French (Quebec French): &lt;phrase&gt;”. Embed **that** string (exact-prompt embedding). Stored translations were also embedded with the same format when they were created, so query and stored vectors live in the same “task + phrase” space.
2. **Filter then rank**: From the **translations** table, keep only rows that match the current target language (filter by `language_name` or equivalent). Rank the remaining rows by **cosine distance** (or cosine similarity) between the query embedding and each row’s embedding. Optionally apply a **dialect weight** (e.g. same region = 1.0, same language different region = 0.5) so same-dialect examples rank higher. Take the **top-K** rows.
3. **Few-shot prompt**: Build the chat prompt with a system message, then for each of the top-K rows add a user message (e.g. “Translate to French (Quebec French): &lt;source&gt;”) and an assistant message (translation + notes). Add the current user message with the new phrase. Call the chat API for translation (and notes, phonetic spelling, etc.).
4. **Store new result**: Save the new (source, translation, notes, embedding, and any cost/token fields) in the translations table. Optionally store the target in the target store if it was parsed by the LLM this run.

The database grows over time; later runs get better few-shot context and, with a target store, fewer normalization calls.

**Examples**: [embeddings_vector_search.py](../src/scripts/examples/embeddings_vector_search.py) (SQLite, dialect weighting); [pg_vector_search.py](../src/scripts/examples/pg_vector_search.py) (Postgres + pgvector, target normalization via translation_targets, exact-prompt embedding, cost/token tracking).

## Best practices

- **Normalize vectors** when using cosine similarity (or use APIs that return normalized vectors).
- **Batch embedding calls** when you have many texts to embed.
- **Store embedding dimension** with each row if you might switch models or mix dimensions.
- **Weight by metadata** (e.g. language/dialect) so retrieval respects user context, not only semantic similarity.
- **Start with simple storage** (SQLite, in-memory); move to FAISS or a vector DB when scale or latency requires it.
- **Use standard identifiers** (e.g. ISO 639-1, ISO 3166-1) for language and region so normalization and weighting stay consistent.
- **Exact-prompt embedding**: When retrieving few-shot examples for a chat flow, embed the same string that will appear in the user message (e.g. “Translate to X: &lt;phrase&gt;”) so query and stored items align and retrieval matches the model’s view of the task.
- **Target normalization with a vector store**: For free-form targets (language/region/dialect), maintain a small target table: embed the raw input, search for the nearest stored target, and if within a distance threshold reuse it and skip the LLM; otherwise parse with the LLM and insert the new target for next time.

## When to use

- Semantic search over your own content (documents, translations, support answers).
- Dynamic few-shot or example selection for prompts.
- RAG (retrieval-augmented generation): embed documents, embed query, retrieve, then generate with retrieved context.
- Clustering, deduplication, or recommendation when items have a text component.

## Learning path

- **Start**: Use **SQLite + BLOB** (or Chroma) for development and small datasets. Implement cosine similarity in your app (e.g. NumPy). No extra infrastructure.
- **Scale up**: Add **FAISS** for higher QPS when the index fits in memory and you can build/update in batch; or move to a **vector DB** (Pinecone, Qdrant, Weaviate, pgvector) when you need filtering, persistence, or shared access.
- **Filter-then-rank**: When your store supports metadata (e.g. language), filter first, then rank by similarity (and optional weights). See the [Translation example](#translation-example-few-shot-from-vector-search) and [embeddings_vector_search.py](../src/scripts/examples/embeddings_vector_search.py).
- **In the progression**: This is Pattern 6 in the [learning progression](./learning_progression.md); it builds on Understanding Models, Prompts, Structured Output, Tools, and Schema-Driven Inference, and leads into Few-Shot and RAG (Pattern 8).

## Related patterns

- **RAG (Retrieval-Augmented Generation)**: Uses embeddings and vector search to retrieve context before generation; see [learning_progression.md](./learning_progression.md) Pattern 8.
- **Few-Shot / In-Context Learning**: Vector search often selects which few-shot examples to include; see [README.md](./README.md) and [learning_progression.md](./learning_progression.md) Pattern 7.
- **Prompt Engineering**: Retrieved content is injected into prompts; see [prompt_engineering.md](./prompt_engineering.md).
- **Understanding Models – Embedding Models**: Technical details on embedding APIs and similarity; see [understanding_models.md](./understanding_models.md#embedding-models).

## Documentation links

- **OpenAI Embeddings Guide**: https://platform.openai.com/docs/guides/embeddings
- **OpenAI Embedding Models**: https://platform.openai.com/docs/models/embeddings
- **RAG with OpenAI**: https://cookbook.openai.com/examples/rag_with_openai_embeddings
