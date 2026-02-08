# Embeddings / Vector Search Pattern

## Overview

**Embeddings / Vector Search** is the use of dense vector representations (embeddings) and similarity search to find relevant content. It is the foundation for semantic search, retrieval-augmented generation (RAG), and dynamic few-shot selection. This pattern covers embedding generation, vector storage, similarity metrics, and how to combine them in an application—for example, storing translation examples and retrieving the most similar ones to build few-shot prompts.

## Description

Embedding models convert text into fixed-length vectors that capture semantic meaning. Similar texts produce similar vectors; you can rank items by similarity (e.g. cosine similarity or dot product) to implement semantic search. Storing embeddings in a local or remote store (SQLite, FAISS, or a dedicated vector DB) lets you scale beyond in-memory lookups while keeping the same workflow: embed the query, compare to stored vectors, take the top-K.

**Key Concepts**:
- **Embedding generation**: Call an embeddings API (e.g. OpenAI `text-embedding-3-small`) with one or many texts; get one vector per text.
- **Vector similarity**: Cosine similarity or dot product (often with normalized vectors) to rank by relevance.
- **Local vector storage**: Start with something simple (e.g. SQLite with a BLOB column, or an in-memory structure); move to FAISS or a vector DB when needed.
- **Dynamic few-shot**: Use the current user input (or its embedding) to select the most relevant stored examples and inject them into the prompt.

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

### 3. Similarity search

Load the relevant rows (e.g. same language/dialect or all rows), reconstruct vectors from BLOBs, compute similarity between the query vector and each stored vector, then take the top-K.

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

### 4. Combine with language/dialect weighting

When stored items have language or region (e.g. French vs German, or French vs French (Quebec)), you can weight similarity by how well the stored item matches the user’s chosen target: same language and same dialect = 1.0, same language different dialect = lower, different language = much lower. Final score = phrase_similarity × language_weight. That way “French” without dialect still ranks French examples above German.

## Translation example: few-shot from vector search

The example script implements this pattern for translation:

1. **Single prompt for target**: User enters “French (Quebec)” or “spanish mexico”; one LLM call normalizes to canonical language name, ISO 639-1 code, and optional ISO 3166-1 region code.
2. **Per phrase**: Embed the source phrase, query the SQLite store for all (or filtered) translations, compute cosine similarity, apply language/dialect weights, take top-K.
3. **Few-shot prompt**: Build the chat prompt with system message, the top-K (source → translation, notes) as examples, then the current phrase. Call the chat API for translation + notes.
4. **Store new result**: Save the new (source, translation, notes, embedding) in the DB for future retrieval.

So the database grows over time and later runs get better few-shot context for the same language.

**Example**: [embeddings_vector_search.py](../src/embeddings_vector_search.py)

## Best practices

- **Normalize vectors** when using cosine similarity (or use APIs that return normalized vectors).
- **Batch embedding calls** when you have many texts to embed.
- **Store embedding dimension** with each row if you might switch models or mix dimensions.
- **Weight by metadata** (e.g. language/dialect) so retrieval respects user context, not only semantic similarity.
- **Start with simple storage** (SQLite, in-memory); move to FAISS or a vector DB when scale or latency requires it.
- **Use standard identifiers** (e.g. ISO 639-1, ISO 3166-1) for language and region so normalization and weighting stay consistent.

## When to use

- Semantic search over your own content (documents, translations, support answers).
- Dynamic few-shot or example selection for prompts.
- RAG (retrieval-augmented generation): embed documents, embed query, retrieve, then generate with retrieved context.
- Clustering, deduplication, or recommendation when items have a text component.

## Related patterns

- **RAG (Retrieval-Augmented Generation)**: Uses embeddings and vector search to retrieve context before generation; see [learning_progression.md](./learning_progression.md) Pattern 8.
- **Understanding Models – Embedding Models**: Technical details on embedding APIs and similarity; see [understanding_models.md](./understanding_models.md#embedding-models).

## Documentation links

- **OpenAI Embeddings Guide**: https://platform.openai.com/docs/guides/embeddings
- **OpenAI Embedding Models**: https://platform.openai.com/docs/models/embeddings
- **RAG with OpenAI**: https://cookbook.openai.com/examples/rag_with_openai_embeddings
