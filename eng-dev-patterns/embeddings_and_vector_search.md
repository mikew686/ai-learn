# Embeddings / Vector Search

## Overview

Dense vector representations (embeddings) and similarity search are used to find relevant content by meaning. The pattern underlies semantic search, retrieval-augmented generation (RAG), and dynamic few-shot selection. It covers embedding generation, vector storage, similarity metrics, and their combination—e.g. storing examples and retrieving the most similar to build few-shot prompts.

## Description

Embedding models convert text into fixed-length vectors that capture semantic meaning. Similar texts yield similar vectors; items are ranked by similarity (e.g. cosine similarity or dot product) to implement semantic search. Vectors are stored in a local or remote store (SQLite, FAISS, pgvector, or a dedicated vector DB). The workflow is consistent: embed the query, compare to stored vectors, take the top-K.

**Key concepts**

- **Embedding generation**: An embeddings API (e.g. OpenAI `text-embedding-3-small`) is called with one or many texts; one vector per text is returned.
- **Vector similarity**: Cosine similarity or dot product (often with normalized vectors) ranks by relevance.
- **Vector storage**: From simple (SQLite BLOB, in-memory) to scaled (FAISS, vector DBs) for persistence and throughput.
- **Dynamic few-shot**: The current input’s embedding selects the most relevant stored examples, which are injected into the prompt.
- **Exact-prompt embedding**: For few-shot retrieval, the same text that appears in the user message is embedded (e.g. “Translate to French (Quebec): &lt;phrase&gt;”), so query and stored items share the same surface form.
- **Target normalization (vector-backed)**: Free-form input (e.g. “French Quebec”) is embedded and compared to a small target store. If the nearest row is within a distance threshold, its canonical fields are used and the LLM is skipped; otherwise the LLM parses and the result is stored for future lookups.

Similarity is often cosine; for normalized vectors, dot product is equivalent. Many indexes (e.g. pgvector) use L2 or cosine distance; ranking remains consistent when vectors are normalized. Storage options range from SQLite + BLOB and Chroma to Postgres + pgvector, FAISS, Pinecone, Weaviate, and Qdrant.

## Translation Example

OpenAI embeddings with Postgres and pgvector: store translation pairs with their embeddings, then find the nearest past translations (by exact-prompt form) to use as few-shot examples before calling the chat model.

```python
# Requires: openai, psycopg2, pgvector. DB: CREATE EXTENSION vector;
# Table: translations(id, target_lang, source_text, translated_text, prompt_text, embedding vector(1536))
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

client = OpenAI()
conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)

def store_translation(target_lang: str, source: str, translated: str):
    prompt_text = f"Translate to {target_lang}: {source}"
    r = client.embeddings.create(model="text-embedding-3-small", input=prompt_text)
    vec = r.data[0].embedding
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO translations (target_lang, source_text, translated_text, prompt_text, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            (target_lang, source, translated, prompt_text, vec),
        )
    conn.commit()

def get_similar_translations(target_lang: str, source: str, top_k: int = 3):
    prompt_text = f"Translate to {target_lang}: {source}"
    r = client.embeddings.create(model="text-embedding-3-small", input=prompt_text)
    qvec = r.data[0].embedding
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_text, translated_text FROM translations "
            "WHERE target_lang = %s ORDER BY embedding <=> %s::vector LIMIT %s",
            (target_lang, qvec, top_k),
        )
        return cur.fetchall()

# Example: fetch similar past translations for few-shot, then call chat API with them
# rows = get_similar_translations("French", "Hello, how are you?")
# Build messages with user/assistant pairs from rows, then add current user message; call chat.
```

Exact-prompt embedding: the same string (“Translate to {target_lang}: {source}”) is used for both storage and query so similarity is in the same task+phrase space. The `<=>` operator is pgvector’s cosine distance.
