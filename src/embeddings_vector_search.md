# Embeddings & Vector Search (Pattern 6)

Side-by-side description of the algorithm and theory behind `embeddings_vector_search.py`: an interactive translator that uses **embedding-based retrieval** to find similar past translations and injects them as **few-shot examples** into the prompt.

---

## 1. Overview

**Use case:** The user chooses a target language (and optional region/dialect) once, then enters phrases to translate. For each phrase, the system:

1. **Normalizes** the target once per session (e.g. "french quebec" → French, `fr`, `CA`, "French (Quebec / Canadian French)") via one LLM call at startup.
2. **Embeds** the source phrase and **retrieves** the most similar past translations in that language (filter-then-rank).
3. **Translates** with a chat model, using those retrieved examples as few-shot (source → translation + notes) so the model can match style and dialect.

**Why embeddings + vector search?** Similar phrases (e.g. "How are you?" and "How's it going?") should get similar translations and notes. Embeddings map text to vectors so we can measure **semantic similarity** and reuse the best past examples instead of picking at random or by keyword.

---

## 2. Algorithm Pipeline

```
User: target "French (Quebec)"  →  parse_language_region  →  (French, fr, CA, "French (Quebec / Canadian French)")
User: phrase "I'm hungry"       →  get_embedding(phrase)  →  query vector q
                                →  get_similar_translations(conn, French, CA, q, top_k=3)
                                     │
                                     ├── Load all rows from DB (same embedding dimension)
                                     ├── Hard filter: keep only target_language == "French"
                                     ├── Cosine similarity: q vs each row's embedding
                                     ├── Score = similarity × dialect_weight(row_region, CA)
                                     └── Return top_k (source, translation, notes)
                                →  translate_phrase(..., few_shot_examples=retrieved)
                                →  Store (phrase, result, embedding) in DB for future retrieval
```

- **One-time:** Language/region parsing (structured LLM output).
- **Per phrase:** Embed phrase → retrieve similar same-language examples → translate with few-shot → store result and its embedding.

---

## 3. Theory

### 3.1 Text embeddings

An **embedding model** maps a piece of text to a fixed-size vector of real numbers (e.g. 1536 dimensions for `text-embedding-3-small`). The training objective is such that:

- Semantically **similar** texts (same topic, intent, or phrasing) end up **close** in the vector space.
- **Unrelated** texts end up **far** apart.

So we can treat “similarity of meaning” as “proximity in embedding space” and use geometry (e.g. cosine similarity) to rank candidates.

### 3.2 Cosine similarity

For a query vector **q** and a set of vectors **v₁, v₂, …** (each a row), we use **cosine similarity**:

$$\text{sim}(\mathbf{q}, \mathbf{v}_i) = \frac{\mathbf{q} \cdot \mathbf{v}_i}{\|\mathbf{q}\| \,\|\mathbf{v}_i\|}$$

- Value in **[-1, 1]**; typically in **[0, 1]** for normalized embeddings.
- Depends on **direction**, not length: good when embeddings are (approximately) length-normalized.
- Implementation: `(vectors @ q) / (norms_v * norm_q)` in NumPy.

Higher score ⇒ more semantically similar ⇒ better candidate for few-shot.

### 3.3 Filter-then-rank (metadata + vector search)

Standard practice in vector search:

1. **Filter** the corpus by **metadata** (e.g. “only French”) so the candidate set is relevant.
2. **Rank** the filtered set by **vector similarity** (and optionally by other weights).

Here:

- **Language = hard filter:** We only consider rows with `target_language == user's target language`. No results from other languages.
- **Region = soft weight:** Among same-language rows, we multiply similarity by a **dialect weight**:
  - Same region: 1.0  
  - One or both regions empty: 0.7  
  - Same language, different region: 0.5  

So we never show off-language examples, and within the same language we prefer same-dialect examples but still allow others (e.g. generic French or French (FR) when the user asked for French (CA)).

### 3.4 Final score and top-k

For each row in the filtered (same-language) set:

$$\text{score}_i = \text{cosine}(\mathbf{q}, \mathbf{v}_i) \times \text{dialect\_weight}(\text{row}_i, \text{user\_region})$$

We sort by `score` descending and take the **top-k** (e.g. 3) as few-shot examples.

---

## 4. Few-shot translation

The retrieved list is a sequence of **(source, translation, notes)**. These are turned into message pairs:

- **User:** `Translate to French (Quebec / Canadian French): <source>`
- **Assistant:** `Translation: <translation>\nNotes: <notes>`

Then the **current** phrase is appended as one more user message. The model sees the target (from `target_description`), the style of past translations, and the new phrase, and returns a new translation + notes (e.g. via structured output).

So the “algorithm” for translation is: **retrieve by semantic similarity (and dialect weight), then few-shot prompt**.

---

## 4.1 Example: same phrase, two dialects

The following shows one phrase—*hello there buddy*—translated first into **French**, then into **Quebec French**. The second run retrieves the first translation as a few-shot example and produces a dialect-appropriate variant.

**Target 1: French**

- **User target:** `french` → normalized to *French* (code `fr`, no region).
- **Phrase:** `hello there buddy`
- **Prompt (simplified):** System + user: *Translate to French: hello there buddy* (no few-shot yet if the store had no similar French examples).
- **Response (structured):**
  - `translated_text`: **Salut, mon pote**
  - `notes`: *The phrase 'hello there buddy' is casual and friendly. 'Salut' is a common informal greeting in French, equivalent to 'hi' or 'hello.' 'Mon pote' translates to 'my buddy' or 'my friend,' commonly used among friends in French-speaking contexts.*

That translation is stored with its source embedding. On the next run, a similar phrase in the same (or related) language can retrieve it.

**Target 2: French (Quebec)**

- **User target:** `quebec french` → normalized to *French (Quebec / Canadian French)* (code `fr`, region `CA`).
- **Phrase:** `hello there buddy` (same as above).
- **Prompt:** System + **one few-shot pair** (e.g. the same phrase from a prior French run: user *Translate to French (Quebec / Canadian French): hello there buddy* → assistant *Translation: Salut, mon pote … Notes: …*) + current user: *Translate to French (Quebec / Canadian French): hello there buddy*. The script reports: *Few-shot: used 1 similar past translation(s)*.
- **Response (structured):**
  - `translated_text`: **Salut, mon chum**
  - `notes`: *In Quebec French, 'mon chum' is a very common informal way to say 'my buddy' or 'my friend,' especially among men. 'Salut' is an informal greeting equivalent to 'hello' or 'hi.' This makes the phrase casual and friendly, fitting the tone of 'hello there buddy.'*

So: **same phrase**, **two targets**. The first run produces standard French *mon pote*; the second run uses that (or a similar) past translation as few-shot and adapts to Quebec French *mon chum*, with notes that explain the dialect choice.

**A new region reuses both.** Suppose the user now picks a third target, e.g. **French (Belgium)** (code `fr`, region `BE`). The store has two French rows for this phrase: one with no region (standard *Salut, mon pote*), one with region `CA` (Quebec *Salut, mon chum*). Retrieval works like this:

1. **Filter:** Keep only rows where `target_language == "French"` → both rows pass.
2. **Embed:** The new phrase is the same (*hello there buddy*), so its embedding matches the stored source embeddings very closely (cosine ≈ 1.0).
3. **Score:** For each row, score = cosine(q, v) × dialect_weight(row_region, user_region):
   - Standard French (region empty) vs user region `BE`: **0.7** (one or both regions empty).
   - Quebec (region `CA`) vs user region `BE`: **0.5** (same language, different region).
4. **Top-k:** Sorted by score, the standard-French example ranks first (e.g. 1.0 × 0.7), the Quebec example second (e.g. 1.0 × 0.5). With `top_k=3`, both are returned as few-shot examples.
5. **Prompt:** The model sees *Translate to French (Belgium): hello there buddy* plus two example pairs (standard *mon pote* and Quebec *mon chum*), and can produce a Belgian-appropriate translation and notes.

So a new dialect in the same language does **not** need prior examples for that exact region: it reuses the same-language store, ranked by similarity and dialect weight, and the LLM adapts to the requested region from the few-shot style and the target description.

---

## 5. Data model

**Table `translations`** (SQLite):

| Column            | Role |
|-------------------|------|
| target_language   | Canonical name (e.g. French); used for **filter**. |
| target_dialect    | Region code (e.g. CA); used for **weight**. |
| source_text       | Original phrase. |
| translated_text   | Model output. |
| notes             | Cultural/contextual notes. |
| embedding         | BLOB (float32 array); used for similarity. |
| embedding_dim     | Dimension of embedding (for compatibility). |
| created_at        | Timestamp. |

- **Embedding** is always for **source_text** (the phrase the user typed). We compare “current phrase” to “past source phrases” to choose which past **(source, translation, notes)** to reuse.

---

## 6. Summary

| Step              | What we do |
|-------------------|------------|
| Target parsing    | One LLM call → language name, code, region code, human-readable description. |
| Embedding         | Map source phrase to vector via embeddings API. |
| Retrieval         | **Filter** to same language; **rank** by cosine similarity × dialect weight; take **top-k**. |
| Translation       | Few-shot chat with (target description + retrieved examples + current phrase); structured output (translation + notes). |
| Storage           | Append (target_language, target_dialect, source, translation, notes, embedding, dim) to DB. |

The “theory” is: **embeddings** give a semantic vector space, **cosine similarity** measures phrase similarity, **filter-then-rank** uses language as a hard constraint and region as a soft weight, and **few-shot** uses the retrieved examples to steer the translator toward the right style and dialect.
