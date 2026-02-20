# Foundations: Embeddings and Vector Search

Embeddings and vector search form the geometric backbone of modern AI systems. They allow meaning to be represented as points in high-dimensional space, enabling semantic similarity, retrieval, clustering, recommendation, and memory systems. When combined with large language models, embeddings make Retrieval-Augmented Generation (RAG), semantic search, and long-context reasoning practical at scale.

This section explains:

1. What embeddings are (theory)
2. How embedding geometry emerges during training
3. How vector search works at scale
4. How ANN algorithms enable production systems
5. How embeddings integrate with LLM reasoning and retrieval

Embeddings can be understood as a learned coordinate system for meaning. Vector search is fast navigation within that coordinate system.

---

## Suggested Reading (Optional but Helpful)

### Foundational Theory

* Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (Word2Vec)
* Pennington et al., *GloVe: Global Vectors for Word Representation*
* Vaswani et al., *Attention Is All You Need*

### Contrastive & Representation Learning

* Oord et al., *Representation Learning with Contrastive Predictive Coding*
* Chen et al., *SimCLR*
* Reimers & Gurevych, *Sentence-BERT*

### Vector Search & ANN

* Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW*
* Jégou et al., *Product Quantization for Nearest Neighbor Search*

### Modern Retrieval Systems

* Lewis et al., *Retrieval-Augmented Generation (RAG)*
* Karpukhin et al., *Dense Passage Retrieval (DPR)*

These works collectively explain how semantic geometry emerges, how it is searched efficiently, and how it integrates into modern AI systems.

---

# 1. What Is an Embedding?

An embedding is a mapping:

[
\text{Object} \rightarrow \mathbb{R}^n
]

It converts discrete symbolic objects into continuous vectors in high-dimensional space.

Examples of objects:

* Words
* Sentences
* Paragraphs
* Images
* Code snippets
* Users
* Products
* Documents

The embedding vector captures semantic meaning as geometry.

### Core Idea

Similar meaning → nearby vectors in space.

If two pieces of text are similar in meaning, their embedding vectors will be close under a distance metric like:

* Cosine similarity
* Dot product
* Euclidean distance

---

# 2. Why Embeddings Exist (Theoretical Motivation)

Neural networks operate on numbers, not symbols.

Text is discrete:

* Words
* Tokens
* Characters

Neural networks require continuous representations.

Embeddings solve this by:

1. Converting discrete tokens into dense numeric vectors
2. Allowing gradient descent to shape meaning
3. Making similarity computable via geometry

---

# 3. Historical Progression

## 3.1 One-Hot Encoding (Early)

Each word = vector with one 1 and rest 0.

Problems:

* Huge dimensionality
* No similarity between related words
* “cat” and “dog” are orthogonal

---

## 3.2 Word2Vec (Distributional Hypothesis)

Key idea:

“You shall know a word by the company it keeps.”

Words appearing in similar contexts develop similar vectors.

Trained via:

* Predict next word (Skip-gram)
* Predict context from word (CBOW)

Result:

* Semantic geometry emerges
* King − Man + Woman ≈ Queen

---

## 3.3 Transformer-Based Embeddings

Modern embedding models:

* Use transformers
* Train on massive corpora
* Capture sentence and document meaning
* Encode long-range context

Unlike Word2Vec:

* Contextual
* Deep semantic representation
* Multi-layer feature composition

---

# 4. How Embeddings Are Learned

Embeddings are trained via self-supervised objectives.

Common objectives:

## 4.1 Language Modeling

Predict next token.
Embedding layer adjusts to reduce loss.

## 4.2 Contrastive Learning

Pull similar items together, push dissimilar apart.

Example:

* Query and matching document → close
* Query and unrelated document → far

Example loss:

[
\text{Loss} = -\log \frac{\exp(sim(q, d^+))}{\sum \exp(sim(q, d_i))}
]

This directly shapes geometric structure.

---

# 5. Geometry of Meaning

After training, embedding space has properties:

## 5.1 Clustering

Similar topics cluster:

* Legal documents
* Sailing discussions
* Python code
* Kubernetes configs

## 5.2 Directional Meaning

Certain directions encode abstract features:

* Formal ↔ informal
* Positive ↔ negative
* Technical ↔ narrative

## 5.3 Linear Relationships

Sometimes arithmetic approximates semantic transitions.

This is not magic.
It is a result of:

* Gradient descent
* Massive co-occurrence structure
* High-dimensional linear separability

---

# 6. Sentence and Document Embeddings

Modern embedding models produce:

* 384–1536 dimensional vectors
* Single vector per text chunk

Typical pipeline:

Text → Tokenization → Transformer → Pooling → Dense vector

Pooling strategies:

* CLS token
* Mean pooling
* Learned pooling layer

---

# 7. Vector Similarity

Given two embeddings:

[
v_1, v_2 \in \mathbb{R}^n
]

We compute similarity via:

## Cosine similarity

[
\frac{v_1 \cdot v_2}{||v_1|| ||v_2||}
]

Range: [-1, 1]

Most common metric.

---

# 8. What Is Vector Search?

Vector search =

Finding nearest vectors in high-dimensional space.

Given:

* A query embedding
* A database of stored embeddings

We retrieve the top-k closest vectors.

---

# 9. The Core Problem: High-Dimensional Nearest Neighbor

Naive search:

* Compute similarity against every vector
* O(N)

Fine for 1,000 items
Impossible for 100 million

So we use:

Approximate Nearest Neighbor (ANN) algorithms.

---

# 10. ANN Algorithms (Practical Theory)

## 10.1 HNSW (Hierarchical Navigable Small Worlds)

Most widely used.

* Graph-based structure
* Multi-layer connectivity
* Fast traversal
* Logarithmic scaling

Used in:

* Pinecone
* Weaviate
* Qdrant
* Milvus

---

## 10.2 IVF (Inverted File Index)

* Cluster vectors
* Search within relevant clusters

---

## 10.3 Product Quantization

* Compress vectors
* Approximate distance cheaply

Used for billion-scale systems.

---

# 11. Practical Embedding + Vector Search Pipeline

## Step 1: Chunk Documents

Break large text into chunks:

* 200–1000 tokens
* Slight overlap

Why?
Embedding models have input limits.

---

## Step 2: Generate Embeddings

For each chunk:

* Compute embedding
* Store vector + metadata

Metadata may include:

* Document ID
* Tags
* Timestamps
* Source reference

---

## Step 3: Store in Vector Database

Common databases:

* Pinecone
* Weaviate
* Qdrant
* Milvus
* pgvector (Postgres extension)

---

## Step 4: Query

User enters query.

Process:

1. Convert query to embedding
2. Search nearest vectors
3. Retrieve top-k chunks
4. Return results (or feed into LLM)

---

# 12. Embeddings + LLMs = Retrieval Augmented Generation (RAG)

RAG architecture:

User query
↓
Embedding
↓
Vector search
↓
Relevant documents
↓
LLM with context
↓
Answer grounded in retrieval

This:

* Reduces hallucination
* Injects fresh data
* Allows domain adaptation
* Scales knowledge beyond model training

---

# 13. Why This Works

LLMs reason over tokens,
but embeddings compress meaning.

Vector search:

* Selects relevant knowledge
* Reduces search space
* Makes reasoning tractable

It is effectively semantic indexing for language models.

---

# 14. Advanced Uses

## 14.1 Long-Term Memory Systems

Store conversation history as embeddings.
Retrieve relevant past memories.

## 14.2 Semantic Search

Better than keyword search:

* Handles paraphrase
* Handles synonyms
* Handles conceptual similarity

## 14.3 Code Search

Find similar functions by behavior, not text match.

## 14.4 Recommendation Systems

Embed:

* Users
* Products
* Behavior

Then compute nearest neighbors.

---

# 15. Failure Modes

## 15.1 Embedding Drift

Different embedding models produce incompatible spaces.

## 15.2 Poor Chunking

Too large → diluted meaning
Too small → lost context

## 15.3 Over-Retrieval

Too many irrelevant chunks reduce LLM clarity.

## 15.4 Domain Mismatch

Generic embeddings may fail on:

* Legal language
* Medical terminology
* Sailing-specific jargon

---

# 16. The Deeper View

Embeddings are:

A learned coordinate system for meaning.

Vector search is:

Fast navigation through that coordinate system.

This is geometric reasoning applied to language.

---

# 17. Connection to Your Transformer Foundations

In GPT systems:

* Token embeddings start the residual stream.
* Sentence embeddings are pooled transformer outputs.
* Retrieval augments the context window.
* Reasoning operates over retrieved tokens.

Embeddings are not separate from LLMs.
They are an abstraction of the same representational machinery.

---

# 18. Conceptual Summary

| Component       | Role                                   |
| --------------- | -------------------------------------- |
| Embedding model | Compress meaning into vectors          |
| Vector DB       | Stores high-dimensional vectors        |
| ANN index       | Enables scalable search                |
| LLM             | Performs reasoning over retrieved text |

Together, they create modern knowledge-aware AI systems.
