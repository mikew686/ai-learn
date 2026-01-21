# Context Engineering Technologies — 2025 Convention Highlights
### Featuring Redis LangCache · TigerData pg_textsearch · AWS Strands

Context engineering emerged as one of the **core technical disciplines** at the 2025 DeepLearning.AI Dev Convention. While model capabilities continue to accelerate, the real competitive advantage increasingly comes from **how effectively developers shape, cache, route, and evaluate context** for LLMs and agentic systems.  

Three technologies stood out for offering practical, production-grade approaches to context management:  
**Redis LangCache**, **TigerData (pg_textsearch hybrid retrieval)**, and **AWS Strands**.

---

## 1. Redis LangCache — Semantic Caching for Token-Efficient Agent Loops

Redis introduced **LangCache**, a semantic caching layer designed to optimize LLM and agent workflows.

### Key capabilities
- **Vector-embedding semantic similarity** to reuse prior LLM results  
- **Probabilistic caching** instead of deterministic lookup tables  
- **Latency reduction** for multi-step agent loops  
- **Token cost savings** by preventing redundant model calls  

### Evaluation Matters  
Redis emphasized treating semantic caches as ML components requiring metrics such as:

- Precision / Recall  
- F1 Score  
- Cache hit rate  
- False-positive impact on agent reliability  

**Why it matters:**  
Agent orchestration frameworks (Strands, Genspark, AI21) often perform dozens or hundreds of calls per task. LangCache becomes a crucial **cost and performance governor**, shaping the effective context each call receives.

---

## 2. TigerData + Postgres pg_textsearch — Hybrid Retrieval Without a Vector Database

TigerData demonstrated how Postgres, with **pg_textsearch**, can be a powerful hybrid retrieval engine.

### Core features
- Combines **full-text lexical search** and **vector similarity** in a single database  
- Improves **retrieval precision**, reducing hallucinations  
- Eliminates the need for a standalone vector DB  
- Simplifies architecture while improving reliability  

### Impact on Context Quality
Hybrid retrieval allows developers to tightly control:

- Which documents appear in the context window  
- Relevance rankings  
- Noise reduction  
- Consistency across repeated queries  

**Why it matters:**  
Most RAG failures are caused by **bad retrieval**, not bad models. TigerData’s approach dramatically improves context quality by staying inside Postgres and reducing “garbage in, garbage out.”

---

## 3. AWS Strands — Model-Driven Multi-Agent Orchestration and Context Routing

AWS Strands—presented in Nicholas Clegg’s session—illustrates the shift from hand-built workflows to **model-driven orchestration**.

### What Strands provides
- **Model-directed reasoning loops**  
- Built-in **planning, delegation, and multi-agent collaboration**  
- **Dynamic context routing** for each agent and tool  
- Smooth **local → production** parity for deployment  

### How Strands Changes Context Engineering
Strands intelligently manages:

- What context each agent sees  
- Which tool definitions are injected into prompts  
- When intermediate outputs should be cached, chunked, or discarded  

**Why it matters:**  
Strands shows the future of agent workflows: systems that **generate, prune, and route context autonomously**, reducing complexity for developers.

---

# Cross-Technology Insight: Context Engineering as a First-Class Discipline

Across Redis, TigerData, and Strands, common themes emerged:

### 1. Cost and performance are now *context* problems  
Token expenses and latency in agent systems depend more on retrieval quality and caching strategy than on compute throughput.

### 2. Hybrid retrieval > pure vector retrieval  
TigerData demonstrated that combining lexical and semantic search dramatically reduces hallucinations and improves context cleanliness.

### 3. Semantic caching is essential for agents  
Redis LangCache prevents repeated, costly queries during long-horizon reasoning loops.

### 4. Model-driven orchestration changes context flow  
AWS Strands dynamically manages context, making agents more adaptive and production-ready.

### 5. Evaluation and context shaping drive reliability  
Context engineering is now measurable, testable, and central to AI system correctness.

---

# Context Engineering: Relevance for content operations and ingestion pipelines

- **TigerData** provides high-precision retrieval ideal for curriculum metadata, structured documents, and clean RAG context.  
- **Redis LangCache** can reduce LLM cost for repetitive extraction tasks, similarity checks, and agent-style ingestion loops.  
- **Strands** aligns with future multi-agent workflows for curriculum ingestion, normalization, and tool-based document processing.

Together, these technologies outline a roadmap for building **fast, reliable, context-aware AI systems** optimized for production environments.

---

*Generated for the 2025 Deeplearning.AI Dev Convention Notes Repository*
