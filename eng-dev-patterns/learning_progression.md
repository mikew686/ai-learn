# AI Engineering Learning Progression

This document outlines a logical self-learning progression for AI engineering patterns. **Core principles come first** (understanding models, prompts, schemas, tools, embeddings); **then** patterns that tie them together (schema-driven inference, few-shot, RAG, streaming, etc.). Each pattern builds on previous concepts and uses best-practice, accessible technologies.

## Overview

This progression demonstrates a learning approach where patterns build on each other through a consistent use case. The philosophy:
- **Core before combination**: Learn how to find/call models, then prompts, schemas, tools, and embeddings as building blocks before combining them.
- **Start minimal**: Begin with OpenAI SDK only
- **Add incrementally**: Introduce new services/patterns only when needed
- **Build on previous**: Each pattern leverages earlier concepts
- **Best practices**: Use industry-standard, accessible technologies
- **Refinement through examples**: Where an example script exists in `scripts/examples/`, the pattern section links to it

### Preferred learning order

1. **Understanding Models** – How to find and call models (chat, reasoning, embeddings, etc.)
2. **Prompt Engineering** – Design effective prompts
3. **Structured Output** – Schemas and validated responses
4. **Function Calling / Tool Use** – LLMs calling external tools
5. **Schema-Driven Inference** – Use schemas as implicit prompts (more fundamental; combines prompts, schemas, tools)
6. **Embeddings / Vector Search** – Vector representations and similarity search
7. **Tying together** – Few-Shot, RAG (retrieval pipeline), Chain-of-Thought, then Streaming, Caching, Memory, Guardrails, Agents, Orchestration, Evaluation, Advanced

### Choosing Your Use Case

Throughout this progression, **translation** is used as the example use case. Choose your own use case based on your interests—whether that's code generation, content creation, data analysis, customer support, or something else entirely. Apply the same progression to your chosen use case.

---

## Pattern 1: Understanding Models

📖 [Detailed Documentation](./understanding_models.md)

**Description**: Gain familiarity with model categories (chat, reasoning, embedding, etc.) and how to find, select, and call them via APIs. Essential before applying patterns that depend on a specific model type.

**Technology**: OpenAI SDK (or OpenRouter); provider docs

**Key Concepts**:
- Chat vs. reasoning vs. embedding vs. fast/cheap models
- Model selection by task and cost
- Basic chat completion and API usage
- When to use which model class

**Translation Example**: Choosing a chat model for translation; later, choosing an embedding model for similar-phrase retrieval.

**Example Implementation**:
- Call a chat model with system/user messages
- Call an embeddings API for text vectors
- Compare chat vs. reasoning model behavior for the same prompt

**Example**: [understand_llm_models.py](../scripts/examples/understand_llm_models.py)

**Related**: [understand_embedding_models.py](../scripts/examples/understand_embedding_models.py), [understand_reasoning_model.py](../scripts/examples/understand_reasoning_model.py)

---

## Pattern 2: Prompt Engineering

📖 [Detailed Documentation](./prompt_engineering.md)

**Description**: Design effective prompts to guide LLM behavior. Learn template-based prompts, few-shot learning, role-setting, structured instructions, and output format specification.

**Technology**: OpenAI SDK

**Key Concepts**:
- System vs user prompts
- Few-shot examples
- Role-setting
- Constraint injection
- Output format specification

**Translation Example**: Basic translation prompt with role-setting ("You are an expert translator...") and format requirements.

**Example Implementation**:
- System message for role and constraints
- User message with source text and target language
- Optional few-shot examples in the prompt

**Example**: [system_prompt_example.py](../scripts/examples/system_prompt_example.py)

**Related**: [schema_driven_translation.py](../scripts/examples/schema_driven_translation.py), [tool_use_patterns.py](../scripts/examples/tool_use_patterns.py), [embeddings_vector_search.py](../scripts/examples/embeddings_vector_search.py)

---

## Pattern 3: Structured Output

📖 [Detailed Documentation](./structured_output.md)

**Description**: Enforce structured, validated responses from LLMs using schemas. Ensure outputs conform to expected formats and can be reliably parsed without manual validation.

**Technology**: OpenAI SDK + Pydantic

**Key Concepts**:
- Pydantic models for type safety
- JSON Schema validation
- Field descriptions and constraints
- Nested models and composition
- Enums for constrained values
- Arrays and lists

**Translation Example**: Translation result with structured fields (source language, target language, translated text, confidence, cultural notes) using Pydantic models.

**Example Implementation**:
- Define a Pydantic model for the translation result
- Use `client.beta.chat.completions.parse()` with `response_format=YourModel`
- Handle validation errors

**Example**: [language_assessment_structured_output.py](../scripts/examples/language_assessment_structured_output.py)

**Related**: [schema_driven_translation.py](../scripts/examples/schema_driven_translation.py), [embeddings_vector_search.py](../scripts/examples/embeddings_vector_search.py)

---

## Pattern 4: Function Calling / Tool Use

📖 [Detailed Documentation](./function_calling_tool_use.md)

**Description**: Enable LLMs to call external functions, APIs, or tools through structured interfaces. Allow models to interact with the real world beyond text generation.

**Technology**: OpenAI SDK (native function calling)

**Key Concepts**:
- Tool schema definitions (JSON Schema)
- Tool selection by LLM
- Parameter extraction and validation
- Sequential, parallel, and interleaved tool use patterns
- Tool execution and response integration

**Translation Example**: Use tools to detect source language, look up language codes, and get cultural context before translating.

**Example Implementation**:
- Define tools with name, description, and parameters
- Pass `tools` and `tool_choice` to chat completion
- Handle `message.tool_calls`, execute functions, append tool results, continue conversation

**Example**: [tool_use_patterns.py](../scripts/examples/tool_use_patterns.py)

**Related**: [schema_driven_translation.py](../scripts/examples/schema_driven_translation.py)

---

## Pattern 5: Schema-Driven Inference (Tying Together)

📖 [Detailed Documentation](./schema_driven_inference.md)

**Description**: Use structured definitions (tool schemas, Pydantic field descriptions) as implicit prompts. Reduce prompt verbosity while maintaining high-quality, validated outputs. Combines prompts, schemas, and tools. More fundamental than embeddings for core inference flow.

**Technology**: OpenAI SDK + Pydantic

**Key Concepts**:
- Tool schema descriptions as implicit instructions
- Pydantic field descriptions guide output generation
- Schema composition (enums, arrays, nested models)
- Minimal prompts with detailed schemas
- Combining explicit and implicit instructions

**Translation Example**: Minimal translation prompt where tool descriptions and Pydantic field descriptions provide most of the guidance, reducing token usage while maintaining quality.

**Example Implementation**:
- Define tools with clear descriptions; define Pydantic output model with detailed Field descriptions
- Use a short system/user prompt; let schemas communicate requirements
- Combine tool calls and structured output in one flow

**Example**: [schema_driven_translation.py](../scripts/examples/schema_driven_translation.py)

**Related**: [tool_use_patterns.py](../scripts/examples/tool_use_patterns.py) (descriptive tool schemas)

---

## Pattern 6: Embeddings / Vector Search

📖 [Detailed Documentation](./embeddings_and_vector_search.md)

**Description**: Convert text into dense vector representations (embeddings) and use similarity search to find relevant content. Foundation for semantic search and RAG.

**Technology**: OpenAI SDK (text-embedding-3-small/large) + Local vector storage (SQLite/FAISS)

**Key Concepts**:
- Embedding generation
- Vector similarity metrics (cosine, dot product)
- Approximate nearest neighbor search
- Embedding normalization
- Local vector storage (start simple, scale later)
- Filter-then-rank (e.g. filter by language, then rank by similarity)

**Translation Example**: Generate embeddings for translation examples, store them, and retrieve similar examples for few-shot learning. Find similar phrases to reuse translations.

**Example Implementation**:
- Embed source phrases and store (source, translation, notes, embedding) in SQLite
- On new phrase: embed query, filter by target language, rank by cosine similarity × dialect weight, take top-K
- Inject top-K as few-shot examples into the translation prompt

**Example**: [embeddings_vector_search.py](../scripts/examples/embeddings_vector_search.py)

---

## Pattern 7: Few-Shot / In-Context Learning

**Description**: Provide examples in prompts to guide LLM behavior without modifying the model. Leverage the model's ability to learn patterns from examples dynamically.

**Technology**: OpenAI SDK

**Key Concepts**:
- Example selection strategies
- Example ordering and formatting
- Dynamic example selection
- Semantic similarity for example retrieval (see Pattern 6)
- Balancing example count vs. token usage

**Translation Example**: Include 2-5 translation examples in the prompt to guide style, formality level, and regional variations. Dynamically select examples based on source/target language pair or semantic similarity (e.g. via embeddings).

**Example Implementation**:
- Simple few-shot prompts with translation examples
- Dynamic example selection based on language pair or embedding similarity (as in [embeddings_vector_search.py](../scripts/examples/embeddings_vector_search.py))
- Template-based few-shot prompt construction for different translation styles

---

## Pattern 8: RAG (Retrieval-Augmented Generation)

**Description**: Augment LLM inputs with relevant context retrieved from external knowledge bases or vector databases. Combine semantic search with generative capabilities. Completes the retrieval pipeline (Embeddings → Few-Shot → RAG).

**Technology**: OpenAI SDK (embeddings + chat) + Local vector storage

**Key Concepts**:
- Document chunking strategies (200-500 tokens)
- Vector database for storage and retrieval
- Semantic search and retrieval
- Context injection into prompts
- Hybrid search (keyword + semantic) - optional

**Translation Example**: Build a knowledge base of domain-specific terminology (medical, legal, technical). When translating, retrieve relevant terminology and context to ensure accurate domain-specific translations.

**Example Implementation**:
- Domain-specific translation with terminology retrieval
- Multi-language glossary retrieval for technical terms
- Context-aware translation using retrieved documents

---

## Pattern 9: Chain-of-Thought / Multi-Step Reasoning

**Description**: Encourage LLMs to show explicit step-by-step reasoning processes. Improve accuracy on complex problems requiring logical reasoning.

**Technology**: OpenAI SDK (with reasoning models: o3-mini, o4-mini)

**Key Concepts**:
- Explicit reasoning prompts ("Let's think step by step")
- Intermediate reasoning steps
- Self-verification
- Problem decomposition
- Reasoning model selection (o-series vs GPT models)

**Translation Example**: For complex or ambiguous phrases, the model reasons through: 1) detecting language and dialect, 2) identifying cultural context, 3) considering multiple translation options, 4) selecting best option with explanation.

**Example Implementation**:
- Multi-step translation reasoning for idiomatic expressions
- Explicit analysis of cultural nuances before translation
- Verification step to check translation accuracy

📖 See also: [llm_vs_reasoning_problems.md](./llm_vs_reasoning_problems.md)

---

## Pattern 10: Streaming / Progressive Generation

**Description**: Stream LLM responses token-by-token for real-time user feedback. Improve perceived latency and user experience.

**Technology**: OpenAI SDK (streaming API)

**Key Concepts**:
- Token-by-token streaming
- Partial response handling
- Cancellation support
- Server-Sent Events (SSE) patterns
- Progressive UI updates

**Translation Example**: Stream translation results as they're generated, allowing users to see progress for long texts. Show partial translations for multi-sentence inputs.

**Example Implementation**:
- Real-time translation streaming
- Progressive display of translation results
- Cancellable long translation operations

---

## Pattern 11: Caching / Optimization

**Description**: Cache LLM responses, embeddings, or intermediate results to reduce latency, costs, and improve performance.

**Technology**: OpenAI SDK + Local caching (SQLite/Redis) or semantic caching library

**Key Concepts**:
- Response caching (exact match)
- Embedding caching
- Semantic caching (similarity-based)
- Cache invalidation strategies
- Cost optimization through caching

**Translation Example**: Cache common phrase translations. Use semantic caching to return cached translations for similar (but not identical) phrases. Cache embeddings for frequently translated text.

**Example Implementation**:
- Translation cache for common phrases
- Semantic cache for similar translation requests
- Embedding cache for RAG retrieval
- Cache hit rate monitoring and optimization

---

## Pattern 12: Memory / State Management

**Description**: Persist context, conversation history, and state across LLM interactions. Enable long-term memory and context retention.

**Technology**: OpenAI SDK + Local storage (SQLite) or simple file-based storage

**Key Concepts**:
- Short-term memory (conversation history)
- Long-term memory (persistent storage)
- Memory summarization
- Context window management
- Session management

**Translation Example**: Remember user's preferred translation style, common language pairs, and terminology preferences across sessions. Maintain conversation context for multi-turn translation discussions.

**Example Implementation**:
- Conversational translation assistant with session memory
- User preference learning (formality, regional variants)
- Multi-turn translation refinement with context

---

## Pattern 13: Guardrails / Safety

**Description**: Content filtering, output validation, and safety checks to prevent harmful, biased, or inappropriate outputs. Recommended for production deployments.

**Technology**: OpenAI SDK + Content moderation API or simple validation patterns

**Key Concepts**:
- Content moderation
- Output validation
- PII detection and redaction
- Rate limiting
- Safety prompt injection

**Translation Example**: Validate translations don't contain harmful content, detect and handle PII in source text, implement rate limiting for API protection.

**Example Implementation**:
- Content filtering for translation inputs
- PII detection and handling in multilingual text
- Rate limiting for translation API
- Output validation with Pydantic schemas

---

## Pattern 14: Agentic Systems

**Description**: Autonomous systems that can plan, execute actions, and adapt based on results. Agents use tools, memory, and reasoning to achieve goals.

**Technology**: OpenAI SDK + Custom orchestration logic

**Key Concepts**:
- Planning capabilities
- Tool/action execution
- Memory/state management
- Goal tracking
- Reflection and adaptation
- Safety limits (max steps, timeouts)

**Translation Example**: Translation agent that plans multi-step translation workflow: 1) detect language, 2) retrieve domain context, 3) check cache, 4) translate with appropriate tools, 5) verify quality, 6) adapt based on user feedback.

**Example Implementation**:
- Autonomous translation agent with planning
- Multi-step translation workflow with tool orchestration
- Self-correcting translation with quality verification
- Adaptive translation based on user feedback

---

## Pattern 15: Orchestration / Workflow Management

**Description**: Coordinate multi-step LLM workflows with proper error handling, state management, and retry logic. Ensures reliable execution of complex processes.

**Technology**: OpenAI SDK + Custom workflow engine or simple state machine

**Key Concepts**:
- Sequential and parallel execution
- Error handling and recovery
- State persistence
- Conditional branching
- Retry mechanisms
- Workflow visualization

**Translation Example**: Orchestrate batch translation workflow: 1) validate inputs, 2) detect languages in parallel, 3) retrieve context, 4) translate with retry on failure, 5) validate outputs, 6) handle errors gracefully.

**Example Implementation**:
- Batch translation pipeline with error recovery
- Multi-document translation workflow
- Translation approval workflow with validation steps
- Parallel translation of independent text segments

---

## Pattern 16: LLM-as-a-Judge (Evaluation)

**Description**: Use LLMs to evaluate the quality, correctness, or alignment of outputs. Enable automated quality assessment at scale.

**Technology**: OpenAI SDK (separate judge model)

**Key Concepts**:
- Rubric-based evaluation
- Comparative evaluation
- Fact-checking
- Alignment checking
- Quality scoring
- Evaluation metrics

**Translation Example**: Use a separate LLM to evaluate translation quality: accuracy, fluency, cultural appropriateness, terminology correctness. Compare multiple translation options and select the best.

**Example Implementation**:
- Automated translation quality assessment
- Comparative evaluation of translation options
- A/B testing for different translation prompts
- Quality scoring and improvement feedback

---

## Pattern 17: Advanced Patterns

**Description**: Advanced optimization and production patterns for scale and efficiency.

**Technology**: Various (depends on pattern)

**Key Patterns**:
- **Hybrid Search**: Combine keyword (BM25) and semantic (vector) search
- **Prompt Compression**: Reduce prompt size while preserving information
- **Model Routing**: Select or combine multiple models based on task/cost
- **Semantic Caching**: Cache based on semantic similarity
- **Tree-of-Thought**: Explore multiple reasoning paths

**Translation Example**:
- Hybrid search for terminology retrieval (exact matches + semantic similarity)
- Compress long context while preserving translation quality
- Route simple translations to cheaper models, complex ones to advanced models
- Semantic cache for similar translation requests
- Explore multiple translation paths and select best result

**Example Implementation**:
- Production-ready translation system with hybrid search
- Cost-optimized prompt compression for long documents
- Multi-model routing for translation tasks
- Advanced semantic caching strategies

---

## Technology Stack Summary

### Core (Patterns 1–5): Building blocks
- **Understanding Models**: Provider docs, OpenAI/OpenRouter SDK
- **Prompts**: OpenAI SDK (system/user messages)
- **Structured Output**: OpenAI SDK + Pydantic
- **Tools**: OpenAI SDK (native function calling)
- **Schema-Driven Inference**: Combines prompts, schemas, tools (implicit prompts)

### Tying together (Patterns 6–9)
- **Embeddings**: OpenAI embeddings API + SQLite (or FAISS / vector DB)
- **Few-Shot**: Prompts + optional embeddings for example selection
- **RAG**: Embeddings + vector store + chat (completes retrieval pipeline)
- **Chain-of-Thought**: Reasoning models or prompting

### Production-oriented (Patterns 10–17)
- **Streaming, Caching, Memory, Guardrails**: SDK + local or external services
- **Agents, Orchestration, Evaluation, Advanced**: Built on previous patterns

---

## Self-Learning Path

This progression is designed for self-paced learning:
- **Core first**: Patterns 1–5 establish how to call models, prompt them, get structured output, use tools, and apply schema-driven inference (schemas as implicit prompts).
- **Then combination**: Pattern 6 (Embeddings) adds vector search; Patterns 7–8 add few-shot and RAG (retrieval pipeline); Pattern 9 adds chain-of-thought reasoning.
- **Then production**: Patterns 10–17 add streaming, caching, memory, safety, agents, orchestration, evaluation, and advanced topics.

Each pattern builds on previous concepts. Choose a use case that interests you and apply these patterns to refine your system. The translation examples serve as a reference for how patterns combine.

### Suggested Learning Order

**Core principles** (Patterns 1–5):
1. Understanding Models
2. Prompt Engineering
3. Structured Output
4. Function Calling / Tool Use
5. Schema-Driven Inference

**Tying together** (Patterns 6–9):
6. Embeddings / Vector Search
7. Few-Shot / In-Context Learning
8. RAG (Retrieval-Augmented Generation)
9. Chain-of-Thought / Reasoning

**Production** (Patterns 10–17):
10. Streaming
11. Caching / Optimization
12. Memory / State Management
13. Guardrails / Safety
14. Agentic Systems
15. Orchestration / Workflows
16. LLM-as-a-Judge
17. Advanced Patterns

---

## Notes

- **Core before combination**: Learn model overview, prompts, schemas, tools, and schema-driven inference before embeddings and patterns that depend on vector search (few-shot retrieval, RAG).
- **Example and Related**: **Example** is the primary script that best demonstrates the pattern (one per pattern). **Related** lists any other `scripts/examples/` scripts that use this pattern as a secondary technology. Use Example to learn the pattern; Related to see it in context elsewhere.
- **Choose your own use case**: Translation is the running example; apply the same order to your use case.
- **Best practices**: Use industry-standard, accessible technologies. Incremental complexity; add services only when needed.

---

*Last Updated: January 2026*
