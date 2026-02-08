# AI Engineering Learning Progression

This document outlines a logical self-learning progression for AI engineering patterns, starting with foundational techniques and building toward advanced production systems. Each pattern builds on previous concepts and uses best-practice, accessible technologies.

## Overview

This progression demonstrates a learning approach where patterns build on each other through a consistent use case. The philosophy:
- **Start minimal**: Begin with OpenAI SDK only
- **Add incrementally**: Introduce new services/patterns only when needed
- **Build on previous**: Each pattern leverages earlier concepts
- **Best practices**: Use industry-standard, accessible technologies
- **Refinement through examples**: All examples build on a single use case with increasing sophistication

### Choosing Your Use Case

Throughout this progression, **translation** is used as an example to demonstrate how patterns combine and build on each other. Choose your own use case based on your interests—whether that's code generation, content creation, data analysis, customer support, or something else entirely.

The translation example shows how each pattern refines the system:
- Start with basic prompt-based translation
- Add structured outputs for reliable parsing
- Integrate tool calling for language detection
- Use schema-driven inference to reduce verbosity
- Add few-shot examples for consistency
- Build RAG for domain-specific terminology
- Add streaming for real-time feedback
- And so on...

This approach demonstrates how patterns combine and build on each other in a real-world scenario. Apply the same progression to your chosen use case.

---

## Pattern 1: Prompt Engineering

📖 [Detailed Documentation](../eng-dev-patterns/prompt_engineering.md)

**Description**: Design effective prompts to guide LLM behavior. Learn template-based prompts, few-shot learning, role-setting, structured instructions, and output format specification.

**Technology**: OpenAI SDK

**Key Concepts**:
- System vs user prompts
- Few-shot examples
- Role-setting
- Constraint injection
- Output format specification

**Translation Example**: Basic translation prompt with role-setting ("You are an expert translator...") and format requirements.

**Example Files**: `src/prompts.py`, `src/system_role_example.py`

---

## Pattern 2: Structured Output

📖 [Detailed Documentation](../eng-dev-patterns/structured_output.md)

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

**Example Files**: `src/structured_output_example.py`

---

## Pattern 3: Function Calling / Tool Use

📖 [Detailed Documentation](../eng-dev-patterns/function_calling_tool_use.md)

**Description**: Enable LLMs to call external functions, APIs, or tools through structured interfaces. Allow models to interact with the real world beyond text generation.

**Technology**: OpenAI SDK (native function calling)

**Key Concepts**:
- Tool schema definitions (JSON Schema)
- Tool selection by LLM
- Parameter extraction and validation
- Sequential, parallel, and interleaved tool use patterns
- Tool execution and response integration

**Translation Example**: Use tools to detect source language, look up language codes, and get cultural context before translating.

**Example Files**: `src/tool_use_example.py`

---

## Pattern 4: Schema-Driven Inference

📖 [Detailed Documentation](../eng-dev-patterns/schema_driven_inference.md)

**Description**: Use structured definitions (tool schemas, Pydantic field descriptions) as implicit prompts. Reduce prompt verbosity while maintaining high-quality, validated outputs.

**Technology**: OpenAI SDK + Pydantic

**Key Concepts**:
- Tool schema descriptions as implicit instructions
- Pydantic field descriptions guide output generation
- Schema composition (enums, arrays, nested models)
- Minimal prompts with detailed schemas
- Combining explicit and implicit instructions

**Translation Example**: Minimal translation prompt where tool descriptions and Pydantic field descriptions provide most of the guidance, reducing token usage while maintaining quality.

**Example Files**: `src/translation.py` (combines tools + structured output + schema-driven inference)

---

## Pattern 5: Few-Shot / In-Context Learning

**Description**: Provide examples in prompts to guide LLM behavior without modifying the model. Leverage the model's ability to learn patterns from examples dynamically.

**Technology**: OpenAI SDK

**Key Concepts**:
- Example selection strategies
- Example ordering and formatting
- Dynamic example selection
- Semantic similarity for example retrieval
- Balancing example count vs. token usage

**Translation Example**: Include 2-5 translation examples in the prompt to guide style, formality level, and regional variations. Dynamically select examples based on source/target language pair.

**Example Implementation**:
- Simple few-shot prompts with translation examples
- Dynamic example selection based on language pair similarity
- Template-based few-shot prompt construction for different translation styles

---

## Pattern 6: Chain-of-Thought / Multi-Step Reasoning

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

---

## Pattern 7: Embeddings / Vector Search

📖 [Detailed Documentation](./embeddings_and_vector_search.md)

**Description**: Convert text into dense vector representations (embeddings) and use similarity search to find relevant content. Foundation for semantic search and RAG.

**Technology**: OpenAI SDK (text-embedding-3-small/large) + Local vector storage (SQLite/FAISS)

**Key Concepts**:
- Embedding generation
- Vector similarity metrics (cosine, dot product)
- Approximate nearest neighbor search
- Embedding normalization
- Local vector storage (start simple, scale later)

**Translation Example**: Generate embeddings for translation examples, store them, and retrieve similar examples for few-shot learning. Find similar phrases to reuse translations.

**Example Implementation**:
- Embed translation examples and retrieve similar ones for few-shot prompts
- Semantic search for previously translated phrases
- Similarity-based example selection for dynamic few-shot learning

**Example**: [embeddings_vector_search.py](../src/embeddings_vector_search.py)

---

## Pattern 8: RAG (Retrieval-Augmented Generation)

**Description**: Augment LLM inputs with relevant context retrieved from external knowledge bases or vector databases. Combine semantic search with generative capabilities.

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

## Pattern 9: Streaming / Progressive Generation

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

## Pattern 10: Caching / Optimization

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

## Pattern 11: Memory / State Management

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

## Pattern 12: Guardrails / Safety

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

## Pattern 13: Agentic Systems

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

## Pattern 14: Orchestration / Workflow Management

**Description**: Coordinate multi-step LLM workflows with proper error handling, state management, and retry logic. Ensure reliable execution of complex processes.

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

## Pattern 15: LLM-as-a-Judge (Evaluation)

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

## Pattern 16: Advanced Patterns

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

### Core Patterns (1-4)
- **OpenAI SDK**: Primary LLM interface
- **Pydantic**: Data validation and structured outputs

### Intermediate Patterns (5-8)
- **OpenAI SDK**: Embeddings API
- **Local Vector Storage**: SQLite with vector extension or FAISS
- **Simple caching**: SQLite or in-memory

### Advanced Patterns (9-12)
- **OpenAI SDK**: Streaming, moderation
- **Local storage**: SQLite for state/memory
- **Custom orchestration**: Python-based workflow engine

### Production Patterns (13-16)
- **OpenAI SDK**: All features
- **External services**: As needed (Redis, PostgreSQL, etc.)
- **Custom frameworks**: Built on previous patterns

---

## Self-Learning Path

This progression is designed for self-paced learning. Each pattern:
- Builds on previous concepts
- Can be applied to your chosen use case
- Can be implemented incrementally
- Demonstrates real-world refinement

Choose a use case that interests you and apply each pattern to refine your system. The translation examples serve as a reference for how patterns can be combined.

### Suggested Learning Order

**Foundation** (Patterns 1-4):
1. Prompt Engineering
2. Structured Output
3. Function Calling / Tool Use
4. Schema-Driven Inference

**Intermediate** (Patterns 5-8):
5. Few-Shot Learning
6. Chain-of-Thought Reasoning
7. Embeddings / Vector Search
8. RAG Basics

**Advanced** (Patterns 9-12):
9. Streaming
10. Caching / Optimization
11. Memory / State Management
12. Guardrails / Safety

**Production** (Patterns 13-16):
13. Agentic Systems
14. Orchestration / Workflows
15. LLM-as-a-Judge
16. Advanced Patterns

---

## Notes

- **Choose your own use case**: The translation examples are just one approach. Select a use case that interests you and apply these patterns to refine your system.
- **Refinement through examples**: Using a consistent use case shows how each pattern adds sophistication and builds on previous concepts.
- **Start simple**: Each pattern builds on previous concepts.
- **Best practices**: Use industry-standard, accessible technologies.
- **Incremental complexity**: Add services only when needed.
- **Examples build on each other**: Later examples incorporate earlier patterns.
- **Production-ready**: Patterns scale from prototype to production.

---

*Last Updated: January 2026*
