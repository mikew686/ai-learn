# AI Technology Engineering Patterns

This is an AI-generated summary of common AI engineering technologies. It covers application-level patterns and techniques used with established models and services, not model training or fine-tuning.

## Overview

**Core principles**
- Understanding Models
- Prompts
- Schemas
- Tools
- Schema-Driven Inference

**Technology concepts**
- Embeddings
- Few-Shot
- Caching
- Memory
- Streaming
- Guardrails

**Bring it together**
- RAG
- Chain-of-Thought
- Tree-of-Thought
- Agents
- Multi-Agent
- Orchestration
- Evaluation and advanced patterns

---

## Understanding Models

Model capabilities vary by class; OpenRouter and similar services expose models that fall into these general categories:

**Chat/Conversational Models**: General-purpose models for dialogue and text generation, with function calling, structured outputs, and streaming. Commonly used for application patterns.
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Google**: Gemini Pro, Gemini Ultra
- **Other Vendors**: Meta Llama 3, Mistral Large, Cohere Command, DeepSeek Chat, xAI Grok

**Reasoning Models**: Models built for explicit step-by-step reasoning, suited to complex problem-solving, multi-step analysis, and logical deduction.
- **OpenAI**: o1-preview, o1-mini, o3-mini
- **Anthropic**: Claude 3.5 Sonnet (with reasoning capabilities)
- **Google**: Gemini Pro (with chain-of-thought prompting)
- **Other Vendors**: DeepSeek R1, Meta Llama 3 (with CoT prompting)

**Fast/Cheap Models**: Lightweight models tuned for speed and cost, used for simple tasks, high-volume workloads, and latency-sensitive use cases.
- **OpenAI**: GPT-3.5-turbo, GPT-4o-mini
- **Anthropic**: Claude 3 Haiku
- **Google**: Gemini Flash
- **Other Vendors**: Meta Llama 3.1 8B, Mistral Small, Cohere Command Light, DeepSeek Chat 7B

**Embedding Models**: Specialized for generating vector representations of text. Used for semantic search, RAG, and similarity matching.
- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Anthropic**: (Embeddings via third-party services)
- **Google**: text-embedding-004, textembedding-gecko
- **Other Vendors**: Cohere Embed, Mistral Embed, Voyage AI, Nomic Embed

**Code Models**: Tuned for code generation and understanding, including completion, debugging, and explanation.
- **OpenAI**: GPT-4 (code capabilities), GPT-3.5-turbo (code)
- **Anthropic**: Claude 3.5 Sonnet (strong code capabilities)
- **Google**: Gemini Pro (code generation)
- **Other Vendors**: Meta CodeLlama, DeepSeek Coder, Mistral Code, StarCoder

**Multimodal Models**: Support both text and image inputs/outputs. Enable image analysis, visual question answering, and image generation workflows.
- **OpenAI**: GPT-4 Vision, GPT-4o (multimodal)
- **Anthropic**: Claude 3.5 Sonnet (vision), Claude 3 Opus (vision)
- **Google**: Gemini Pro Vision, Gemini Ultra (multimodal)
- **Other Vendors**: Meta Llama 3.1 (vision), Mistral Large (multimodal), Cohere Command R+ (multimodal)

📖 [Detailed Documentation](./understanding_models.md) — Model capabilities, selection criteria, and use case mapping.

📖 [LLM vs. Reasoning Problems](./llm_vs_reasoning_problems.md) — Problem types suited to chat models vs. reasoning models, with examples.

**Related Patterns**:
- **Prompt Engineering**: Model choice influences prompt design and which techniques apply
- **Embeddings / Vector Search**: Embedding models produce vectors for semantic search and RAG
- **RAG**: Often uses embedding models for document and query encoding
- **Function Calling / Tool Use**: Chat models with tool support enable agentic and tool-augmented flows

---

## Prompt Engineering

📖 [Detailed Documentation](./prompt_engineering.md)

**Description**: Designing prompts to steer LLM behavior and shape outputs. Involves structuring instructions, adding context, and applying techniques that improve response quality.

**Key Techniques**:
- Template-based prompts with variable substitution
- Few-shot learning (providing examples in prompts)
- Role-setting ("You are an expert at...")
- Structured instructions with numbered steps
- Constraint injection (business rules in prompts)
- Output format specification

**Popular Solutions**:
- **LangChain Prompt Templates**: Template system with variable substitution
- **Jinja2 Templates**: Flexible templating for complex prompts
- **PromptLayer**: Prompt versioning and management
- **Weights & Biases Prompts**: Prompt tracking and optimization

**Common practices**:
- Clear, specific instructions
- Examples included for complex tasks
- System prompts separated from user prompts
- Prompt versioning
- Systematic prompt testing

**Use Cases**:
- Content generation
- Data extraction
- Code generation
- Question answering
- Translation

**Related Patterns**:
- **Few-Shot / In-Context Learning**: Examples in prompts complement prompt engineering
- **Structured Output**: Prompts often specify or constrain output format
- **Schema-Driven Inference**: Minimal prompts work when schemas carry implicit instructions
- **RAG**: Retrieved context is injected into prompts

---

## Function Calling / Tool Use

📖 [Detailed Documentation](./function_calling_tool_use.md)

**Description**: Enabling LLMs to call external functions, APIs, or tools through structured interfaces. Allows models to interact with the real world beyond text generation.

**Key Concepts**:
- Function definitions (JSON Schema)
- Tool selection by LLM
- Parameter extraction
- Function execution
- Response integration

**Popular Solutions**:
- **OpenAI Function Calling** / **Anthropic Tool Use**: Native support in GPT and Claude
- **LangChain Tools**: Tool abstraction framework
- **AWS Bedrock Agents**: Managed agents with native tool calling
- **CrewAI**: Multi-agent framework with tools

**Common practices**:
- Clear, specific function schemas
- Input validation before execution
- Graceful error handling
- Descriptive tool descriptions
- Retry logic

**Use Cases**:
- API integrations
- Database queries
- File operations
- Web scraping
- Calculator tools
- Code execution

**Related Patterns**:
- **Structured Output**: Tool parameters and responses often use structured schemas
- **Schema-Driven Inference**: Tool schemas act as implicit prompts for the model
- **Agentic Systems**: Agents rely on tools for planning and execution
- **Orchestration**: Workflows coordinate tool calls across steps

---

## Structured Output

📖 [Detailed Documentation](./structured_output.md)

**Description**: Enforcing structured, validated responses from LLMs using schemas. Ensures outputs conform to expected formats and can be reliably parsed.

**Key Approaches**:
- JSON Schema validation
- Pydantic model parsing
- XML output formats
- TypeScript type definitions

**Popular Solutions**:
- **Pydantic**: Python data validation (used with OpenAI `.parse()`)
- **Structured Outputs (OpenAI)**: Native structured output support
- **JSON Schema**: Standard schema format
- **Zod**: TypeScript schema validation

**Common practices**:
- Strict schemas
- Graceful parsing error handling
- Fallback mechanisms
- Output validation
- Type-safe languages where applicable

**Use Cases**:
- Data extraction
- API response generation
- Database record creation
- Configuration generation
- Form filling

**Related Patterns**:
- **Function Calling / Tool Use**: Tool definitions use JSON Schema; outputs can be structured
- **Schema-Driven Inference**: Pydantic and schemas reduce need for verbose prompts
- **Prompt Engineering**: Schemas complement explicit instructions

---

## Schema-Driven Inference

📖 [Detailed Documentation](./schema_driven_inference.md)

**Description**: Using structured definitions (tool schemas, Pydantic field descriptions, JSON Schema) as implicit prompts that allow the model to infer behavior and requirements without verbose explicit instructions. Reduces prompt verbosity while maintaining high-quality, validated outputs.

**Key Concepts**:
- Tool schema descriptions guide model behavior
- Pydantic field descriptions specify data requirements
- Structured definitions serve as implicit prompts
- Model infers details from schema metadata
- Reduced prompt complexity with maintained quality

**Common practices**:
- Clear, descriptive tool function descriptions
- Detailed Pydantic Field descriptions
- Schemas used to communicate requirements implicitly
- Minimal prompts when schemas supply context
- Explicit prompts combined only when needed

**Use Cases**:
- Translation with language detection and cultural context
- Data extraction with structured validation
- API integrations with tool calling
- Multi-step workflows combining tools and structured output
- Reducing prompt token usage while maintaining quality

**Related Patterns**:
- **Function Calling / Tool Use**: Tool schemas provide implicit instructions
- **Structured Output**: Pydantic field descriptions guide output generation
- **Prompt Engineering**: Complements explicit prompting techniques

---

## Embeddings / Vector Search

📖 [Detailed Documentation](./embeddings_and_vector_search.md)

**Description**: Using dense vector representations (embeddings) and similarity search to find relevant content. This pattern is the foundation for semantic search, RAG, and dynamic few-shot selection. It covers embedding generation, vector storage, similarity metrics (e.g. cosine similarity), and combining them—for example, storing examples and retrieving the most similar ones to build few-shot prompts.

**Key Concepts**:
- **Embedding generation**: Call an embeddings API to convert text into fixed-length vectors that capture semantic meaning
- **Vector similarity**: Cosine similarity or dot product (often with normalized vectors) to rank by relevance
- **Vector storage**: From simple (SQLite BLOB, in-memory) to scaled (FAISS, vector DBs) for persistence and throughput
- **Dynamic few-shot**: Use the current input’s embedding to select the most relevant stored examples and inject them into the prompt
- **Exact-prompt embedding**: Embed the same text that will appear in the user message so query and stored items align
- **Target normalization (vector-backed)**: Embed free-form user input (e.g. “French Quebec”), search a small target store; if nearest row is within a distance threshold, reuse it and skip the LLM

**Popular Solutions**:
- **OpenAI Embeddings** / **Cohere Embed**: Widely used embedding APIs
- **Chroma**: Lightweight embedded vector store
- **Postgres + pgvector**: Vector search in PostgreSQL
- **Pinecone**: Managed vector database
- **LangChain Vector Stores** / **LlamaIndex**: Abstraction over multiple backends

**Common practices**:
- Normalized vectors for cosine similarity (or APIs that return normalized vectors)
- Batched embedding calls; embedding dimension stored when switching models is likely
- Metadata weighting (e.g. language/dialect) so retrieval respects context
- Simple storage first; FAISS or a vector DB as scale or latency demands
- Exact-prompt embedding when retrieving few-shot examples for chat flows

**Use Cases**:
- Semantic search over owned content (documents, translations, support answers)
- Dynamic few-shot or example selection for prompts
- RAG: embed documents and query, retrieve, then generate with context
- Clustering, deduplication, or recommendation when items have a text component
- **Semantic caching**: Same machinery (embed query, similarity search)—reuse cached LLM responses for similar queries; see **Caching / Optimization**

**Related Patterns**:
- **RAG**: Uses embeddings and vector search to retrieve context before generation
- **Few-Shot / In-Context Learning**: Vector search often selects which few-shot examples to include
- **Prompt Engineering**: Retrieved content is injected into prompts
- **Understanding Models – Embedding Models**: Technical details on embedding APIs and similarity

---

## Few-Shot / In-Context Learning

**Description**: Providing examples in prompts to guide LLM behavior without modifying the model. Leverages the model's ability to learn patterns from examples.

**Key Techniques**:
- Example selection
- Example ordering
- Example formatting
- Dynamic example selection
- Example diversity

**Popular Solutions**:
- **LangChain FewShotPromptTemplate**: Template for few-shot prompts
- **Example Selectors**: Dynamic example selection
- **Semantic Similarity Example Selection**: Selects examples by similarity
- **Custom Example Stores**: Store and retrieve examples

**Common practices**:
- Diverse, representative examples
- Strategic example ordering
- Example format matched to desired output
- Typically 2-5 examples
- Examples updated based on performance

**Use Cases**:
- Style transfer
- Format learning
- Task-specific behavior
- Domain adaptation
- Custom output formats

**Related Patterns**:
- **Embeddings / Vector Search**: Dynamic few-shot uses vector similarity to select examples
- **Prompt Engineering**: Few-shot examples are part of prompt design
- **RAG**: Retrieved passages act as in-context knowledge; similar to few-shot retrieval
- **Schema-Driven Inference**: Examples can illustrate schema expectations

---

## Caching / Optimization

**Description**: Caching LLM responses, embeddings, or intermediate results to reduce latency, costs, and improve performance. **Semantic caching** (cache hits for semantically similar queries) uses **embeddings and vector search**: embed the query, find nearest cached query, return cached response—same pattern as in Embeddings / Vector Search.

**Key Strategies**:
- Response caching (exact key, e.g. Redis)
- Embedding caching (store embedding API outputs to avoid re-calls)
- Prompt caching (provider-side repeated prefix)
- **Semantic caching**: Embeddings + vector similarity to reuse responses for similar queries
- Result memoization

**Popular Solutions**:
- **Redis** (e.g. ElastiCache): Response and session caching
- **GPTCache**: Semantic cache for LLM responses
- **LangChain Cache**: Multiple caching backends
- **DynamoDB**: Caching with TTL
- **SQLite**: Lightweight local cache

**Common practices**:
- Caching of expensive operations
- Semantic caching for similar queries
- Cache invalidation strategies
- Cache hit rate monitoring
- Tradeoff between freshness and performance

**Use Cases**:
- Cost reduction
- Latency improvement
- Rate limit management
- Repeated queries
- Batch processing

**Semantic caching (detail)**: Query is embedded and compared to cached (query_embedding, response) pairs via vector similarity; cached response returned if within threshold. Involves similarity thresholds, high-quality embeddings, and cache warming. Solutions: **GPTCache**, **LangChain Semantic Cache**, custom vector cache, **Redis with Embeddings**. Applied when cache hits for *similar* queries are desired, not only exact repeats.

**Related Patterns**:
- **Embeddings / Vector Search**: Same machinery (embed + similarity search) for semantic cache keys and lookup
- **Orchestration**: Cache workflow or step results for idempotency
- **Streaming**: Cache streamed responses or segments when appropriate
- **RAG**: Semantic caches often store RAG-style query/response pairs

---

## Memory / State Management

**Description**: Persisting context, conversation history, and state across LLM interactions. Enables long-term memory and context retention.

**Key Types**:
- Short-term memory (conversation history)
- Long-term memory (persistent storage)
- Episodic memory (specific events)
- Semantic memory (facts and knowledge)

**Popular Solutions**:
- **LangChain Memory** / **LlamaIndex Memory**: Multiple memory types and persistence
- **Redis** (e.g. ElastiCache): Session and short-term memory
- **Vector Store Memory**: Semantic/long-term memory
- **DynamoDB** / **PostgreSQL**: Durable state storage
- **Zep**: Long-term memory for chat

**Common practices**:
- Separation of short-term and long-term memory
- Memory summarization
- Vector search for semantic memory
- Memory pruning
- Privacy and data retention policies

**Use Cases**:
- Conversational AI
- Personal assistants
- Customer relationship management
- Learning systems
- Context-aware applications

**Related Patterns**:
- **Embeddings / Vector Search**: Semantic memory often uses vector stores
- **RAG**: Long-term knowledge can be stored and retrieved via RAG
- **Agentic Systems**: Agents need memory for context across turns and tasks
- **Orchestration**: Workflows persist state across steps

---

## Streaming / Progressive Generation

**Description**: Streaming LLM responses token-by-token for real-time user feedback. Improves perceived latency and user experience.

**Key Features**:
- Token-by-token streaming
- Server-Sent Events (SSE)
- WebSocket support
- Partial response handling
- Cancellation support

**Popular Solutions**:
- **OpenAI Streaming API**: Native streaming support
- **Anthropic Streaming**: Claude streaming API
- **LangChain Streaming**: Streaming callbacks
- **FastAPI Streaming**: Server-side streaming
- **Server-Sent Events**: HTTP-based streaming

**Common practices**:
- Error handling and cancellation support
- Graceful handling of partial responses
- Loading indicators
- Low-latency optimization

**Use Cases**:
- Real-time chatbots
- Live content generation
- Interactive applications
- Code generation
- Translation services

**Related Patterns**:
- **Prompt Engineering**: Streaming delivers prompt-driven generation incrementally
- **Caching**: Cache streamed responses or segments when appropriate
- **Orchestration**: Workflows can stream steps or final output

---

## Guardrails / Safety

**Description**: Content filtering, output validation, and safety checks to prevent harmful, biased, or inappropriate outputs. Critical for production deployments.

**Key Areas**:
- Content moderation
- Toxicity detection
- PII detection and redaction
- Output validation
- Rate limiting

**Popular Solutions**:
- **OpenAI Moderation API**: Content moderation
- **AWS Bedrock Guardrails**: Built-in content filtering
- **NVIDIA NeMo Guardrails**: Open-source guardrails framework
- **Presidio**: PII detection and anonymization
- **LangChain Guardrails**: Safety integrations

**Common practices**:
- Multiple layers of safety
- User input validation
- Continuous output monitoring
- Human review for sensitive content
- Updated safety rules

**Use Cases**:
- Public-facing applications
- Educational platforms
- Healthcare applications
- Financial services
- Content moderation

**Related Patterns**:
- **Structured Output**: Validate outputs against schemas as a guardrail
- **LLM-as-a-Judge**: Evaluate content for safety and alignment
- **Prompt Engineering**: System prompts can encode safety and behavior constraints

---

## RAG (Retrieval-Augmented Generation)

**Description**: Augmenting LLM inputs with relevant context retrieved from external knowledge bases or vector databases. Combines the power of semantic search with generative capabilities.

**Key Components**:
- Vector embeddings of documents
- Vector database for storage and retrieval
- Retrieval mechanism (semantic search)
- Context injection into prompts

**Popular Solutions**:
- **LangChain Vector Stores** / **LlamaIndex**: Data frameworks and abstraction over vector backends
- **AWS Bedrock Knowledge Bases**: Managed RAG with vector search
- **Pinecone**: Managed vector database
- **Chroma**: Embedded vector database
- **Weaviate** / **Qdrant**: Open-source vector databases

**Common practices**:
- Document chunking (e.g. 200-500 tokens)
- Metadata filtering for precision
- Hybrid search (keyword + semantic)
- Re-ranking for relevance
- Retrieval quality monitoring

**Use Cases**:
- Document Q&A systems
- Knowledge base assistants
- Research tools
- Customer support chatbots
- Legal document analysis

**Related Patterns**:
- **Embeddings / Vector Search**: Foundation for encoding documents and queries; vector DBs store and retrieve context
- **Few-Shot / In-Context Learning**: Retrieved chunks are often used as in-context examples
- **Prompt Engineering**: Retrieved context is injected into prompts
- **Hybrid Search**: Combines keyword and semantic search for better RAG retrieval

---

## Chain-of-Thought / Multi-Step Reasoning

**Description**: Encouraging LLMs to show explicit step-by-step reasoning processes. Improves accuracy on complex problems requiring logical reasoning.

**Key Techniques**:
- Explicit reasoning prompts ("Let's think step by step")
- Intermediate reasoning steps
- Self-verification
- Decomposition of complex problems

**Popular Solutions**:
- **Chain-of-Thought Prompting**: "Let's think step by step" and variants
- **ReAct**: Reasoning + tool use
- **LangChain Chains**: Multi-step reasoning workflows
- **Tree of Thoughts**: Search-based reasoning

**Common practices**:
- Decomposition of complex problems into steps
- Explicit reasoning in outputs
- Verification of intermediate results
- Multiple reasoning paths for critical decisions
- Reasoning quality monitoring

**Use Cases**:
- Mathematical problem solving
- Logical reasoning tasks
- Multi-step planning
- Code debugging
- Scientific analysis

**Related Patterns**:
- **Function Calling / Tool Use**: ReAct and similar approaches combine reasoning with tool use
- **Tree-of-Thought / Search-Based Reasoning**: Explores multiple reasoning paths; extends CoT with search
- **Agentic Systems**: Agents use reasoning for planning and reflection
- **LLM-as-a-Judge**: Evaluate reasoning quality and step correctness

---

## Tree-of-Thought / Search-Based Reasoning

**Description**: Exploring multiple reasoning paths simultaneously and selecting the best solution. Uses search algorithms to find optimal reasoning chains.

**Key Concepts**:
- Multiple reasoning branches
- Search algorithms (BFS, DFS, beam search)
- Evaluation of reasoning paths
- Best path selection

**Popular Solutions**:
- **Tree of Thoughts**: Original research implementation
- **LangGraph**: State machine with branching
- **Custom Search Algorithms**: Implement search logic
- **Beam Search**: Explores top-k paths

**Common practices**:
- Bounded search space for performance
- Heuristics to guide search
- Efficient path evaluation
- Pruning strategies
- Balance of exploration and exploitation

**Use Cases**:
- Complex problem solving
- Multi-step planning
- Code generation
- Mathematical proofs
- Strategic decision making

**Related Patterns**:
- **Chain-of-Thought**: Tree-of-Thought extends CoT with multiple branches and search
- **Agentic Systems**: Search over plans or actions fits agentic loops
- **LLM-as-a-Judge**: Evaluate and score reasoning paths
- **Orchestration**: Coordinate search steps and backtracking

---

## Agentic Systems

**Description**: Autonomous systems that can plan, execute actions, and adapt based on results. Agents use tools, memory, and reasoning to achieve goals.

**Key Components**:
- Planning capabilities
- Tool/action execution
- Memory/state management
- Goal tracking
- Reflection and adaptation

**Popular Solutions**:
- **LangGraph**: State machine for agent workflows
- **AWS Bedrock Agents** / **AWS Strands Agents**: Managed and open-source agent SDKs ([Strands](https://github.com/aws/strands-agents))
- **CrewAI**: Multi-agent orchestration
- **AutoGen**: Multi-agent conversation framework
- **LlamaIndex Agents**: Agent framework with RAG

**Common practices**:
- Clear goals and constraints
- Safety limits (max steps, timeouts)
- Agent behavior monitoring
- Structured memory
- Human-in-the-loop for critical actions
- Reliance on model-native capabilities (reasoning, tool use) over complex orchestration
- Semantic search for tool selection with large tool sets (1000+ tools)

**Use Cases**:
- Research assistants
- Task automation
- Code generation and testing
- Data analysis workflows
- Customer service automation

**Related Patterns**:
- **Function Calling / Tool Use**: Agents invoke tools; tool schemas define actions
- **Orchestration**: Workflows coordinate agent steps and handle failures
- **Memory / State Management**: Agents need context and state across steps
- **RAG**: Agents often use retrieval for knowledge-grounded actions
- **Multi-Agent Systems**: Multiple agents collaborate within or across workflows

---

## Multi-Agent Systems

**Description**: Systems with multiple specialized agents that collaborate, communicate, and coordinate to solve complex tasks. Each agent has specific roles and capabilities.

**Key Concepts**:
- Agent specialization
- Inter-agent communication
- Coordination mechanisms
- Shared memory/state
- Conflict resolution

**Popular Solutions**:
- **CrewAI**: Multi-agent orchestration framework
- **AutoGen**: Multi-agent conversation framework
- **LangGraph Multi-Agent**: Multi-agent state machines
- **Semantic Kernel**: Multi-agent orchestration
- **Swarm**: Decentralized agent networks

**Common practices**:
- Clear agent roles
- Communication protocols
- Conflict resolution
- Agent interaction monitoring
- Scalability-oriented design

**Use Cases**:
- Complex research tasks
- Software development teams
- Content creation workflows
- Data analysis pipelines
- Customer service teams

**Related Patterns**:
- **Agentic Systems**: Multi-agent systems extend single-agent patterns with coordination
- **Orchestration**: Coordinates agents and workflows
- **Memory / State Management**: Shared or per-agent memory for context
- **Function Calling / Tool Use**: Agents use tools; inter-agent communication can be tool-based

---

## Orchestration / Workflow Management

**Description**: Coordinating multi-step LLM workflows with proper error handling, state management, and retry logic. Ensures reliable execution of complex processes.

**Key Features**:
- Sequential and parallel execution
- Error handling and recovery
- State persistence
- Conditional branching
- Retry mechanisms

**Popular Solutions**:
- **LangChain**: Orchestration and chains
- **AWS Step Functions**: Serverless workflow with LLM integration
- **Temporal**: Durable execution engine
- **Prefect** / **Airflow**: Workflow orchestration

**Common practices**:
- Idempotent workflows
- Comprehensive error handling
- State checkpoints
- Workflow execution monitoring
- Failure recovery design

**Use Cases**:
- Multi-step content generation
- Data processing pipelines
- ETL with LLM steps
- Approval workflows
- Batch processing

**Related Patterns**:
- **Agentic Systems**: Orchestration runs agent workflows and step functions
- **Function Calling / Tool Use**: Workflow steps often call tools or APIs
- **Structured Output**: Evaluation and routing benefit from structured judge outputs
- **Caching**: Cache step results for idempotency and cost reduction

---

## LLM-as-a-Judge (Evaluation)

**Description**: Using LLMs to evaluate the quality, correctness, or alignment of outputs. Enables automated quality assessment at scale.

**Key Approaches**:
- Rubric-based evaluation
- Comparative evaluation
- Fact-checking
- Alignment checking
- Quality scoring

**Popular Solutions**:
- **LangSmith**: LLM observability and evaluation
- **TruLens**: LLM evaluation framework
- **RAGAS**: RAG-specific evaluation metrics
- **LangChain Evaluators**: Built-in evaluation tools
- **Weights & Biases**: Experiment tracking

**Common practices**:
- Separate judge models for objectivity
- Clear evaluation criteria
- Structured evaluation outputs
- Multiple evaluation dimensions
- Judge consistency validation

**Use Cases**:
- Output quality assessment
- Hallucination detection
- Alignment verification
- A/B testing evaluation
- Automated testing

**Related Patterns**:
- **Structured Output**: Use schemas for rubric-based or comparative evaluation
- **Prompt Engineering**: Judge prompts define criteria and rubrics
- **RAG**: RAGAS and similar metrics evaluate retrieval-augmented systems
- **Chain-of-Thought**: Evaluate reasoning steps and correctness

---

## Hybrid Search

**Description**: Combining keyword (BM25) and semantic (vector) search for improved retrieval accuracy. Leverages strengths of both approaches.

**Key Components**:
- Keyword search (BM25, TF-IDF)
- Semantic search (vector embeddings)
- Result fusion/reranking
- Weighted combination

**Popular Solutions**:
- **Elasticsearch** / **OpenSearch**: BM25 + vector search
- **Weaviate** / **Pinecone**: Native hybrid search
- **Qdrant**: Hybrid search with metadata filters
- **AWS Bedrock Knowledge Bases**: Managed hybrid RAG

**Common practices**:
- Tuned keyword/semantic weights
- Appropriate reranking
- Effective result combination
- Search quality monitoring
- A/B-tested configurations

**Use Cases**:
- Document search
- E-commerce search
- Knowledge base search
- Content discovery
- RAG systems

**Related Patterns**:
- **Embeddings / Vector Search**: Vector search is the semantic side of hybrid
- **RAG**: Hybrid search improves retrieval for RAG pipelines
- **Caching / Optimization**: Semantic caching uses embeddings; hybrid can support cache matching

---

## Prompt Compression / Optimization

**Description**: Reducing prompt size while preserving essential information. Improves efficiency, reduces costs, and enables longer context windows.

**Key Techniques**:
- Summarization
- Key information extraction
- Template optimization
- Context pruning
- Token reduction

**Popular Solutions**:
- **LLMLingua**: Prompt compression library
- **LongLLMLingua**: Long context compression
- **Custom Summarization**: Summarize context
- **Selective Context**: Selects relevant parts

**Common practices**:
- Preservation of critical information
- Testing of compressed prompts
- Quality monitoring after compression
- Selective use of compression
- Balance of size and quality

**Use Cases**:
- Long context management
- Cost reduction
- Token limit optimization
- Performance improvement
- Batch processing

**Related Patterns**:
- **Prompt Engineering**: Compression preserves essential prompt content
- **RAG**: Compress retrieved context before injection
- **Caching**: Compressed prompts can reduce cache key size or enable longer cached context

---

## Model Routing / Ensemble

**Description**: Selecting or combining multiple models based on task characteristics, cost, or performance requirements. Optimizes for different use cases.

**Key Strategies**:
- Task-based routing
- Cost-based routing
- Performance-based routing
- Model ensembling
- Fallback chains

**Popular Solutions**:
- **OpenRouter**: Multi-model routing and fallbacks
- **AWS Bedrock**: Multiple foundation models and model selection
- **LangChain Router Chains**: Route by task or metadata
- **Custom routing** (e.g. Lambda): Task-based or cost-based routing
- **Fallback chains**: Automatic model switching on errors

**Common practices**:
- Clear routing criteria
- Routing decision monitoring
- Fallback mechanisms
- Testing of all routes
- Cost and performance optimization

**Use Cases**:
- Cost optimization
- Performance optimization
- Specialized task handling
- Reliability improvement
- Multi-model systems

**Related Patterns**:
- **Understanding Models**: Routing chooses among model types and capabilities
- **LLM-as-a-Judge**: Judge model can be routed separately from production model
- **Custom Model Training**: Fine-tuned models are often used in routing or fallback chains
- **Structured Output**: Routing criteria can use structured outputs from classifiers

---

## Custom Model Training

**Description**: Training or fine-tuning models on custom datasets to achieve domain-specific performance, specialized behavior, or improved accuracy for specific use cases. Involves adapting pre-trained foundation models to particular tasks or domains.

**Key Approaches**:
- Fine-tuning: Adapting pre-trained models with task-specific data
- Full training: Training models from scratch (rare, resource-intensive)
- Parameter-efficient fine-tuning (PEFT): LoRA, QLoRA, adapter methods
- Continual learning: Updating models with new data over time
- Domain adaptation: Specializing models for specific domains

**Popular Solutions**:
- **OpenAI Fine-tuning API** / **Anthropic Fine-tuning**: Vendor fine-tuning for GPT and Claude
- **Hugging Face** (Transformers, PEFT): Open-source training and LoRA/QLoRA
- **AWS Bedrock Custom Models** / **SageMaker**: Fine-tune on AWS
- **LoRA / QLoRA**: Parameter-efficient fine-tuning
- **Weights & Biases** / **MLflow**: Experiment tracking

**Common practices**:
- High-quality, representative training data
- Parameter-efficient methods (e.g. LoRA) to reduce costs
- Data quality and diversity validation before training
- Train/validation/test splits
- Monitoring for overfitting and degradation
- Versioned datasets and model checkpoints
- Evaluation on held-out test sets
- Domain-specific evaluation metrics
- Data augmentation where appropriate
- Transfer learning from strong foundation models

**Use Cases**:
- Domain-specific language models (legal, medical, technical)
- Custom style or tone adaptation
- Task-specific optimization (classification, extraction, generation)
- Multilingual model adaptation
- Specialized code generation models
- Custom chatbot personalities
- Industry-specific assistants
- Brand voice alignment

**Related Patterns**:
- **Understanding Models**: Fine-tuned models extend or specialize base model capabilities
- **Model Routing / Ensemble**: Custom models are often used in routing or ensemble setups
- **Structured Output**: Training data and evaluation use structured formats
- **LLM-as-a-Judge**: Evaluate fine-tuned model outputs and drift

---

## Implementation Considerations

### Technique selection

Common progression: prompt engineering and structured outputs first; RAG, function calling, and orchestration added as needed. Metrics are tracked per technique, functionality is balanced with API cost, and systems are designed for scale.

### Common pattern combinations

- **RAG + Function Calling**: Knowledge retrieval with tool use
- **Orchestration + Caching**: Workflow coordination with performance optimization
- **Prompt Engineering + Structured Output**: Reliable, validated outputs
- **Function Calling + Structured Output + Schema-Driven Inference**: Tools gather data, schemas guide output, minimal prompts
- **Multi-Agent + Memory**: Collaborative systems with context retention

### Cross-cutting concerns

- **Observability**: LLM interaction monitoring
- **Error handling**: Comprehensive error handling
- **Testing**: Per-component testing
- **Documentation**: Implementation documentation
- **Version control**: Prompts, schemas, and configurations
- **Security**: Authentication and authorization
- **Cost management**: Cost monitoring and optimization

---

## Resources

### Framework Documentation
- **LangChain Documentation**: https://python.langchain.com/
- **LlamaIndex Documentation**: https://docs.llamaindex.ai/
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
- **Semantic Kernel**: https://learn.microsoft.com/en-us/semantic-kernel/
- **CrewAI Documentation**: https://docs.crewai.com/

### Provider Documentation
- **OpenAI API Documentation**: https://platform.openai.com/docs/
- **OpenAI Best Practices**: https://platform.openai.com/docs/guides/prompt-engineering
- **OpenAI Function Calling Guide**: https://platform.openai.com/docs/guides/function-calling
- **Anthropic Documentation**: https://docs.anthropic.com/
- **OpenRouter Documentation**: https://openrouter.ai/docs
- **AWS Bedrock Documentation**: https://docs.aws.amazon.com/bedrock/
- **AWS Bedrock Agents**: https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html
- **Amazon SageMaker**: https://docs.aws.amazon.com/sagemaker/

### Vector Databases & Embeddings
- **Vector Database Comparison**: https://www.pinecone.io/learn/vector-database/
- **Pinecone Documentation**: https://docs.pinecone.io/
- **Weaviate Documentation**: https://weaviate.io/developers/weaviate
- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **Milvus Documentation**: https://milvus.io/docs

### Model Training & Fine-tuning
- **Hugging Face PEFT**: https://huggingface.co/docs/peft/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **Weights & Biases**: https://wandb.ai/

### Evaluation & Observability
- **LangSmith**: https://docs.smith.langchain.com/
- **TruLens**: https://www.trulens.org/
- **RAGAS**: https://docs.ragas.io/

### Best Practices & Guides
- **OpenAI Cookbook**: https://cookbook.openai.com/
- **Anthropic Prompt Engineering**: https://docs.anthropic.com/claude/docs/prompt-engineering
- **AWS Well-Architected ML**: https://aws.amazon.com/architecture/well-architected/machine-learning/
- **LLM Best Practices**: https://platform.openai.com/docs/guides/production-best-practices

### Research & Papers
- **Retrieval-Augmented Generation (RAG)**: https://arxiv.org/abs/2005.11401
- **ReAct: Synergizing Reasoning and Acting**: https://arxiv.org/abs/2210.03629
- **Chain-of-Thought Prompting**: https://arxiv.org/abs/2201.11903
- **Tree of Thoughts**: https://arxiv.org/abs/2305.10601

---

*Last Updated: 2025*
