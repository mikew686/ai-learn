# AI Technology Engineering Patterns

This document outlines key AI engineering patterns and techniques for building production systems using established models and services. These patterns focus on application-level engineering rather than model training or fine-tuning.

## Learning Progression

📚 [Learning Progression Guide](./learning_progression.md) - A self-learning progression that demonstrates how these patterns build on each other through a consistent use case. Start with foundational patterns and progressively add complexity as you learn.

---

## Understanding Models

Understanding model capabilities helps you select the right model for your use case. Models available through OpenRouter and similar services can be categorized into general classes:

**Chat/Conversational Models**: General-purpose models optimized for dialogue and text generation. Support function calling, structured outputs, and streaming. Best for most application patterns.
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Google**: Gemini Pro, Gemini Ultra
- **Other Vendors**: Meta Llama 3, Mistral Large, Cohere Command, DeepSeek Chat, xAI Grok

**Reasoning Models**: Models designed for explicit step-by-step reasoning. Better for complex problem-solving, multi-step analysis, and tasks requiring logical deduction.
- **OpenAI**: o1-preview, o1-mini, o3-mini
- **Anthropic**: Claude 3.5 Sonnet (with reasoning capabilities)
- **Google**: Gemini Pro (with chain-of-thought prompting)
- **Other Vendors**: DeepSeek R1, Meta Llama 3 (with CoT prompting)

**Fast/Cheap Models**: Lightweight models optimized for speed and cost. Suitable for simple tasks, high-volume operations, and when latency is critical.
- **OpenAI**: GPT-3.5-turbo, GPT-4o-mini
- **Anthropic**: Claude 3 Haiku
- **Google**: Gemini Flash
- **Other Vendors**: Meta Llama 3.1 8B, Mistral Small, Cohere Command Light, DeepSeek Chat 7B

**Embedding Models**: Specialized for generating vector representations of text. Used for semantic search, RAG, and similarity matching.
- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Anthropic**: (Embeddings via third-party services)
- **Google**: text-embedding-004, textembedding-gecko
- **Other Vendors**: Cohere Embed, Mistral Embed, Voyage AI, Nomic Embed

**Code Models**: Optimized for code generation and understanding. Better at code completion, debugging, and code explanation tasks.
- **OpenAI**: GPT-4 (code capabilities), GPT-3.5-turbo (code)
- **Anthropic**: Claude 3.5 Sonnet (strong code capabilities)
- **Google**: Gemini Pro (code generation)
- **Other Vendors**: Meta CodeLlama, DeepSeek Coder, Mistral Code, StarCoder

**Multimodal Models**: Support both text and image inputs/outputs. Enable image analysis, visual question answering, and image generation workflows.
- **OpenAI**: GPT-4 Vision, GPT-4o (multimodal)
- **Anthropic**: Claude 3.5 Sonnet (vision), Claude 3 Opus (vision)
- **Google**: Gemini Pro Vision, Gemini Ultra (multimodal)
- **Other Vendors**: Meta Llama 3.1 (vision), Mistral Large (multimodal), Cohere Command R+ (multimodal)

📖 [Detailed Documentation](./understanding_models.md) - Comprehensive guide to model capabilities, selection criteria, and use case mapping.

📖 [LLM vs. Reasoning Problems](./llm_vs_reasoning_problems.md) - Problem types that suit chat models vs. reasoning models, with examples and selection guidelines.

---

## Prompt Engineering

📖 [Detailed Documentation](./prompt_engineering.md)

**Description**: The art and science of designing effective prompts to guide LLM behavior and achieve desired outputs. Involves structuring instructions, providing context, and using various techniques to improve response quality.

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

**Best Practices**:
- Use clear, specific instructions
- Provide examples for complex tasks
- Separate system prompts from user prompts
- Version control your prompts
- Test prompts systematically

**Use Cases**:
- Content generation
- Data extraction
- Code generation
- Question answering
- Translation

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
- **AWS Bedrock Agents**: Managed agents with native tool/function calling support
- **AWS Lambda**: Serverless functions for tool execution
- **OpenAI Function Calling**: Native support in GPT models
- **Anthropic Tool Use**: Claude's tool calling API
- **LangChain Tools**: Tool abstraction framework
- **AutoGPT**: Agent framework with tool use
- **CrewAI**: Multi-agent framework with tools

**Best Practices**:
- Define clear, specific function schemas
- Validate inputs before execution
- Handle errors gracefully
- Use tool descriptions effectively
- Implement retry logic

**Use Cases**:
- API integrations
- Database queries
- File operations
- Web scraping
- Calculator tools
- Code execution

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
- **JSON Schema**: Standard schema format
- **Zod**: TypeScript schema validation
- **Structured Outputs (OpenAI)**: Native structured output support
- **Outlines**: Structured generation library

**Best Practices**:
- Define strict schemas
- Handle parsing errors gracefully
- Provide fallback mechanisms
- Validate all outputs
- Use type-safe languages when possible

**Use Cases**:
- Data extraction
- API response generation
- Database record creation
- Configuration generation
- Form filling

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

**Best Practices**:
- Write clear, descriptive tool function descriptions
- Use detailed Pydantic Field descriptions
- Let schemas communicate requirements implicitly
- Keep prompts minimal when schemas provide context
- Combine with explicit prompts only when necessary

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

## RAG (Retrieval-Augmented Generation)

**Description**: Augmenting LLM inputs with relevant context retrieved from external knowledge bases or vector databases. Combines the power of semantic search with generative capabilities.

**Key Components**:
- Vector embeddings of documents
- Vector database for storage and retrieval
- Retrieval mechanism (semantic search)
- Context injection into prompts

**Popular Solutions**:
- **AWS Bedrock Knowledge Bases**: Managed RAG with vector search and data sources
- **Amazon Kendra**: Enterprise search service with AI-powered semantic search
- **Amazon OpenSearch Serverless**: Vector search and hybrid search capabilities
- **Pinecone**: Managed vector database
- **Weaviate**: Open-source vector database
- **Chroma**: Embedded vector database
- **Qdrant**: High-performance vector search
- **Milvus**: Scalable vector database
- **LangChain Vector Stores**: Abstraction layer for multiple backends
- **LlamaIndex**: Data framework for LLM applications

**Best Practices**:
- Chunk documents appropriately (200-500 tokens)
- Use metadata filtering for precision
- Implement hybrid search (keyword + semantic)
- Re-rank results for better relevance
- Monitor retrieval quality

**Use Cases**:
- Document Q&A systems
- Knowledge base assistants
- Research tools
- Customer support chatbots
- Legal document analysis

---

## Chain-of-Thought / Multi-Step Reasoning

**Description**: Encouraging LLMs to show explicit step-by-step reasoning processes. Improves accuracy on complex problems requiring logical reasoning.

**Key Techniques**:
- Explicit reasoning prompts ("Let's think step by step")
- Intermediate reasoning steps
- Self-verification
- Decomposition of complex problems

**Popular Solutions**:
- **Chain-of-Thought Prompting**: Original technique
- **ReAct (Reasoning + Acting)**: Combines reasoning with tool use
- **Self-Consistency**: Multiple reasoning paths
- **Tree of Thoughts**: Search-based reasoning
- **LangChain Chains**: Multi-step reasoning workflows

**Best Practices**:
- Break complex problems into steps
- Encourage explicit reasoning
- Verify intermediate results
- Use multiple reasoning paths for critical decisions
- Monitor reasoning quality

**Use Cases**:
- Mathematical problem solving
- Logical reasoning tasks
- Multi-step planning
- Code debugging
- Scientific analysis

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
- **AWS Strands Agents**: Open-source AI agents SDK with model-driven approach, supports Amazon Bedrock, Anthropic, Ollama, Meta, and other providers via LiteLLM. Used in production by Amazon Q Developer, AWS Glue, and VPC Reachability Analyzer. [GitHub](https://github.com/aws/strands-agents) | [AWS Blog](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)
- **AWS Bedrock Agents**: Managed agent service with tool use and knowledge bases
- **AWS Step Functions**: Workflow orchestration for agent systems
- **AutoGPT**: Early autonomous agent framework
- **LangGraph**: State machine for agent workflows
- **CrewAI**: Multi-agent orchestration
- **AutoGen**: Multi-agent conversation framework
- **Semantic Kernel**: Microsoft's agent framework
- **LlamaIndex Agents**: Agent framework with RAG

**Best Practices**:
- Define clear goals and constraints
- Implement safety limits (max steps, timeouts)
- Monitor agent behavior
- Use structured memory
- Implement human-in-the-loop for critical actions
- Leverage model-native capabilities (reasoning, tool use) rather than complex orchestration
- Use semantic search for tool selection when dealing with large tool sets (1000+ tools)

**Use Cases**:
- Research assistants
- Task automation
- Code generation and testing
- Data analysis workflows
- Customer service automation

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
- **AWS Step Functions**: Serverless workflow orchestration with LLM integration
- **AWS Bedrock**: Managed LLM service with orchestration capabilities
- **AWS Lambda**: Serverless functions for workflow steps
- **Amazon EventBridge**: Event-driven orchestration
- **LangChain**: Comprehensive orchestration framework
- **LlamaIndex**: Data and workflow orchestration
- **Prefect**: Workflow orchestration platform
- **Temporal**: Durable execution engine
- **Airflow**: Workflow management (can integrate LLMs)
- **Dagster**: Data orchestration with LLM support

**Best Practices**:
- Design idempotent workflows
- Implement comprehensive error handling
- Use state checkpoints
- Monitor workflow execution
- Design for failure recovery

**Use Cases**:
- Multi-step content generation
- Data processing pipelines
- ETL with LLM steps
- Approval workflows
- Batch processing

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
- **LangSmith**: LLM observability with evaluation
- **Weights & Biases**: Experiment tracking and evaluation
- **TruLens**: LLM evaluation framework
- **RAGAS**: RAG-specific evaluation metrics
- **DeepEval**: LLM evaluation framework
- **LangChain Evaluators**: Built-in evaluation tools

**Best Practices**:
- Use separate judge models for objectivity
- Define clear evaluation criteria
- Use structured evaluation outputs
- Combine multiple evaluation dimensions
- Validate judge consistency

**Use Cases**:
- Output quality assessment
- Hallucination detection
- Alignment verification
- A/B testing evaluation
- Automated testing

---

## Embeddings / Vector Search

📖 [Detailed Documentation](./embeddings_and_vector_search.md)

**Description**: Converting text into dense vector representations (embeddings) and using similarity search to find relevant content. Foundation for semantic search and RAG.

**Key Concepts**:
- Embedding models
- Vector similarity metrics (cosine, dot product)
- Approximate nearest neighbor search
- Dimensionality considerations

**Popular Solutions**:
- **AWS Bedrock Titan Embeddings**: Amazon's embedding models (multimodal support)
- **Amazon OpenSearch Serverless**: Vector search with embedding support
- **AWS Bedrock Knowledge Bases**: Integrated embedding and vector search
- **OpenAI Embeddings**: text-embedding-ada-002, text-embedding-3-small/large
- **Cohere Embeddings**: multilingual and domain-specific
- **Sentence Transformers**: Open-source embedding models
- **Voyage AI**: High-quality embeddings
- **Anthropic Embeddings**: Claude-based embeddings
- **FAISS**: Facebook's similarity search library
- **Annoy**: Approximate nearest neighbors

**Best Practices**:
- Choose appropriate embedding models for your domain
- Normalize vectors for cosine similarity
- Use appropriate dimensions (balance quality vs. cost)
- Consider multilingual models for global apps
- Monitor embedding quality

**Use Cases**:
- Semantic search
- Document similarity
- Recommendation systems
- Clustering
- Anomaly detection

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
- **Semantic Similarity Example Selection**: Choose examples by similarity
- **Custom Example Stores**: Store and retrieve examples

**Best Practices**:
- Select diverse, representative examples
- Order examples strategically
- Match example format to desired output
- Use 2-5 examples typically
- Update examples based on performance

**Use Cases**:
- Style transfer
- Format learning
- Task-specific behavior
- Domain adaptation
- Custom output formats

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

**Best Practices**:
- Define clear agent roles
- Establish communication protocols
- Implement conflict resolution
- Monitor agent interactions
- Design for scalability

**Use Cases**:
- Complex research tasks
- Software development teams
- Content creation workflows
- Data analysis pipelines
- Customer service teams

---

## Memory / State Management

**Description**: Persisting context, conversation history, and state across LLM interactions. Enables long-term memory and context retention.

**Key Types**:
- Short-term memory (conversation history)
- Long-term memory (persistent storage)
- Episodic memory (specific events)
- Semantic memory (facts and knowledge)

**Popular Solutions**:
- **Amazon DynamoDB**: NoSQL database for state and memory storage
- **Amazon ElastiCache (Redis)**: Fast in-memory storage for session memory
- **Amazon RDS (PostgreSQL)**: Persistent relational storage for long-term memory
- **Amazon MemoryDB**: Redis-compatible with persistence
- **LangChain Memory**: Various memory types
- **LlamaIndex Memory**: Persistent memory for agents
- **Vector Store Memory**: Semantic memory storage
- **Zep**: Long-term memory for AI applications

**Best Practices**:
- Separate short-term and long-term memory
- Implement memory summarization
- Use vector search for semantic memory
- Implement memory pruning
- Respect privacy and data retention policies

**Use Cases**:
- Conversational AI
- Personal assistants
- Customer relationship management
- Learning systems
- Context-aware applications

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
- **Amazon Comprehend**: Content moderation, PII detection, and toxicity analysis
- **AWS Bedrock Guardrails**: Built-in content filtering and safety controls
- **Amazon Macie**: PII and sensitive data detection
- **NVIDIA NeMo Guardrails**: Open-source guardrails framework
- **Azure Content Safety**: Microsoft's content moderation
- **OpenAI Moderation API**: Content moderation service
- **Perspective API**: Toxicity detection
- **Presidio**: PII detection and anonymization
- **LangChain Guardrails**: Safety integrations

**Best Practices**:
- Implement multiple layers of safety
- Validate all user inputs
- Monitor outputs continuously
- Use human review for sensitive content
- Keep safety rules updated

**Use Cases**:
- Public-facing applications
- Educational platforms
- Healthcare applications
- Financial services
- Content moderation

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

**Best Practices**:
- Implement proper error handling
- Support cancellation
- Handle partial responses gracefully
- Provide loading indicators
- Optimize for low latency

**Use Cases**:
- Real-time chatbots
- Live content generation
- Interactive applications
- Code generation
- Translation services

---

## Caching / Optimization

**Description**: Caching LLM responses, embeddings, or intermediate results to reduce latency, costs, and improve performance.

**Key Strategies**:
- Response caching
- Embedding caching
- Prompt caching
- Semantic caching
- Result memoization

**Popular Solutions**:
- **Amazon ElastiCache (Redis/Memcached)**: Managed caching service
- **Amazon DynamoDB**: Fast NoSQL caching with TTL support
- **Amazon CloudFront**: CDN caching for API responses
- **GPTCache**: Semantic cache for LLM responses
- **LangChain Cache**: Multiple caching backends
- **SQLite**: Lightweight local cache
- **Vector Cache**: Semantic similarity caching

**Best Practices**:
- Cache expensive operations
- Use semantic caching for similar queries
- Implement cache invalidation strategies
- Monitor cache hit rates
- Balance freshness vs. performance

**Use Cases**:
- Cost reduction
- Latency improvement
- Rate limit management
- Repeated queries
- Batch processing

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

**Best Practices**:
- Limit search space for performance
- Use heuristics to guide search
- Evaluate paths efficiently
- Implement pruning strategies
- Balance exploration vs. exploitation

**Use Cases**:
- Complex problem solving
- Multi-step planning
- Code generation
- Mathematical proofs
- Strategic decision making

---

## Semantic Caching

**Description**: Caching based on semantic similarity rather than exact matches. Returns cached results for semantically similar queries.

**Key Features**:
- Semantic similarity matching
- Embedding-based comparison
- Similarity thresholds
- Cache key generation from embeddings

**Popular Solutions**:
- **GPTCache**: Semantic cache implementation
- **LangChain Semantic Cache**: Semantic caching support
- **Custom Vector Cache**: Build with vector DBs
- **Redis with Embeddings**: Vector similarity in Redis

**Best Practices**:
- Set appropriate similarity thresholds
- Use high-quality embeddings
- Monitor cache quality
- Implement cache warming
- Balance precision vs. recall

**Use Cases**:
- Similar query handling
- Cost reduction
- Latency improvement
- User experience enhancement
- API rate limit management

---

## Hybrid Search

**Description**: Combining keyword (BM25) and semantic (vector) search for improved retrieval accuracy. Leverages strengths of both approaches.

**Key Components**:
- Keyword search (BM25, TF-IDF)
- Semantic search (vector embeddings)
- Result fusion/reranking
- Weighted combination

**Popular Solutions**:
- **Amazon OpenSearch Serverless**: Hybrid search with BM25 and vector capabilities
- **Amazon Kendra**: Enterprise search with hybrid keyword and semantic search
- **AWS Bedrock Knowledge Bases**: Integrated hybrid search
- **Weaviate Hybrid Search**: Native hybrid search
- **Pinecone Hybrid Search**: Keyword + vector
- **Elasticsearch**: BM25 + vector search
- **Qdrant Hybrid Search**: Combined search
- **Vespa**: Hybrid search engine

**Best Practices**:
- Tune keyword/semantic weights
- Use appropriate reranking
- Combine results effectively
- Monitor search quality
- A/B test configurations

**Use Cases**:
- Document search
- E-commerce search
- Knowledge base search
- Content discovery
- RAG systems

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
- **Selective Context**: Choose relevant parts

**Best Practices**:
- Preserve critical information
- Test compressed prompts
- Monitor quality after compression
- Use compression selectively
- Balance size vs. quality

**Use Cases**:
- Long context management
- Cost reduction
- Token limit optimization
- Performance improvement
- Batch processing

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
- **AWS Bedrock**: Access to multiple foundation models (Claude, Llama, Titan, etc.)
- **Amazon Bedrock Model Selection**: Route requests to different models
- **AWS Lambda**: Custom routing logic and model selection
- **OpenRouter**: Model routing service
- **LangChain Router Chains**: Route to different models
- **Custom Routing Logic**: Build routing systems
- **Model Ensembling**: Combine multiple models
- **Fallback Chains**: Automatic model switching

**Best Practices**:
- Define clear routing criteria
- Monitor routing decisions
- Implement fallback mechanisms
- Test all routes
- Optimize for cost and performance

**Use Cases**:
- Cost optimization
- Performance optimization
- Specialized task handling
- Reliability improvement
- Multi-model systems

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
- **Base10**: Custom model training and fine-tuning platform
- **AWS Bedrock Custom Models**: Fine-tune foundation models with your data
- **Amazon SageMaker**: End-to-end ML platform for model training and deployment
- **Hugging Face Transformers**: Open-source library with training utilities
- **OpenAI Fine-tuning API**: Fine-tune GPT models on custom datasets
- **Anthropic Fine-tuning**: Customize Claude models for specific tasks
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning method
- **QLoRA**: Quantized LoRA for memory-efficient fine-tuning
- **PEFT (Parameter-Efficient Fine-Tuning)**: Hugging Face library for efficient fine-tuning
- **Weights & Biases**: Experiment tracking for model training
- **MLflow**: Model lifecycle management and tracking

**Best Practices**:
- Start with high-quality, representative training data
- Use parameter-efficient methods (LoRA) when possible to reduce costs
- Validate data quality and diversity before training
- Implement proper train/validation/test splits
- Monitor for overfitting and model degradation
- Version control datasets and model checkpoints
- Evaluate on held-out test sets
- Consider domain-specific evaluation metrics
- Implement data augmentation when appropriate
- Use transfer learning from strong foundation models

**Use Cases**:
- Domain-specific language models (legal, medical, technical)
- Custom style or tone adaptation
- Task-specific optimization (classification, extraction, generation)
- Multilingual model adaptation
- Specialized code generation models
- Custom chatbot personalities
- Industry-specific assistants
- Brand voice alignment

---

## Implementation Considerations

### Choosing the Right Techniques

1. **Start Simple**: Begin with prompt engineering and structured outputs
2. **Add Complexity Gradually**: Introduce RAG, function calling, and orchestration as needed
3. **Measure Impact**: Track metrics for each technique you implement
4. **Consider Costs**: Balance functionality with API costs
5. **Plan for Scale**: Design systems that can handle growth

### Common Patterns

- **RAG + Function Calling**: Combine knowledge retrieval with tool use
- **Orchestration + Caching**: Coordinate workflows with performance optimization
- **Prompt Engineering + Structured Output**: Reliable, validated outputs
- **Function Calling + Structured Output + Schema-Driven Inference**: Tools gather data, schemas guide output, minimal prompts
- **Multi-Agent + Memory**: Collaborative systems with context retention

### Best Practices Across All Techniques

- **Observability**: Monitor all LLM interactions
- **Error Handling**: Implement comprehensive error handling
- **Testing**: Test each component thoroughly
- **Documentation**: Document your implementations
- **Version Control**: Version prompts, schemas, and configurations
- **Security**: Implement proper authentication and authorization
- **Cost Management**: Monitor and optimize costs continuously

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
