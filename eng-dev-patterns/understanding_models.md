# Understanding Models

## Overview

Understanding model capabilities and categories is essential for selecting the right model for your use case. Different models are optimized for different tasks, and choosing appropriately can significantly impact performance, cost, and user experience. This guide categorizes models by their primary capabilities and provides technical details on how to use them effectively.

## Model Categories

Models available through OpenRouter and similar services can be categorized into six general classes:

1. **Chat/Conversational Models** - General-purpose dialogue and text generation
2. **Reasoning Models** - Explicit step-by-step reasoning and problem-solving
3. **Fast/Cheap Models** - Lightweight models optimized for speed and cost
4. **Embedding Models** - Vector representations for semantic search and RAG
5. **Code Models** - Specialized for code generation and understanding
6. **Multimodal Models** - Support text and image inputs/outputs

---

## Chat/Conversational Models

### Description

General-purpose models optimized for dialogue and text generation. These are the most versatile models, supporting function calling, structured outputs, streaming, and a wide range of application patterns. They form the foundation for most AI applications.

### Key Capabilities

- **Function Calling / Tool Use**: Models can call external functions and APIs
- **Structured Outputs**: Support for JSON Schema and Pydantic model parsing
- **Streaming**: Token-by-token response generation for real-time feedback
- **Context Management**: Handle long conversation histories and context windows
- **Multi-turn Conversations**: Maintain context across multiple exchanges
- **Instruction Following**: Strong adherence to system prompts and user instructions

### Technical Usage

#### Basic Chat Completion

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto"
)
```

#### Structured Outputs

```python
from pydantic import BaseModel, Field

class UserInfo(BaseModel):
    name: str = Field(description="User's full name")
    email: str = Field(description="Email address")
    age: int = Field(description="Age in years", ge=0, le=150)

response = client.beta.chat.completions.parse(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract user info: John Doe, john@example.com, 30"}],
    response_format=UserInfo
)

user = response.choices[0].message.parsed
```

#### Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Model Examples

- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Google**: Gemini Pro, Gemini Ultra
- **Other Vendors**: Meta Llama 3, Mistral Large, Cohere Command, DeepSeek Chat, xAI Grok

### Best Practices

- Use system messages to set role and behavior
- Leverage function calling for external integrations
- Use structured outputs for reliable parsing
- Implement streaming for better user experience
- Monitor token usage and optimize context length
- Choose appropriate temperature (lower for factual, higher for creative)

### Documentation Links

- **OpenAI Chat Completions**: https://platform.openai.com/docs/guides/text-generation
- **Anthropic Messages API**: https://docs.anthropic.com/claude/docs/getting-started-with-the-api
- **Google Gemini API**: https://ai.google.dev/docs
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **OpenAI Structured Outputs**: https://platform.openai.com/docs/guides/structured-outputs

---

## Reasoning Models

### Description

Models specifically designed for explicit step-by-step reasoning. These models show their "thinking process" and are better at complex problem-solving, multi-step analysis, and tasks requiring logical deduction. They excel at mathematical problems, code debugging, and strategic planning.

### Key Capabilities

- **Explicit Reasoning**: Show intermediate reasoning steps
- **Self-Verification**: Check their own work before finalizing answers
- **Problem Decomposition**: Break complex problems into manageable steps
- **Multi-Step Planning**: Plan and execute sequences of logical steps
- **Error Correction**: Identify and correct mistakes in reasoning chains

### Technical Usage

#### Basic Reasoning

```python
response = client.chat.completions.create(
    model="o1-preview",
    messages=[{
        "role": "user",
        "content": "Solve this step by step: If a train travels 120 miles in 2 hours, and another train travels 180 miles in 3 hours, which train is faster?"
    }]
)

print(response.choices[0].message.content)
```

#### Reasoning with Tool Use

Reasoning models can combine explicit reasoning with tool calls:

```python
response = client.chat.completions.create(
    model="o1-preview",
    messages=[{
        "role": "user",
        "content": "I need to plan a trip. First, let me think about what information I need, then use tools to gather it."
    }],
    tools=[weather_tool, flight_tool]
)
```

### Model Examples

- **OpenAI**: o1-preview, o1-mini, o3-mini
- **Anthropic**: Claude 3.5 Sonnet (with reasoning capabilities via chain-of-thought prompting)
- **Google**: Gemini Pro (with chain-of-thought prompting)
- **Other Vendors**: DeepSeek R1, Meta Llama 3 (with CoT prompting)

### When to Use

- Complex mathematical problems
- Multi-step logical reasoning
- Code debugging and analysis
- Strategic planning
- Scientific problem-solving
- Tasks requiring verification

### Best Practices

- Use explicit reasoning prompts ("Let's think step by step")
- Allow models to show intermediate steps
- Verify reasoning chains for critical decisions
- Use for tasks where correctness is more important than speed
- Combine with tool use for real-world data gathering

### Documentation Links

- **OpenAI Reasoning Models**: https://platform.openai.com/docs/guides/reasoning
- **OpenAI Reasoning Best Practices**: https://platform.openai.com/docs/guides/reasoning-best-practices
- **Chain-of-Thought Prompting**: https://arxiv.org/abs/2201.11903

---

## Fast/Cheap Models

### Description

Lightweight models optimized for speed and cost efficiency. These models trade some capability for significantly lower latency and cost, making them ideal for high-volume operations, simple tasks, and applications where speed is critical.

### Key Capabilities

- **Low Latency**: Fast response times (often < 1 second)
- **Cost Effective**: Significantly cheaper per token
- **High Throughput**: Can handle many concurrent requests
- **Simple Tasks**: Well-suited for straightforward operations
- **Scaling**: Enable cost-effective scaling to high volumes

### Technical Usage

#### High-Volume Processing

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def process_batch(texts):
    tasks = [
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            max_tokens=100
        )
        for text in texts
    ]
    return await asyncio.gather(*tasks)

# Process 100 items concurrently
results = await process_batch(large_text_list)
```

#### Simple Classification

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": f"Classify this text as positive, negative, or neutral: {text}"
    }],
    temperature=0,  # Deterministic for classification
    max_tokens=10
)
```

### Model Examples

- **OpenAI**: GPT-3.5-turbo, GPT-4o-mini
- **Anthropic**: Claude 3 Haiku
- **Google**: Gemini Flash
- **Other Vendors**: Meta Llama 3.1 8B, Mistral Small, Cohere Command Light, DeepSeek Chat 7B

### When to Use

- High-volume operations
- Simple classification tasks
- Content moderation
- Basic Q&A
- Text preprocessing
- When latency is critical
- Cost-sensitive applications

### Best Practices

- Use for simple, well-defined tasks
- Batch requests when possible
- Use async/concurrent processing
- Set lower temperature for consistency
- Monitor cost vs. quality trade-offs
- Consider caching for repeated queries

### Documentation Links

- **OpenAI Model Pricing**: https://openai.com/api/pricing/
- **OpenAI Rate Limits**: https://platform.openai.com/docs/guides/rate-limits
- **Cost Optimization**: https://platform.openai.com/docs/guides/production-best-practices

---

## Embedding Models

### Description

Specialized models that convert text into dense vector representations (embeddings). These vectors capture semantic meaning and enable similarity search, clustering, and retrieval-augmented generation (RAG). Embeddings are the foundation for semantic search and many knowledge-based applications.

### Key Capabilities

- **Semantic Representation**: Convert text to numerical vectors
- **Similarity Search**: Find semantically similar content
- **Dimensionality**: Typically 384-1536 dimensions
- **Multilingual Support**: Many models support multiple languages
- **Domain Adaptation**: Some models fine-tuned for specific domains

### Technical Usage

#### Generating Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog"
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
```

#### Batch Embedding Generation

```python
texts = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks",
    "Natural language processing enables text understanding"
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [item.embedding for item in response.data]
```

#### Semantic Similarity

```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Generate embeddings
emb1 = client.embeddings.create(
    model="text-embedding-3-small",
    input="Python programming language"
).data[0].embedding

emb2 = client.embeddings.create(
    model="text-embedding-3-small",
    input="Coding in Python"
).data[0].embedding

similarity = cosine_similarity(emb1, emb2)
print(f"Similarity: {similarity:.3f}")
```

#### RAG with Embeddings

```python
# 1. Generate embeddings for documents
documents = ["Document 1 text...", "Document 2 text...", ...]
doc_embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
).data

# 2. Store in vector database (simplified example)
vector_db = {i: emb.embedding for i, emb in enumerate(doc_embeddings)}

# 3. Query embedding
query = "What is machine learning?"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# 4. Find most similar documents
similarities = [
    cosine_similarity(query_embedding, vec)
    for vec in vector_db.values()
]
most_similar_idx = np.argmax(similarities)

# 5. Use retrieved document in RAG
retrieved_doc = documents[most_similar_idx]
```

### Model Examples

- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Anthropic**: (Embeddings via third-party services)
- **Google**: text-embedding-004, textembedding-gecko
- **Other Vendors**: Cohere Embed, Mistral Embed, Voyage AI, Nomic Embed

### When to Use

- Semantic search
- RAG (Retrieval-Augmented Generation)
- Document similarity
- Clustering and classification
- Recommendation systems
- Anomaly detection

### Best Practices

- Normalize embeddings for cosine similarity
- Use appropriate dimensions (balance quality vs. cost)
- Batch embedding generation for efficiency
- Cache embeddings for frequently accessed content
- Choose models based on language and domain
- Monitor embedding quality with evaluation metrics

### Documentation Links

- **OpenAI Embeddings Guide**: https://platform.openai.com/docs/guides/embeddings
- **OpenAI Embedding Models**: https://platform.openai.com/docs/models/embeddings
- **Vector Database Guide**: https://www.pinecone.io/learn/vector-database/
- **RAG Tutorial**: https://cookbook.openai.com/examples/rag_with_openai_embeddings

---

## Code Models

### Description

Models optimized specifically for code generation, understanding, and manipulation. These models excel at code completion, debugging, refactoring, explanation, and understanding code across multiple programming languages. They're trained on large codebases and understand programming patterns, syntax, and best practices.

### Key Capabilities

- **Code Generation**: Generate code from natural language descriptions
- **Code Completion**: Autocomplete code snippets
- **Code Explanation**: Explain what code does
- **Debugging**: Identify and fix bugs
- **Refactoring**: Improve code structure and style
- **Multi-language Support**: Work with many programming languages
- **Code Understanding**: Parse and understand complex codebases

### Technical Usage

#### Code Generation

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "system",
        "content": "You are an expert Python programmer. Write clean, well-documented code."
    }, {
        "role": "user",
        "content": "Write a Python function that calculates the Fibonacci sequence up to n terms"
    }],
    temperature=0.3  # Lower temperature for more deterministic code
)

code = response.choices[0].message.content
```

#### Code Explanation

```python
code_snippet = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Explain what this code does:\n\n```python\n{code_snippet}\n```"
    }]
)
```

#### Code Debugging

```python
buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Potential division by zero
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Find and fix bugs in this code:\n\n```python\n{buggy_code}\n```"
    }]
)
```

#### Code Refactoring

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Refactor this code to be more Pythonic and follow PEP 8:\n\n```python\n{code}\n```"
    }]
)
```

### Model Examples

- **OpenAI**: GPT-4 (code capabilities), GPT-3.5-turbo (code)
- **Anthropic**: Claude 3.5 Sonnet (strong code capabilities)
- **Google**: Gemini Pro (code generation)
- **Other Vendors**: Meta CodeLlama, DeepSeek Coder, Mistral Code, StarCoder

### When to Use

- Code generation from specifications
- Code completion and autocomplete
- Code review and quality checks
- Debugging assistance
- Code documentation generation
- Refactoring and optimization
- Learning programming concepts

### Best Practices

- Use lower temperature (0.2-0.3) for deterministic code
- Provide clear specifications and requirements
- Include examples of desired code style
- Use system prompts to set coding standards
- Test generated code thoroughly
- Consider code security implications
- Use structured outputs for code extraction

### Documentation Links

- **OpenAI Code Generation**: https://platform.openai.com/docs/guides/code-generation
- **GitHub Copilot**: https://github.com/features/copilot
- **CodeLlama Documentation**: https://ai.meta.com/blog/code-llama-large-language-model-coding/
- **Best Practices for Code Models**: https://platform.openai.com/docs/guides/production-best-practices

---

## Multimodal Models

### Description

Models that support both text and image inputs and/or outputs. These models can analyze images, answer questions about visual content, generate image descriptions, and in some cases, generate images. They enable applications that combine visual understanding with language processing.

### Key Capabilities

- **Image Analysis**: Understand and describe image content
- **Visual Question Answering**: Answer questions about images
- **Image Description**: Generate detailed descriptions of images
- **Text + Image Input**: Process both text and images together
- **Image Generation**: Some models can generate images (separate category)
- **Document Understanding**: Analyze documents with text and images

### Technical Usage

#### Image Analysis

```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "photo.jpg"
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image? Describe it in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }],
    max_tokens=300
)
```

#### Visual Question Answering

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How many people are in this image and what are they doing?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }]
)
```

#### Document Analysis

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Extract all text and data from this invoice. Format as JSON."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{invoice_image}"
                }
            }
        ]
    }]
)
```

#### Using URLs Instead of Base64

```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this image"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
    }]
)
```

### Model Examples

- **OpenAI**: GPT-4 Vision, GPT-4o (multimodal)
- **Anthropic**: Claude 3.5 Sonnet (vision), Claude 3 Opus (vision)
- **Google**: Gemini Pro Vision, Gemini Ultra (multimodal)
- **Other Vendors**: Meta Llama 3.1 (vision), Mistral Large (multimodal), Cohere Command R+ (multimodal)

### When to Use

- Image analysis and description
- Visual question answering
- Document understanding (invoices, forms, etc.)
- Content moderation (image analysis)
- Accessibility (image descriptions)
- Visual data extraction
- Image-based search

### Best Practices

- Optimize image size (models have size limits)
- Use appropriate image formats (JPEG, PNG)
- Provide clear text prompts for image analysis
- Consider image resolution vs. cost trade-offs
- Handle multiple images in sequence if needed
- Use structured outputs for extracted data
- Consider privacy implications of image processing

### Documentation Links

- **OpenAI Vision Guide**: https://platform.openai.com/docs/guides/vision
- **Anthropic Vision**: https://docs.anthropic.com/claude/docs/vision
- **Google Gemini Vision**: https://ai.google.dev/docs/gemini_api_overview#vision
- **Image Processing Best Practices**: https://platform.openai.com/docs/guides/vision

---

## Model Selection Guide

### Decision Framework

1. **Task Complexity**: Simple tasks → Fast/Cheap models; Complex reasoning → Reasoning models
2. **Cost Sensitivity**: High volume → Fast/Cheap; Quality critical → Premium models
3. **Latency Requirements**: Real-time → Fast/Cheap; Can wait → More capable models
4. **Input Type**: Text only → Chat models; Images → Multimodal; Code → Code models
5. **Output Requirements**: Structured → Chat with structured outputs; Vectors → Embeddings

### Cost vs. Capability Trade-offs

| Model Type | Cost | Capability | Speed | Best For |
|------------|------|------------|-------|----------|
| Fast/Cheap | Low | Medium | Fast | High volume, simple tasks |
| Chat/Conversational | Medium | High | Medium | Most applications |
| Reasoning | High | Very High | Slow | Complex problem-solving |
| Embeddings | Low | Specialized | Fast | Search, RAG |
| Code | Medium | High (code) | Medium | Code tasks |
| Multimodal | High | High | Medium | Image + text tasks |

### Provider Selection

- **OpenAI**: Strong API, comprehensive features, good documentation
- **Anthropic**: Excellent safety, strong reasoning, good for production
- **Google**: Competitive pricing, strong multimodal, good integration
- **Other Vendors**: Often more cost-effective, open-source options, specialized models

### Cost Comparison Resources

- **OpenRouter Models & Pricing**: https://openrouter.ai/models - Compare costs across multiple models and providers in one place. Useful for finding cost-effective alternatives across all model categories.

---

## Best Practices Across All Models

### General Guidelines

1. **Start with Chat Models**: Most versatile, good starting point
2. **Use Appropriate Models**: Don't use reasoning models for simple tasks
3. **Monitor Costs**: Track token usage and optimize
4. **Handle Errors**: Implement retry logic and error handling
5. **Cache When Possible**: Reduce redundant API calls
6. **Use Streaming**: Improve perceived latency
7. **Set Appropriate Temperature**: Lower for deterministic, higher for creative
8. **Version Control Prompts**: Track prompt changes and their impact

### Common Parameters by Model Category

Understanding key parameters and their effects helps optimize model behavior for each use case:

#### Chat/Conversational Models

**Temperature** (0.0-2.0):
- **0.0-0.3**: Deterministic, factual responses. Best for data extraction, classification, structured outputs
- **0.4-0.7**: Balanced creativity and consistency. Good default for most applications
- **0.8-1.2**: More creative and varied. Good for content generation, brainstorming
- **1.3-2.0**: Highly creative, less predictable. Use sparingly for artistic tasks

**Max Tokens**:
- Set based on expected response length
- Too low: Responses may be cut off mid-sentence
- Too high: Wastes tokens and increases latency
- Typical ranges: 100-500 (short), 500-2000 (medium), 2000+ (long)

**Top-p (Nucleus Sampling)** (0.0-1.0):
- Controls diversity by limiting token selection probability
- **0.1-0.5**: More focused, deterministic
- **0.6-0.9**: Balanced (often used with temperature)
- **0.9-1.0**: More diverse outputs
- Often used instead of or alongside temperature

**Frequency/Presence Penalty** (-2.0 to 2.0):
- **Frequency Penalty**: Reduces repetition of tokens
- **Presence Penalty**: Encourages new topics/concepts
- **0.0**: No penalty (default)
- **0.1-0.6**: Moderate reduction in repetition
- **0.7-2.0**: Strong reduction, may affect coherence

#### Reasoning Models

**Temperature**:
- Often fixed or very low (0.0-0.2) - reasoning requires consistency
- Some models don't expose temperature (e.g., o1-series)

**Max Tokens**:
- Typically higher (2000-8000+) to allow for reasoning steps
- Reasoning models show intermediate steps, requiring more tokens

**Note**: Reasoning models often have fewer adjustable parameters as they're optimized for step-by-step thinking

#### Fast/Cheap Models

**Temperature**:
- **0.0-0.3**: Recommended for simple tasks (classification, extraction)
- Lower temperature improves consistency and reduces retries

**Max Tokens**:
- Keep low (50-200) for simple tasks to minimize cost
- Only increase if task requires longer responses

**Top-p**:
- Often set lower (0.5-0.7) for more focused outputs
- Reduces need for retries, saving cost

#### Embedding Models

**Dimensions** (model-specific):
- **384-512**: Fast, lower cost, good for most use cases
- **768-1024**: Balanced quality and cost
- **1536+**: Higher quality, more expensive
- Choose based on dataset size and quality requirements

**Normalize**:
- Most embedding APIs normalize by default
- Required for cosine similarity calculations
- Ensures vectors are unit length

#### Code Models

**Temperature**:
- **0.0-0.3**: Recommended for code generation (deterministic, correct)
- Higher temperatures can introduce bugs or non-functional code

**Max Tokens**:
- Set based on expected code length
- Consider function/class boundaries
- Typical: 500-2000 for functions, 2000+ for larger blocks

**Top-p**:
- Lower values (0.5-0.7) preferred for code correctness
- Reduces chance of generating invalid syntax

#### Multimodal Models

**Temperature**:
- **0.0-0.5**: For factual image analysis and descriptions
- **0.6-0.9**: For creative image interpretation

**Max Tokens**:
- Image descriptions: 200-500
- Detailed analysis: 500-1500
- Multi-image tasks: 1000+

**Image Parameters**:
- **Detail**: "low" (faster, cheaper) vs "high" (more detailed analysis)
- **Size Limits**: Varies by model (typically 4-20MB)
- **Format**: JPEG, PNG, WebP supported

### Parameter Interaction

- **Temperature + Top-p**: Often used together. Temperature controls randomness, Top-p controls diversity
- **Lower Temperature + Lower Top-p**: Most deterministic, best for factual tasks
- **Higher Temperature + Higher Top-p**: Most creative, best for generation tasks
- **Frequency/Presence Penalty**: Works with both temperature and top-p to reduce repetition

### Performance Optimization

- Batch requests when possible
- Use async/concurrent processing
- Implement request queuing for rate limits
- Cache embeddings and common responses
- Optimize context length
- Use appropriate model sizes

### Cost Optimization

- Choose cheaper models for simple tasks
- Cache frequently used embeddings
- Optimize prompt length
- Use streaming to reduce latency costs
- Monitor and set usage limits
- Consider fine-tuning for domain-specific tasks

---

## Self-Learning Resources

### Official Documentation

- **OpenAI Platform**: https://platform.openai.com/docs
- **Anthropic Documentation**: https://docs.anthropic.com/
- **Google AI Studio**: https://aistudio.google.com/
- **OpenRouter Models & Pricing**: https://openrouter.ai/models - Unified API for multiple models with cost comparisons across providers

### Model Comparison Resources

- **LLM Comparison**: https://lmsys.org/blog/2024-01-17-arena/
- **Model Cards**: Check provider websites for detailed model specifications
- **Benchmark Results**: Hugging Face Open LLM Leaderboard

### Learning Paths

1. Start with Chat/Conversational models (most versatile)
2. Learn function calling and structured outputs
3. Explore embeddings for RAG applications
4. Try reasoning models for complex problems
5. Experiment with multimodal for image tasks
6. Optimize with fast/cheap models for scale

### Community Resources

- **OpenAI Cookbook**: https://cookbook.openai.com/
- **Hugging Face**: https://huggingface.co/
- **LangChain Documentation**: https://python.langchain.com/
- **Reddit r/LocalLLaMA**: Community discussions on models

---

*Last Updated: January 2026*
