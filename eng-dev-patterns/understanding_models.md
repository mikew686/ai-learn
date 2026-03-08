# Understanding Models

## Overview

Models exposed via OpenRouter and similar services fall into categories by primary capability. Category informs selection for a given task: chat for dialogue and tool use, reasoning for step-by-step logic, embeddings for vector search, and so on. Performance, cost, and latency vary by class and vendor.

## Description

**Chat/conversational models** handle dialogue, function calling, structured outputs, and streaming; they are the default for most application patterns. **Reasoning models** are trained to reason step-by-step before answering and suit multi-step logic and math. **Fast/cheap models** trade capability for speed and cost. **Embedding models** produce vector representations for semantic search and RAG. **Code models** target code generation and understanding. **Multimodal models** accept or produce text and images. **Hybrid** systems route between chat and reasoning. Vendor offerings (OpenAI, Anthropic, Google, etc.) map into these classes; APIs differ by provider but follow similar chat/completion and embedding patterns.

**Key concepts**

- **Chat/conversational**: General-purpose dialogue, tools, structured output, streaming.
- **Reasoning**: Explicit chain-of-thought, better on logic and multi-step problems; higher latency and token use.
- **Fast/cheap**: Lower cost and latency, simpler tasks.
- **Embedding**: Vector output for similarity search and RAG.
- **Code**: Tuned for code completion, explanation, and generation.
- **Multimodal**: Text and image inputs/outputs.
- **Hybrid**: Routing or combined chat/reasoning behavior.

## Translation Example

Translation flow using two model types: chat for the translation itself, embeddings for storing or retrieving similar phrases (e.g. for few-shot). The model identifier selects the capability.

```python
from openai import OpenAI

client = OpenAI()

# Chat model: produce the translation
chat_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a translator. Reply with only the translation."},
        {"role": "user", "content": "Translate to French: Hello, how are you?"},
    ],
    max_tokens=100,
)
translation = chat_response.choices[0].message.content
print(translation)

# Embedding model: vector for the phrase (e.g. for similarity search over past translations)
embed_response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Translate to French: Hello, how are you?",
)
vector = embed_response.data[0].embedding  # store or compare for few-shot retrieval
print(len(vector))  # e.g. 1536
```

Switching the chat `model` (e.g. to a reasoning model) changes behavior and cost; for simple translation, a chat or fast model is typical. See the section on LLM vs. reasoning problems for when to use chat vs. reasoning models.
