# Schema-Driven Inference

## Overview

Schema-driven inference uses structured definitions—tool schemas, Pydantic field descriptions, JSON Schema—as implicit prompts. The model infers behavior and requirements from this metadata, reducing the need for long explicit instructions while still producing validated, structured outputs.

## Description

Modern LLMs use schema metadata (function descriptions, parameter descriptions, Pydantic field descriptions) to infer what to do and what to return. Detailed tool descriptions indicate when and how to call a tool; detailed field descriptions indicate what to extract or generate. Explicit prompt text can stay short when schemas carry the requirements. The pattern often appears together with function calling and structured output.

**Key concepts**

- Tool schema descriptions that guide when and how the model uses tools
- Pydantic field descriptions that specify data requirements and format
- Structured definitions acting as implicit prompts
- Inference of behavior from schema metadata rather than long instructions
- Reduced prompt length with maintained output quality and validation

## Translation Example

A translation tool and a Pydantic translation result. The tool description tells the model when to resolve a language code; the Pydantic fields describe the expected translation output. The user prompt can stay minimal.

```python
from openai import OpenAI
from pydantic import BaseModel, Field
import json

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "lookup_language_code",
        "description": "Look up ISO 639-1 language code for a language name. Use when you need to convert a language name (e.g. 'French', 'Spanish') to its code ('fr', 'es').",
        "parameters": {
            "type": "object",
            "properties": {"language_name": {"type": "string", "description": "Language name in English"}},
            "required": ["language_name"],
        },
    },
}]

class TranslationResult(BaseModel):
    source_text: str = Field(description="Original text")
    translated_text: str = Field(description="Translation in the target language")
    target_language_code: str = Field(description="ISO 639-1 code, e.g. fr, es")

# Minimal prompt; schema descriptions carry the requirements
messages = [{"role": "user", "content": "Translate 'Hello' to French and return the structured result."}]
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
)
# ... handle tool call if present, then request structured output with response_format=TranslationResult
```

The tool description tells the model when to use it; the Pydantic fields tell it what to include in the final answer. Explicit instructions in the prompt can be brief.
