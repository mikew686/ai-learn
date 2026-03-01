# Structured Output

## Overview

Structured output is the use of schemas to constrain LLM responses so they conform to a fixed format and can be parsed and validated reliably. Outputs become directly usable in code without ad-hoc parsing.

## Description

Schemas (e.g. JSON Schema or Pydantic models) define the expected shape and types of the response. The API is called with this schema; the model returns JSON that fits it, and the client parses it into typed objects. OpenAI supports structured output via `response_format` and a `.parse()`-style API that returns Pydantic instances. Validation and error handling apply when the model output does not match the schema.

**Key concepts**

- Schema definition (JSON Schema, Pydantic models)
- Response validation and type safety
- Automatic parsing into application types
- Error handling for invalid or malformed output

## Translation Example

OpenAI chat with a Pydantic response format for a translation result. The reply is parsed into a typed object (source, translation, language code, optional notes).

```python
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()

class TranslationResult(BaseModel):
    """Structured translation result."""
    source_text: str = Field(description="Original text to translate")
    translated_text: str = Field(description="Translation in the target language")
    target_language_code: str = Field(description="ISO 639-1 code, e.g. fr, es")
    notes: str | None = Field(default=None, description="Optional translator note or alternate phrasing")

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Translate 'Hello, how are you?' to French (Quebec). "
         "Return the translation and the target language code."}
    ],
    response_format=TranslationResult,
)

parsed = response.choices[0].message.parsed  # TranslationResult instance
print(parsed.translated_text, parsed.target_language_code)
```

The model’s reply is constrained to the schema; `.parse()` returns a validated Pydantic object. Field descriptions guide what to include (e.g. target language code, notes).
