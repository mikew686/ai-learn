# Prompt Engineering

## Overview

Prompt engineering is the design of prompts that steer LLM behavior and shape outputs. It involves structuring instructions, supplying context, and applying techniques that improve response quality, reliability, and consistency.

## Description

Prompts are the primary interface to LLMs. System and user messages carry role, task, constraints, and examples. Template-based prompts with variable substitution support reuse; few-shot examples in the prompt guide format and style without fine-tuning. Role-setting (e.g. “You are an expert at…”) and structured, numbered instructions are common. Business rules and output format can be specified in the prompt text.

**Key concepts**

- Template-based prompts with variable substitution
- Few-shot learning (examples included in the prompt)
- Role-setting (e.g. “You are an expert at…”)
- Structured instructions with numbered steps
- Constraint injection (business rules in prompts)
- Output format specification

## Translation Example

OpenAI chat completion with a translation role and a templated user prompt. Optional few-shot user/assistant pairs can be appended before the phrase to translate.

```python
from openai import OpenAI

client = OpenAI()

system_prompt = (
    "You are a professional translator. "
    "Translate the user's phrase into the requested language. "
    "Use natural, idiomatic phrasing. Reply with only the translation, no explanation."
)
user_template = "Translate to {target_language}:\n{phrase}"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_template.format(target_language="French", phrase="Hello, how are you?")},
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=150,
)

print(response.choices[0].message.content)
```

System and user messages are sent together; the model uses both to produce the translation. For few-shot behavior, prepend user/assistant pairs (e.g. past translation examples) before the final user message.
