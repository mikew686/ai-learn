# LLM vs. Reasoning Models: Problem Selection

## Overview

Chat/conversational LLMs (e.g. GPT-4o, Claude) and reasoning models (e.g. o1, o3-mini) suit different problem types. Chat models respond quickly and are strong at lookup, summarization, and creative generation. Reasoning models perform explicit step-by-step reasoning and tend to do better on multi-step logic, math, and trick questions, at higher token cost and latency. This document summarizes which problem types favor which class.

## Description

**Chat/conversational LLMs** answer directly and quickly. They rely on pattern matching and implicit reasoning over training data. They excel at: lookup and retrieval, straightforward Q&A, creative generation, and tool orchestration without deep deduction.

**Reasoning models** are trained to reason step-by-step before answering. They verify intermediate steps and handle multi-step logic and math more reliably. They tend to do better on: classic trick questions (e.g. bat-and-ball), multi-step arithmetic and word problems, logic puzzles and constraint satisfaction, problems with non-obvious edge steps (e.g. snail climbing a wall), logical validity and argument analysis, planning under constraints, and code debugging with execution tracing. They use more tokens and are slower; for simple lookup or creative tasks they often add cost and latency without benefit.

**Key concepts**

- **Chat/LLM**: Fast, low-cost; good for lookup, summarization, Q&A, creative generation, tool use without deep logic.
- **Reasoning**: Step-by-step reasoning; good for multi-step logic, math, trick questions, constraints, planning, verification.
- **Task fit**: Single-step or creative → chat; multi-step logic, verification, or edge cases → reasoning.
- **Tradeoffs**: Reasoning improves accuracy on hard logic/math at the cost of latency and tokens.

## Basic Example

Same prompt sent to a chat model and a reasoning model. The problem has a non-obvious edge case (on the final day the snail reaches the top and does not slide back), so a naive “net 1 foot per day → 10 days” is wrong. The reasoning model typically traces day-by-day and catches the last day; chat models often give 10.

```python
from openai import OpenAI

client = OpenAI()
prompt = (
    "A snail climbs 3 feet up a wall each day but slides back 2 feet each night. "
    "How many days does it take to reach the top of a 10-foot wall? "
    "Explain your reasoning step by step, then give the final number of days."
)

# Chat model: often assumes the slide happens every night and answers 10 (wrong); may not trace day-by-day
chat = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=400,
)
print("Chat:", chat.choices[0].message.content.strip())

# Reasoning model: typically traces each day, notices on day 8 the snail reaches 10 and does not slide, answers 8
reasoning = client.chat.completions.create(
    model="o3-mini",  # or current reasoning model name
    messages=[{"role": "user", "content": prompt}],
    max_tokens=600,
)
print("Reasoning:", reasoning.choices[0].message.content.strip())
```

Correct answer: 8 days (after night 7 the snail is at 7 ft; day 8 it climbs to 10 and reaches the top, so it does not slide back). The “explain your reasoning” instruction plus this edge case forces explicit step-by-step tracing; reasoning models handle it more reliably than chat models. For simple lookup or creative tasks, the chat model is usually faster and sufficient.
