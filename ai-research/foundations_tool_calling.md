# Foundations: Tool Calling

*(LLM Overview Project)*

---

## Introduction

Tool calling extends the transformer’s core capability — next-token prediction — into the external world.

A modern LLM does not execute code, access APIs, or retrieve real-time data internally. Instead, it learns to emit structured tokens that represent tool invocations. External systems detect those structured outputs, execute the requested operation, and return results to the model as additional context. The model then continues autoregressive generation conditioned on this new information.

Tool use is therefore not a separate reasoning engine or symbolic planner. It is:

> Structured next-token prediction + external execution + context reintegration.

At scale, transformers learn schema adherence, argument binding, and multi-step continuation patterns that make tool use appear deliberate and goal-directed.

---

## Suggested Reading (Brief)

For foundational background relevant to tool calling:

* Vaswani et al., *Attention Is All You Need* (2017) — transformer architecture
* OpenAI Function Calling documentation (2023–2025) — structured output schemas
* Anthropic Tool Use documentation — structured tool integration patterns
* Wei et al., *Chain-of-Thought Prompting* (2022) — multi-step reasoning patterns
* Nye et al., *Show Your Work* (2021) — reasoning traces as latent computation externalization

These provide context for how autoregressive transformers can produce structured reasoning and structured outputs.

---

# Core Concept

Tool calling is not symbolic planning.

It is:

1. Next-token prediction
2. Structured schema emission
3. External runtime execution
4. Context reintegration
5. Continued autoregressive generation

Everything reduces to probabilistic token continuation.

---

# 1. Decision Emerges from Next-Token Prediction

The model does not explicitly decide to call a tool.

Given the context and prompt, it predicts the next most probable token.

If training patterns associate certain queries with tool invocation, the highest-probability continuation becomes a structured call.

### Example: Weather Query

**User Prompt**

> "What’s the weather in San Francisco right now?"

Instead of predicting natural language:

> "The weather in San Francisco is..."

The model predicts structured tokens:

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "San Francisco"
  }
}
```

There is no internal execution.
Just token probability.

---

# 2. Structured Emission

The system provides tool schemas in the context:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"}
    },
    "required": ["city"]
  }
}
```

The model:

* Attends to "San Francisco"
* Routes that feature into the `"city"` field
* Emits syntactically valid JSON

This works because:

* Transformers track long-range dependencies
* Attention allows argument binding
* Schema tokens constrain valid continuation space

Structured output is learned pattern completion.

---

# 3. Runtime Execution (External System)

After the model emits:

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "San Francisco"
  }
}
```

The runtime:

1. Detects structured tool output
2. Executes `get_weather("San Francisco")`
3. Returns result:

```json
{
  "temperature": "58°F",
  "conditions": "Partly cloudy",
  "wind": "8 mph NW"
}
```

The model does not execute code.

Execution is always external.

---

# 4. Result Reintegration

The tool result is appended to the context window.

Now the model continues next-token prediction conditioned on:

* The original question
* Its tool call
* The tool output

Continuation:

> "It’s currently 58°F in San Francisco with partly cloudy skies and winds at 8 mph from the northwest."

Again: just autoregressive continuation.

---

# Multi-Step Example

### Sales Tax Calculation

**User Prompt**

> "How much would I pay in sales tax if I buy a $2,499 laptop in San Jose?"

The model predicts:

```json
{
  "name": "get_sales_tax_rate",
  "arguments": {
    "city": "San Jose"
  }
}
```

Tool returns:

```json
{
  "rate": 0.0925
}
```

Model emits second tool call:

```json
{
  "name": "calculate_total",
  "arguments": {
    "price": 2499,
    "tax_rate": 0.0925
  }
}
```

Tool returns:

```json
{
  "total": 2729.66
}
```

Model integrates:

> "With a 9.25% sales tax in San Jose, your total would be $2,729.66."

This is iterative structured continuation.

No symbolic reasoning layer exists outside autoregression.

---

# Code Execution Example

### Fibonacci

**User Prompt**

> "Write Python code to compute the first 10 Fibonacci numbers and show the result."

Model emits:

```json
{
  "name": "execute_python",
  "arguments": {
    "code": "def fib(n):\n    seq=[0,1]\n    for _ in range(n-2):\n        seq.append(seq[-1]+seq[-2])\n    return seq\nfib(10)"
  }
}
```

Runtime executes.

Returns:

```json
{
  "result": [0,1,1,2,3,5,8,13,21,34]
}
```

Model continues:

> "Here are the first 10 Fibonacci numbers: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]."

Again:
Structured tokens → external execution → context continuation.

---

# Fresh Information Example

### News Lookup

**User Prompt**

> "Who won the most recent Super Bowl?"

Model emits:

```json
{
  "name": "search_news",
  "arguments": {
    "query": "most recent Super Bowl winner"
  }
}
```

Tool returns:

```json
{
  "headline": "Kansas City Chiefs win Super Bowl LX",
  "date": "Feb 2026"
}
```

Model integrates:

> "The Kansas City Chiefs won the most recent Super Bowl (February 2026)."

Tool calling enables grounding beyond frozen training data.

---

# Mechanistic Framing

Under a transformer interpretability lens, tool calling involves:

* Feature detection ("requires external capability")
* Schema-conditioned continuation
* Attention-based argument routing
* Residual stream state accumulation
* Multi-step autoregressive planning via context growth

Tool use is:

> A learned continuation circuit operating over structured schemas.

It is not:

* A symbolic planner
* A separate reasoning module
* An internal execution engine

---

# Canonical Summary

Tool calling =

1. Next-token prediction
2. Structured schema emission
3. External execution
4. Context reintegration
5. Continued autoregressive generation

Everything reduces to probabilistic continuation over tokens.

---
