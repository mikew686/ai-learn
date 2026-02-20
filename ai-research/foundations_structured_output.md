# Foundations: Structured Output

---

## Introduction

Structured output is the practice of making an LLM emit **machine-parseable, schema-conforming data** instead of free-form prose.

Most real-world LLM systems are not “chatbots.” They are:

* Workflow engines
* API orchestrators
* Data extractors
* UI state generators
* Tool routers
* Policy validators

In all of these cases, reliability requires **structure**.

Structured output turns probabilistic language generation into something closer to a typed interface.

---

## Suggested Reading

* Vendor documentation on structured outputs / function calling
* JSON Schema documentation
* Articles on constrained decoding and grammar-based decoding
* Blog posts on LLM validation and repair loops
* Papers on inference-time verification and self-correction

---

# 1. Core Idea

Structured output =

> **Format + Constraints + Validation**

Where:

* **Format** → JSON, YAML, SQL, DSL, etc.
* **Constraints** → JSON Schema, grammar, or tool signature
* **Validation** → Parsing + schema checking + business-rule checking
* **Repair loop** → Re-ask model when invalid

Without validation, structured output is only a polite suggestion.

With validation + repair, it becomes reliable infrastructure.

---

# 2. Relationship to Tool Calling

Tool calling is a special case of structured output.

Tool calling emits:

```json
{
  "tool_name": "...",
  "arguments": { ... }
}
```

Structured output generalizes this idea:

Instead of calling a tool, the output might:

* Create a workflow plan
* Extract structured entities
* Produce UI configuration
* Generate SQL
* Emit a JSON API payload

So:

> Tool calling = structured output + execution semantics

---

# 3. What Goes Into Context

A structured-output system typically injects:

1. System instructions (strict formatting rules)
2. Schema definition
3. Task description
4. Source data
5. (On retry) validation errors

The model never “knows” about JSON in an abstract sense.

It sees:

* Tokens describing the schema
* Tokens describing constraints
* Tokens describing instructions

Its next-token distribution shifts accordingly.

---

# 4. Detailed Example: Extraction to JSON (Full Context Walkthrough)

We will extract contact information from an email using strict JSON schema validation.

---

## Step 1 — Context Assembly

### (A) System Message

```text
You are a structured data extraction engine.

You must output ONLY valid JSON.
Do not include explanations.
Do not include markdown.
If a field cannot be determined, use null.
```

This sharply reduces probability of commentary tokens.

---

### (B) Schema Injected by Runtime

```text
The output must conform to this JSON schema:

{
  "type": "object",
  "required": ["name", "email", "phone", "company"],
  "properties": {
    "name": { "type": ["string", "null"] },
    "email": { "type": ["string", "null"] },
    "phone": { "type": ["string", "null"] },
    "company": { "type": ["string", "null"] }
  },
  "additionalProperties": false
}
```

This:

* Defines required keys
* Restricts types
* Forbids extra fields

It narrows the output space dramatically.

---

### (C) User Task + Source Text

```text
Extract the contact information from the following email.

Email:

Hi — this is Sarah Chen from Redwood Analytics.
You can reach me at sarah.chen@redwood.io or (415) 555-0198.
Looking forward to connecting.
```

---

## Step 2 — What the Model Does Internally

Mechanistically:

* Embeds entire context.
* Attention connects:

  * "Sarah Chen" → name
  * Email pattern → email field
  * Phone pattern → phone field
  * "from Redwood Analytics" → company
* Schema tokens bias toward emitting required keys.
* "additionalProperties": false lowers probability of inventing fields.

The model begins emitting JSON tokens.

---

## Step 3 — First Model Output (Valid Case)

```json
{
  "name": "Sarah Chen",
  "email": "sarah.chen@redwood.io",
  "phone": "(415) 555-0198",
  "company": "Redwood Analytics"
}
```

Runtime:

* Parse → success
* Schema validate → success
* Object accepted

---

## Step 3b — First Model Output (Invalid Case)

Suppose instead it emits:

```json
{
  "name": "Sarah Chen",
  "email": "sarah.chen@redwood.io",
  "phone": "(415) 555-0198",
  "company": "Redwood Analytics",
  "title": "Unknown"
}
```

Violation:

```json
"additionalProperties": false
```

---

## Step 4 — Runtime Validation

Validator detects:

```
ValidationError:
Additional property "title" not allowed.
```

---

## Step 5 — What Gets Added Back to Context

The runtime appends:

### System Reminder

```
Your previous output did not conform to the schema.
You must return ONLY corrected JSON.
Do not add extra fields.
```

### Validation Error

```
Validation error:
Additional property "title" is not allowed.
Remove this property and return valid JSON.
```

### Previous Output (optional but common)

```
Previous output:

{
  "name": "Sarah Chen",
  "email": "sarah.chen@redwood.io",
  "phone": "(415) 555-0198",
  "company": "Redwood Analytics",
  "title": "Unknown"
}
```

---

## Step 6 — Repair Response

Model now emits:

```json
{
  "name": "Sarah Chen",
  "email": "sarah.chen@redwood.io",
  "phone": "(415) 555-0198",
  "company": "Redwood Analytics"
}
```

Validation passes.

---

# 5. What Changed Mechanistically?

When the model sees:

```
Additional property "title" is not allowed.
```

That token sequence:

* Decreases probability of emitting `"title"`
* Reinforces `"additionalProperties": false`
* Increases probability of deleting that field

This is inference-time probability shaping via explicit constraint feedback.

---

# 6. Constrained Decoding Variant

Instead of validating after generation, decoding can be constrained.

At each token:

* Runtime computes which tokens preserve valid JSON.
* Invalid tokens are masked out.
* Model cannot emit syntactically illegal sequences.

This guarantees syntactic validity and often schema-level validity.

Semantic correctness still requires validation.

---

# 7. Syntax vs Semantics

Even valid JSON may be wrong:

* Phone number malformed
* Email domain invalid
* Company name hallucinated
* Cross-field inconsistency

Thus two layers are required:

1. Schema validation (shape + types)
2. Business validation (real-world constraints)

---

# 8. Failure Modes

Common issues:

* Extra commentary
* Missing required fields
* Incorrect types
* Hallucinated values
* Truncated JSON
* Regex mismatches
* Cross-field logic errors

Structured output systems should log:

* Raw output
* Parsed object
* Validation errors
* Retry count
* Schema version

---

# 9. Structured Output in Agent Loops

Structured output becomes critical in agent workflows:

1. Model emits structured plan object.
2. Runtime executes tools.
3. Tool results added to context.
4. Model emits next structured state.

This keeps the agent deterministic and inspectable.

Without structured output, agent systems drift into narrative instability.

---

# 10. Practical Best Practices

* Prefer JSON + schema.
* Keep schemas shallow and typed.
* Use enums aggressively.
* Validate strictly.
* Implement retry loops.
* Separate syntax validation from business validation.
* Treat model output as untrusted input.
* Log schema versions and prompt versions.

---

# 11. Compact Definition

Structured output is:

> The process of constraining LLM generation to emit parseable, schema-valid data and coupling that with validation and repair loops so downstream systems can reliably execute, store, or render the result.

Mechanistically, it is:

* Autoregressive token prediction
* Under constrained probability space
* With explicit validation feedback
* Iteratively converging toward a valid object

---
