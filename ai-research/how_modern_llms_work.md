
# How Modern LLMs Work

Modern large language models (LLMs) are massively scaled neural networks trained to predict the next token in a sequence. From that simple objective, surprisingly powerful behaviors emerge.

---

## 1. The Core Idea: Next-Token Prediction

At heart, an LLM learns:

$$
P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

Given prior tokens, predict the next one.

During training:

* Text is broken into subword tokens.
* The model sees trillions of tokens from books, websites, code, math explanations, and dialogue.
* It learns to minimize prediction error on the next token.

There are no explicit reasoning rules, symbolic logic modules, or built-in knowledge bases.

It is trained purely through self-supervised learning.

---

## 2. The Architecture: Transformers

Modern LLMs use the Transformer architecture, built from:

* **Self-attention layers** — allowing each token to reference other tokens.
* **Feedforward (MLP) layers** — transforming representations.
* **Residual connections and normalization** — stabilizing deep computation.

Self-attention lets the model dynamically route information across the entire context window.

Stack many layers (often 40–100+), and the network becomes extremely expressive.

---

## 3. The Big Surprise: Scale Changes Behavior

As models scaled in:

* Parameter count
* Training data size
* Compute

Researchers observed something unexpected:

Capabilities emerged that were not explicitly programmed:

* Few-shot learning
* Multi-step reasoning
* Code generation
* Tool usage
* Structured problem solving

The training objective never changed — only scale did.

At large scale, the internal representations become rich enough to support distributed algorithmic behavior.

---

## 4. What the Model Actually Learns

Because it is trained on vast amounts of structured human output, the model learns:

* Grammar and syntax
* Semantic associations
* Discourse structure
* Procedural patterns (how explanations and solutions unfold)
* Common reasoning templates

It does not learn:

* Grounded world experience
* Formal logical guarantees
* A persistent internal world model

It learns statistical structure in text at massive scale.

---

## 5. Why Reasoning Appears to Work

LLMs generate reasoning by producing coherent multi-step token sequences.

When prompted to “show your work,” performance improves because:

* Intermediate steps become part of the context.
* The model can attend back to earlier reasoning steps.
* Text acts as a scratchpad.

Reasoning is implemented as structured token generation supported by distributed internal representations.

It is not symbolic logic execution, but it can approximate reasoning behavior surprisingly well.

---

## 6. Code Generation

Code is especially well-suited to LLMs because:

* It has strong syntax rules.
* It is highly repetitive.
* It appears frequently in training data.
* Many examples include explanations and corrections.

The model learns patterns of:

* Function structure
* Variable usage
* API calls
* Debugging fixes

At scale, it can maintain consistency across longer code spans.

But it does not internally execute the code — it predicts plausible continuations.

---

## 7. Tool Calling

Modern LLM systems can call tools (search, calculator, code execution).

Mechanically:

1. The model emits structured tokens indicating a tool call.
2. The runtime executes the tool.
3. The result is appended to the context.
4. The model continues generation conditioned on that result.

The model does not internally perform the action — it learns patterns of delegation.

Tool use is learned sequence behavior shaped during fine-tuning.

---

## 8. Mechanistic Interpretability

Research has shown that:

* Some attention heads implement identifiable subroutines (e.g., copying patterns, matching parentheses).
* Features are stored in distributed superposition.
* Circuits span multiple layers to implement structured behaviors.

This suggests transformers internally develop algorithm-like circuits without being explicitly programmed to do so.

---

# 9. This Inference Loop

The most important insight is not any single capability — it is how they combine at inference time.

A modern LLM-based system operates in an iterative loop:

1. Read context (prompt + prior outputs + tool results)
2. Generate next structured step (text, plan, code, or tool call)
3. If a tool call is emitted:

   * Execute the tool
   * Append result to context
4. Continue generation
5. Repeat until task completion

Everything is still next-token prediction — but the context grows richer at each step.

This loop creates an **approximation of reasoning and planning**.

---

## Example 1: “Build a Library of Popular Recipes”

Suppose you ask:

> Build a library of popular recipes, categorized by cuisine, with ingredient lists and preparation steps.

The system might proceed like this:

### Step 1: Decompose the Task

The model generates:

* Categories (Italian, Mexican, Indian, Japanese, etc.)
* A schema (title, ingredients, steps, prep time)
* A plan to populate each category

This decomposition reflects patterns learned from cookbooks, blogs, and structured recipe sites.

---

### Step 2: Generate Structured Content

For each cuisine, it produces:

* Recipe names
* Ingredient lists
* Step-by-step instructions

It maintains consistent formatting because:

* Recipe structure is highly regular.
* It has seen many examples of recipe formatting.

---

### Step 3: Use Tools (Optional)

If tools are available:

* It could search for “most popular Italian dishes.”
* It could retrieve up-to-date rankings.
* It could store recipes in a database.
* It could validate nutritional values using a calculation tool.

Each tool result becomes new context.

---

### Step 4: Maintain Global Structure

Across many recipes, the model:

* Keeps consistent schema
* Avoids duplicating categories
* Preserves formatting
* Maintains coherence across the full “library”

Not because it has a database schema internally — but because:

* The expanding context anchors structure.
* Attention retrieves earlier formatting decisions.
* The model has learned patterns of structured content generation.

This already resembles planning.

---

## Example 2: Building a Large Software Project

Now imagine:

> Build a scalable REST backend for a content processing system with job queues, Redis, observability, and CI/CD.

The system might:

### Step 1: Decompose the Problem

Generate:

* Architecture overview
* Service separation (API, worker, Redis)
* Deployment plan (Docker, Kubernetes)
* Logging strategy

---

### Step 2: Generate Code

Write:

* API skeleton
* Queue integration
* Helm charts
* Dockerfiles
* CI/CD config

---

### Step 3: Use Tools

* Run code.
* Receive error.
* Fix dependency.
* Re-run tests.

Each tool output becomes new context.

---

### Step 4: Maintain Global Coherence

Across many iterations, it:

* Keeps service names consistent
* Maintains configuration alignment
* Preserves architectural decisions

Not through symbolic understanding — but through context anchoring and distributed representation.

---

## Why This Feels Like Real Reasoning

The loop:

* Decomposes tasks
* Produces intermediate artifacts
* Uses external verification
* Revises based on feedback
* Maintains global consistency

It resembles engineering workflows.

Internally, however, it is:

> Iterative next-token prediction over an expanding structured context, augmented by tool execution.

Reasoning is distributed across:

* Transformer latent state
* Text as working memory
* External computation

---

# 10. Clean Mental Model

A modern LLM-based system is:

> A massively scaled next-token predictor embedded inside an iterative inference loop, where text acts as working memory and tools provide external grounding.

Or more intuitively:

> A high-dimensional sequence model that, when combined with structured prompting and tool feedback, can approximate reasoning and large-scale problem solving.

---
