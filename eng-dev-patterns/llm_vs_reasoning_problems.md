# LLM vs. Reasoning Models: Problem Selection Guide

This guide describes problem types that highlight the differences between chat/conversational LLMs (e.g., GPT-4o, Claude) and reasoning models (e.g., o1, o4-mini). Choosing the right model for your problem type improves accuracy, reduces cost, and optimizes latency.

## Why Model Choice Matters

**Chat/Conversational LLMs** answer quickly and directly. They rely on pattern matching, implicit reasoning, and retrieval over vast training data. They excel at tasks where the answer is well-represented in their training distribution: summaries, lookups, straightforward Q&A, and creative generation.

**Reasoning Models** are trained to "think step by step" before answering. They perform explicit chain-of-thought, verify intermediate conclusions, and handle multi-step logical or mathematical deduction more reliably. They use more tokens and take longer but reduce subtle logical errors on harder problems.

## Problem Types Suited to Chat/LLMs

### Lookup and retrieval

*Example: "What's the current weather in San Francisco?"*

- **Why LLM**: The model or its tool calls fetch data; the task is to interpret and present it. No deep deduction. Low latency and cost are beneficial.
- **Use case**: Weather, definitions, factual lookup, content summarization.

### Straightforward Q&A

*Example: "What is the capital of France?"*

- **Why LLM**: Single-hop retrieval. The answer is well-represented in training data. Reasoning overhead adds no value.
- **Use case**: Trivia, definitions, simple explanations.

### Creative generation

*Example: "Write a short poem about autumn."*

- **Why LLM**: Generation benefits from fluency and diversity, not stepwise proof. Reasoning models may overthink and slow output.
- **Use case**: Drafting, brainstorming, storytelling.

### Tool orchestration without deep logic

*Example: "Fetch this URL and summarize the main points."*

- **Why LLM**: The hard part is calling the tool and condensing content. No multi-step logic or verification needed.
- **Use case**: RAG-style retrieval, API orchestration, content condensation.

---

## Problem Types Suited to Reasoning Models

### 1. Classic trick questions

*Example: "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?"*

- **Why reasoning model**: The intuitive answer (10 cents) is wrong. Correct answer: 5 cents. Reasoning models set up the equation (ball = x, bat = x + 1, 2x + 1 = 1.10) and solve systematically.
- **Why not LLM**: Chat models often produce the intuitive answer without verifying. They under-check their own logic.
- **Use case**: Any problem where the obvious answer is a trap.

### 2. Multi-step arithmetic and word problems

*Example: "If 7 workers build a wall in 12 days, how long would 4 workers take? Show each step."*

- **Why reasoning model**: Requires rate (work per day), then total work, then new rate, then new time. Each step must be correct; one error propagates.
- **Why not LLM**: Chat models may mix inverse vs. direct proportion, skip steps, or produce inconsistent numbers.
- **Use case**: Rate problems, proportions, unit conversions, budget calculations.

### 3. Logic puzzles and constraint satisfaction

*Example: "Alice, Bob, and Carol finished 1st, 2nd, and 3rd. Alice wasn't 1st. Bob wasn't 3rd. Carol beat Bob. Who was 1st, 2nd, 3rd? Explain your reasoning."*

- **Why reasoning model**: Needs systematic elimination of possibilities, tracking constraints, and checking consistency.
- **Why not LLM**: May give contradictory assignments (e.g., Bob 2nd and Carol beating Bob but Carol 3rd) or jump to an answer without valid deduction.
- **Use case**: Scheduling, ranking, constraint-based planning, puzzle games.

### 4. Problems with non-obvious "edge" steps

*Example: "A snail climbs 3 feet by day but slides back 2 feet each night. How many days to climb a 10-foot wall?"*

- **Why reasoning model**: Must recognize that on the final day the snail reaches the top and does not slide back. A naive day-by-day loop gives the wrong count if the last day is not handled specially.
- **Why not LLM**: Often assumes the slide happens every night and miscounts.
- **Use case**: Processes with final-step exceptions, boundary conditions, off-by-one situations.

### 5. Logical validity and argument analysis

*Example: "'All mammals are warm-blooded. Snakes are mammals. Therefore snakes are warm-blooded.' Is this argument logically valid? Why or why not?"*

- **Why reasoning model**: Can separate validity (does the conclusion follow from the premises?) from factual truth (are snakes mammals?). Validity depends only on logical form.
- **Why not LLM**: Often conflates validity with fact. May say "invalid because snakes aren't mammals" instead of "valid form, but a false premise."
- **Use case**: Legal/formal reasoning, standardized test prep, debate structure, fallacy detection.

### 6. Planning under constraints

*Example: "You have 24 hours in a city. Four museums open 10am–6pm, are 30 minutes apart, and each needs 2 hours. Can you visit all four? Plan the order."*

- **Why reasoning model**: Must sequence visits, respect opening hours, account for travel, and verify feasibility step by step.
- **Why not LLM**: May produce a plausible-looking schedule that violates constraints or double-counts time.
- **Use case**: Scheduling, resource allocation, route planning.

### 7. Code debugging and execution tracing

*Example: "This code has a bug. Find it and explain why it fails."*

- **Why reasoning model**: Can trace execution step by step, track variable values, and identify where behavior diverges from intent.
- **Why not LLM**: May guess or pattern-match instead of systematically tracing.
- **Use case**: Code review, debugging assistance, programming education.

---

## Head-to-Head: Expected Wins and Losses

These examples are useful for benchmarking. Run them against both a chat model (e.g., GPT-4o-mini) and a reasoning model (e.g., o3-mini) and compare results.

### LLM typically fails, reasoning typically succeeds

**Bat and ball (trick question)**  
> "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Only give the number of cents."

- **LLM**: Often returns 10 (the intuitive wrong answer).
- **Reasoning**: Sets up the equation (2x + 1 = 1.10), solves, and returns 5.
- **Why**: The correct answer requires checking the logic; pattern-matching the "obvious" answer fails.

### Reasoning typically worse, LLM typically better

**Quick creative generation**  
> "Write a 3-line haiku about coffee."

- **LLM**: Produces a haiku quickly and fluently.
- **Reasoning**: May over-reason, delay output, and produce something stiff or over-structured.
- **Why**: Haiku benefit from fluency and immediacy, not stepwise logic.

**Simple lookup / completion**  
> "Complete this: 'The capital of France is ___'."

- **LLM**: Returns "Paris" immediately.
- **Reasoning**: May spend tokens "reasoning" toward a trivial answer, adding latency and cost.
- **Why**: Single retrieval; reasoning adds no value.

### Summary

| Task                    | LLM                     | Reasoning              |
|-------------------------|-------------------------|------------------------|
| Bat and ball (5 cents)  | Often wrong             | Usually correct        |
| Haiku about coffee      | Fast, natural           | Often slow or stiff    |
| "Capital of France is"  | Immediate               | Slower, more tokens    |

---

## Quick Selection Guidelines

| Task characteristic              | Prefer LLM | Prefer reasoning model |
|---------------------------------|------------|-------------------------|
| Single-step lookup or summary    | ✓          |                         |
| Creative or fluent generation    | ✓          |                         |
| Fast, low-cost responses         | ✓          |                         |
| Multi-step logic or math         |            | ✓                       |
| Constraint satisfaction          |            | ✓                       |
| Trick questions, edge cases      |            | ✓                       |
| Verifying logical validity       |            | ✓                       |
| Planning and scheduling          |            | ✓                       |

---

## Try Both

When in doubt, run the same prompt against both a chat model and a reasoning model. Compare:

- **Accuracy** on your specific examples
- **Consistency** across multiple runs
- **Latency and cost** for your use case

Use the examples in this guide as a starting point for your own benchmarks.

---

## Related patterns

- **Understanding Models**: Model categories (chat vs. reasoning vs. fast/cheap) and when to use each; see [understanding_models.md](./understanding_models.md).
- **Learning progression**: Pattern 9 covers Chain-of-Thought / Multi-Step Reasoning; see [learning_progression.md](./learning_progression.md).
- **Prompt Engineering**: For chat models, techniques like "think step by step" can improve reasoning; see [prompt_engineering.md](./prompt_engineering.md).

---

*Last updated: January 2026*
