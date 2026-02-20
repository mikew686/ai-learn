# Foundations: Reasoning

## Introduction

Reasoning in large language models is not implemented as a symbolic logic engine or an explicit search algorithm. Instead, it emerges from multi-step autoregressive token generation operating over transformer architectures trained at scale.

Although LLMs are trained only to predict the next token, they exhibit structured behaviors that resemble logical deduction, arithmetic computation, planning, and abstraction. These behaviors arise from distributed feature composition in the residual stream, attention-based routing of information, and scale-enabled emergence of reusable algorithmic circuits.

A central insight in modern LLM research is that prompting models to “show their work” dramatically improves performance. This reveals that reasoning is not a separate system inside the model, but a trajectory through token space that can be shaped and stabilized through explicit intermediate steps.

### Suggested Reading

* **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022)
* **Emergent Abilities of Large Language Models** (Wei et al., 2022)
* **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
* Anthropic and OpenAI mechanistic interpretability research (induction heads, circuits, superposition)
* Transformer Circuits research threads (Olah et al.)

These works collectively explain how reasoning emerges from scale, how prompting alters reasoning trajectories, and how internal transformer circuits implement algorithm-like behavior.

---

## 1. What “Reasoning” Is in an LLM

Reasoning is not a symbolic module.

It is:

> Multi-step autoregressive token generation in which intermediate tokens act as externalized working memory, while the residual stream maintains distributed internal state across layers.

The model generates text step-by-step, and each step conditions future steps. Complex reasoning is therefore an extended structured trajectory in token space.

---

## 2. Autoregressive Computation

At each token position:

$$
P(t_i \mid t_{<i})
$$

The model predicts the most likely continuation.

If it attempts to jump directly from problem to final answer, it must compress all intermediate computation into one prediction.

Stepwise reasoning instead distributes the computation across many tokens.

This converts:

* One high-entropy prediction
  into
* Multiple lower-entropy local predictions

This decomposition makes reasoning more stable.

---

## Diagram — Reasoning Flow in an LLM

```mermaid
flowchart LR

A[Input Prompt + Prior Tokens] --> B[Embedding + Positional Encoding]
B --> C[Transformer Layers]

subgraph Transformer Block
    D[Self-Attention\n(Variable Routing)]
    E[MLP\n(Feature Transformation)]
    D --> E
    E --> F[Residual Stream Update]
end

C --> D
F --> G[Updated Residual State]
G --> H[Next Token Prediction]
H --> I[Token Appended to Context]
I --> C
```

### How to Read the Diagram

1. Tokens enter the model and are embedded.
2. Transformer layers apply attention and MLP transformations.
3. The residual stream accumulates structured state.
4. A next token is predicted.
5. That token is appended to context.
6. The full stack runs again.

Each new token re-enters the entire transformer stack, effectively providing another full forward-pass refinement of the reasoning process.

---

## 3. Chain-of-Thought as Externalized Working Memory

When a model writes intermediate steps:

1. It stores partial results in the context window
2. Those tokens become re-readable anchors
3. Attention heads can bind to them
4. Later steps condition on explicit prior state

Without explicit reasoning tokens, intermediate values must persist only as distributed activation patterns in the residual stream — which is more fragile.

Thus:

> Chain-of-thought externalizes computation into stable symbolic scaffolding.

---

## 4. The Residual Stream as Distributed State

Internally:

* Each transformer layer reads from the residual stream
* Applies attention and MLP transformations
* Writes updates back into the residual stream

The residual stream contains:

* Variable bindings
* Hypotheses
* Logical relationships
* Numerical structure
* Partial conclusions

Reasoning corresponds to progressive refinement of this distributed state across layers and across tokens.

---

## 5. Attention as Variable Routing

Attention enables:

* Tracking entities across context
* Binding values to names
* Comparing quantities
* Aligning premises with conclusions

Example:

“All mammals are warm-blooded. Dolphins are mammals.”

Attention heads can:

* Link “dolphins” to “mammals”
* Link “mammals” to “warm-blooded”
* Route relational structure forward

The conclusion emerges as the most coherent continuation.

---

## 6. Emergent Algorithmic Circuits

At scale, models develop reusable internal circuits, including:

* Induction heads (pattern continuation)
* Comparison mechanisms
* Arithmetic-like behaviors
* Structured decomposition tendencies

These circuits were not explicitly programmed.

They emerge because:

* Large-scale next-token prediction requires modeling latent structure
* Sufficient depth and width enable stable feature reuse
* Scaling laws show smooth performance gains

Researchers were surprised that models trained only for next-token prediction could perform multi-step reasoning so effectively.

---

## 7. Why “Show Your Work” Improves Performance

Prompting models to explain their reasoning improves performance because:

1. It externalizes intermediate state
2. It reduces entropy at each token step
3. It allows attention to re-bind explicit variables
4. It stabilizes numerical or logical structure
5. It effectively increases available compute

Each additional token re-enters the full transformer stack, providing another forward-pass refinement over prior reasoning.

Thus:

> Longer reasoning traces simulate iterative computation.

---

## 8. Concrete Examples

### Example 1 — Arithmetic

Problem:
If a train travels 60 mph for 2.5 hours, how far does it go?

Stepwise reasoning:

1. Distance = speed × time
2. 60 × 2.5 = 150
3. Answer: 150 miles

The intermediate multiplication stabilizes the result before conclusion.

---

### Example 2 — Logical Deduction

Problem:
All mammals are warm-blooded. Dolphins are mammals. Are dolphins warm-blooded?

Intermediate reasoning:

1. Dolphins ∈ mammals
2. Mammals ⊆ warm-blooded
3. Therefore dolphins ∈ warm-blooded

Attention aligns relational structure; the residual stream accumulates set inclusion logic.

---

### Example 3 — Planning

Problem:
Plan a migration from Flask to FastAPI.

Reasoning involves:

* Decomposition into tasks
* Dependency ordering
* Risk analysis
* Stepwise refinement

This is structured multi-token planning, not symbolic graph search.

---

## 9. Limitations

LLM reasoning has structural constraints:

* No persistent memory beyond context window
* Drift in long reasoning chains
* Susceptibility to subtle inconsistencies
* No guaranteed logical correctness
* No built-in verification

It approximates reasoning through high-dimensional statistical continuation rather than formal proof search.

---

## 10. Compact Definition

> Reasoning in LLMs is multi-step autoregressive token generation in which intermediate tokens act as working memory, the residual stream accumulates distributed state, attention routes variables across context, and scale enables the emergence of reusable algorithmic circuits — producing behavior that approximates structured reasoning without explicit symbolic machinery.

---
