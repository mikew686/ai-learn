# Foundations: LLM Transformers to GPTs

*(Building on Foundations: From Neural Networks to LLMs)*

We now assume familiarity with:

* Feed-forward neural networks
* Backpropagation and gradient descent
* Cross-entropy loss
* Embeddings
* Basic attention concepts

This section provides a technical and mechanistic explanation of how modern LLMs (GPT-style models) work — and how scaling transforms transformer architectures into systems that exhibit reasoning, tool use, abstraction, and code generation.

For deeper technical detail, the following readings are especially valuable:

### Core Architecture

* **Attention Is All You Need** (Vaswani et al., 2017) – Original transformer paper
* **Language Models are Unsupervised Multitask Learners** (GPT-2 paper)
* **GPT-3 (Brown et al., 2020)** – Emergent in-context learning at scale

### Scaling & Emergence

* **Kaplan et al., 2020 – Scaling Laws for Neural Language Models**
* **Hoffmann et al., 2022 – Chinchilla Scaling Laws**
* Anthropic papers on emergent behavior and phase transitions

### Mechanistic Interpretability

* Anthropic’s work on **Induction Heads**
* Anthropic’s research on **Superposition and Polysemantic Neurons**
* OpenAI interpretability research on feature directions

### Alignment & Post-Training

* **InstructGPT (Ouyang et al., 2022)**
* RLHF and preference optimization literature

Together, these works provide the architectural, scaling, and interpretability foundations necessary to understand how GPT-style models operate at both the algorithmic and systems level.

---

# 1. From Deep Networks to Transformers

Earlier neural architectures:

* CNNs: spatial feature extraction
* RNNs/LSTMs: sequential memory via recurrence

Limitations:

* Sequential bottlenecks
* Vanishing gradients
* Poor long-range dependency modeling
* Limited parallelism

The Transformer (Vaswani et al., 2017) replaced recurrence with:

> Parallelized self-attention over the entire sequence.

GPT models are:

> Decoder-only transformers trained autoregressively for next-token prediction.

---

# 2. Autoregressive Language Modeling

A GPT models:

$$
P(x_t \mid x_{<t})
$$

Training objective:

$$
\mathcal{L} = -\sum_t \log P(x_t^{true})
$$

There is:

* No explicit reasoning objective
* No symbolic supervision
* No tool-specific objective in base pretraining

Everything emerges from minimizing next-token prediction loss at scale.

---

# 3. Tokenization and Embeddings

## Tokenization

Input text is converted into discrete tokens via BPE or related methods.

Example:

```
"Transformers are powerful"
→ [5023, 389, 9376]
```

Vocabulary size:
~30k–200k tokens.

---

## Embedding Layer

Each token index maps to:

$$
x_i \in \mathbb{R}^{d_{model}}
$$

The input becomes:

$$
X \in \mathbb{R}^{T \times d_{model}}
$$

This embedding space becomes:

> The semantic coordinate system of the model.

---

# 4. Positional Encoding

Transformers are permutation invariant.

So positional information is injected via:

$$
H_0 = X + P
$$

Where:

* P = learned positional embeddings (GPT-style)
* Or sinusoidal encodings (original transformer)

This gives the model order awareness.

---

# 5. Transformer Block Mechanics

Each block consists of:

1. Masked multi-head self-attention
2. Feedforward MLP
3. Residual connections
4. Layer normalization

Stacked dozens to hundreds of times.

---

# 6. Self-Attention (Core Computation)

Given hidden states:

$$
H \in \mathbb{R}^{T \times d}
$$

Compute:

$$
Q = HW_Q,\quad K = HW_K,\quad V = HW_V
$$

Attention scores:

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)
$$

Causal mask ensures:

$$
j > i \Rightarrow \text{masked}
$$

So the model cannot see future tokens.

Final attention output:

$$
\text{Attention}(Q,K,V) = \text{softmax}(...)V
$$

---

# 7. Multi-Head Attention

Instead of one attention operation:

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$

Concatenate:

$$
\text{Concat}(\text{head}_1, ..., \text{head}_h) W_O
$$

Different heads specialize in:

* Syntax
* Coreference resolution
* Pattern copying
* Induction
* Long-range retrieval
* Structural alignment

Mechanistic interpretability has identified:

* Induction heads
* Name-matching heads
* Delimiter-detection heads
* Copy suppression heads

---

# 8. Feedforward (MLP) Layer

Each token independently passes through:

$$
\text{MLP}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

Typically:

* Hidden size = 4× model dimension
* GELU activation

Function:

* Feature expansion
* Nonlinear combination
* Writing new features into the residual stream

---

# 9. Residual Stream: The Central State Space

Each block adds to the residual:

$$
H_{l+1} = H_l + \text{Attention}(H_l)
$$

$$
H_{l+1} = H_{l+1} + \text{MLP}(H_{l+1})
$$

This creates:

> A high-dimensional shared workspace.

Interpretability framing:

* Attention heads = information routers
* MLP neurons = feature writers
* Residual stream = distributed working memory

This residual stream accumulates increasingly abstract representations across layers.

---

# 10. Output Projection and Logits

Final hidden state:

$$
H_L
$$

Projected to vocabulary:

$$
\text{logits} = H_L W_{out}
$$

Often:

$$
W_{out} = W_{embed}^T
$$

Then:

$$
P(x_t) = \text{softmax}(\text{logits})
$$

Training optimizes cross-entropy.

---

# 11. Inference and KV Caching

During generation:

1. Forward pass
2. Sample next token
3. Append token
4. Repeat

Naively:
Attention is $O(T^2)$.

Optimized via KV caching:

* Store previous K and V
* Compute attention only for new token

This reduces per-token complexity to $O(T)$.

---

# 12. Scaling Laws and Emergence

Empirical scaling laws show:

$$
\text{Loss} \propto N^{-\alpha}
$$

Where:

* N = parameter count

However, researchers observed that capabilities emerge at scale:

* In-context learning
* Multi-step reasoning
* Code generation
* Tool schema modeling

These are not explicitly programmed.

They arise because:

> Minimizing next-token loss at scale requires building increasingly structured internal world models.

---

# 13. Induction Heads and Algorithmic Circuits

Mechanistic interpretability revealed:

Certain attention heads learn to implement pattern copying.

Example:

```
A B C A B → predict C
```

These are called induction heads.

They function as:

> Learned algorithmic circuits implemented via attention and residual composition.

This suggests transformers can approximate discrete algorithms internally.

---

# 14. Superposition and Feature Geometry

LLMs store far more conceptual features than dimensionality alone suggests.

They accomplish this via:

* Superposition
* Sparse activation
* Distributed encoding

Neurons do not represent single concepts.
Concepts are directions in activation space.

This explains:

* Polysemantic neurons
* Feature interference
* Emergent abstraction

---

# 15. Why Next-Token Prediction Is Sufficient

Language compresses:

* Causality
* Intent
* Planning
* Social reasoning
* Logic
* Mathematics
* Code

To predict the next token well, a model must approximate:

* World models
* Agent models
* Problem-solving structures
* Hierarchical abstraction

Thus:

> Next-token prediction forces implicit modeling of the generative structure behind language.

At sufficient scale, this becomes generalized competence.

---

# 16. From Transformers to Modern Public LLMs

Modern public LLMs extend base GPT pretraining with:

### Instruction Tuning

Supervised fine-tuning on prompt → response pairs.

### RLHF / Preference Optimization

Aligning outputs with human feedback.

### Tool-Use Training

Learning structured schema emission for:

* Function calls
* Code execution
* Retrieval
* Browsing

### Long-Context Improvements

Architectural improvements such as:

* RoPE scaling
* Memory optimizations
* Efficient attention variants

---

# 17. Connection to Reasoning, Code, and Tool Use

From your prior foundational sections:

### Reasoning

Mechanistically:

* Multi-layer feature composition
* Attention-based variable routing
* Residual stream accumulation
* Autoregressive constraint refinement

### Tool Calling

Mechanistically:

* Next-token prediction over structured schemas
* Emission of JSON-like tool calls
* External runtime execution
* Context reintegration

### Code Generation

Mechanistically:

* Structured syntax modeling
* Plan → implement → refine loops
* Induction-based copying
* Abstraction emerging at scale

---

# 18. Final Synthesis

A GPT-style LLM is:

* A deep stack of masked self-attention and MLP layers
* Trained via cross-entropy next-token prediction
* Operating over a high-dimensional residual state
* Scaling according to predictable power laws
* Exhibiting emergent algorithmic behavior

Modern public LLMs extend this core with:

* Alignment layers
* Tool-use capabilities
* Long-context optimization
* Instruction-following behavior

Underneath all of it remains:

> A large autoregressive transformer minimizing next-token prediction — whose scale induces structured internal world modeling and emergent algorithmic competence.

---
