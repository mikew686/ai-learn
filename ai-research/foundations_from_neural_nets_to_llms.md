# Foundations: From Neural Networks to LLMs

## Assumed Background

This section assumes familiarity with basic deep learning concepts including feed-forward neural networks, gradient descent, and backpropagation.

If you would like a refresher on deep neural networks before proceeding, the following are excellent references:

### Recommended Primer Reading

* **Michael Nielsen — *Neural Networks and Deep Learning***
  A clear and intuitive derivation of backpropagation and neural network training.
  [https://neuralnetworksanddeeplearning.com/](https://neuralnetworksanddeeplearning.com/)

* **Goodfellow, Bengio, Courville — *Deep Learning***
  The canonical theoretical reference for deep learning foundations, optimization, and representation learning.
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

* **Stanford CS231n Notes**
  An engineering-oriented overview of neural networks, backpropagation, and optimization techniques.
  [https://cs231n.github.io/](https://cs231n.github.io/)

---

## Recommended Reading for the New Material (Sequence Models → Transformers → LLMs)

For readers wanting deeper grounding in the architectural progression toward LLMs:

* **Bahdanau et al. (2014) — Neural Machine Translation by Jointly Learning to Align and Translate**
  Introduces attention in sequence models.

* **Vaswani et al. (2017) — Attention Is All You Need**
  The original Transformer paper.

* **Kaplan et al. (2020) — Scaling Laws for Neural Language Models**
  Empirical laws governing performance vs. scale.

* **Brown et al. (2020) — Language Models are Few-Shot Learners**
  Demonstrates in-context learning emergence in GPT-3.

* **Anthropic — “Transformer Circuits” & Mechanistic Interpretability Work**
  Insight into internal structure and emergent circuits.

These works provide the theoretical and empirical foundation for modern LLM understanding.

---

# 1. Neural Networks as Function Approximators

At the most fundamental level, neural networks approximate parameterized functions:

$$
f_\theta(x) \approx y
$$

Where:

* $x$ = input
* $y$ = target output
* $\theta$ = weights and biases

Training minimizes:

$$
\mathcal{L}(\theta) = \text{Loss}(f_\theta(x), y)
$$

Using gradient descent:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

Gradients are computed via backpropagation.

---

# 2. Feed-Forward Neural Networks (MLPs)

A multilayer perceptron consists of stacked affine transformations and nonlinearities:

$$
h = \sigma(Wx + b)
$$

Stacking layers:

$$
f(x) = W_n \sigma(W_{n-1} \sigma(... \sigma(W_1 x)))
$$

Key properties:

* Fully connected
* Fixed input size
* No memory
* Universal approximation capability

Limitation: Cannot model sequences or contextual relationships.

---

# 3. Backpropagation and Optimization

Training involves:

1. Forward pass
2. Loss computation
3. Reverse-mode automatic differentiation
4. Parameter update

Challenges:

* Vanishing gradients
* Exploding gradients
* Training instability

Solutions developed over time:

* ReLU/GELU activations
* Improved initialization
* Normalization layers
* Residual connections

Backpropagation remains foundational to LLM training.

---

# 4. Distributed Representations and Embeddings

Discrete tokens must be mapped into continuous space.

Learned embeddings:

$$
e = E[w]
$$

Where:

* $E \in \mathbb{R}^{V \times d}$
* $d \ll V$

Result:

* Semantic similarity emerges geometrically
* Linear relationships encode meaning

Embeddings form the input layer of transformers.

---

# 5. Recurrent Neural Networks (RNNs)

RNN recurrence:

$$
h_t = \phi(Wx_t + Uh_{t-1})
$$

Provides sequential memory.

Limitations:

* Vanishing gradients
* Sequential (non-parallelizable) computation
* Weak long-range modeling

---

## LSTMs and GRUs

Introduce gating mechanisms:

* Input gate
* Forget gate
* Output gate

Improves stability and memory retention, but scaling remains difficult.

---

# 6. Convolutional Neural Networks (CNNs)

CNN contributions relevant to LLM history:

* Local receptive fields
* Weight sharing
* Hierarchical feature extraction

However, limited long-range modeling.

---

# 7. Sequence-to-Sequence Models

Encoder–decoder models:

$$
c = \text{Encoder}(x_1...x_n)
$$

$$
y_t = \text{Decoder}(c, y_{\text{<}t})
$$

Limitation: Fixed-length context bottleneck.

---

# 8. Attention Mechanism

Attention computation:

$$
\text{Attention}(Q,K,V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

Benefits:

* Dynamic context access
* Long-range dependency modeling
* Differentiable memory routing

This is the critical conceptual precursor to transformers.

---

# 9. The Transformer Architecture

Each transformer block includes:

1. Multi-head self-attention
2. Feed-forward network
3. Residual connections
4. Layer normalization

---

## Self-Attention

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

Each token attends to all others.

---

## Multi-Head Attention

Parallel attention heads model different relational subspaces.

---

## Positional Encoding

Injects order information:

* Sinusoidal
* Learned embeddings

---

## Residual Architecture

$$
x_{l+1} = x_l + \text{Block}(x_l)
$$

Enables deep stable training.

---

# 10. Autoregressive Language Modeling (GPT)

Training objective:

$$
P(x_t | x_{\text{<}t})
$$

With causal masking.

Loss:

$$
\mathcal{L} = - \sum_t \log P(x_t \mid x_{\text{<}t})
$$

Next-token prediction at scale produces emergent capabilities.

---

# 11. Scaling and Emergence

As parameter count, data, and compute increase:

* Performance improves predictably
* In-context learning emerges
* Few-shot reasoning appears
* Structured reasoning improves

Scale amplifies representational and algorithmic capacity.

---

# 12. What Makes LLMs Distinct

Modern LLMs are:

* Deep transformer stacks
* Internet-scale trained
* Billion–trillion parameter models
* Distributed-trained

Architecturally still:

* Embeddings
* Self-attention
* MLP layers
* Residual streams
* Layer normalization
* Next-token objective

Reasoning, tool use, and code generation all emerge from this architecture.

---

# Evolution Summary

| Stage                 | Contribution                        |
| --------------------- | ----------------------------------- |
| Feed-forward networks | Universal approximation             |
| Backpropagation       | Gradient-based learning             |
| Embeddings            | Dense semantic representation       |
| RNNs                  | Sequence modeling                   |
| LSTMs/GRUs            | Gated memory                        |
| Attention             | Context routing                     |
| Transformers          | Scalable global modeling            |
| GPT                   | Large-scale autoregressive learning |

---
