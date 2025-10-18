# ⚡ 3. Attention and the Transformer Architecture

The **attention mechanism** marked a fundamental shift in NLP, allowing models to dynamically focus on relevant parts of a sequence.  
The **Transformer** (Vaswani et al., 2017) then replaced recurrence altogether, enabling massive scalability and parallelization.

---

## 3.1 Motivation: The Limits of Recurrence

Recurrent Neural Networks (RNNs) and LSTMs process sequences token by token:

$$
h_t = f(W_x x_t + W_h h_{t-1})
$$

This sequential nature has drawbacks:
- Difficult to parallelize.
- Vanishing gradients for long sequences.
- Fixed-size memory representation.

We need a mechanism that:
1. Relates *any* two positions in a sequence directly.  
2. Works in parallel.  
3. Learns what to attend to — automatically.

---

## 3.2 The Core Idea of Attention

Given a **query vector** $q$, a set of **key vectors** $K = [k_1, k_2, ..., k_n]$, and **value vectors** $V = [v_1, ..., v_n]$,  
the model computes **attention weights** measuring how much each value should contribute.

### Formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- $Q$: queries  
- $K$: keys  
- $V$: values  
- $d_k$: dimension of key vectors (used for scaling)

This produces a weighted combination of values, where each token “attends” to others proportionally to their similarity.

---

## 3.3 Intuitive Example

Sentence:  
> “The animal didn’t cross the street because it was too tired.”

Question: what does *it* refer to?

The model should **attend** more strongly to “animal” than to “street”.

Attention allows this dynamic focusing — every token can look at all others.

---

## 3.4 Self-Attention

When attention is applied **within the same sequence**, we call it **self-attention**.

Each token generates:
- its own **query** $q_i$  
- a **key** $k_i$  
- and a **value** $v_i$

Then all tokens attend to one another:

$$
\text{SelfAttention}(X) = \text{softmax}\left(\frac{XW_Q (XW_K)^T}{\sqrt{d_k}}\right) XW_V
$$

This operation captures relationships such as subject–verb agreement or long-range dependencies.

---

## 3.5 Multi-Head Attention

Instead of computing one attention function, Transformers use **multiple heads** in parallel.

Each head projects $Q, K, V$ into subspaces of dimension $d_k / h$, attends separately, and then concatenates results.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O
$$

where  

$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

This lets different heads learn different relationships — syntax, coreference, dependencies, etc.

---

## 3.6 Positional Encoding

Unlike RNNs, Transformers have **no notion of sequence order**.  
They rely on **positional encodings** to inject order information.

### Sinusoidal Encoding
For position $pos$ and dimension $i$:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

These continuous encodings allow the model to learn relative positions.

---

## 3.7 Transformer Block Structure

A single Transformer block consists of:
1. **Multi-head self-attention**
2. **Feed-forward network (FFN)**
3. **Residual connections + Layer normalization**

### Formally
$$
\begin{align}
\text{AttentionOut} &= \text{LayerNorm}(X + \text{MultiHead}(X)) \\
\text{Output} &= \text{LayerNorm}(\text{AttentionOut} + \text{FFN}(\text{AttentionOut}))
\end{align}
$$

---

## 3.8 Encoder–Decoder Architecture

The original Transformer (Vaswani et al., 2017) was designed for machine translation.

**Encoder:**
- Processes the input sequence using self-attention + FFN blocks.
- Outputs contextual embeddings.

**Decoder:**
- Uses masked self-attention (cannot see future tokens).
- Attends to the encoder output.

**Information flow:**
Input → Encoder → Context → Decoder → Output token


---

## 3.9 Masked and Causal Attention

- **Masked (bidirectional):** Used in *BERT* for denoising autoencoding.  
  Each token can attend to all others, including itself.

- **Causal (autoregressive):** Used in *GPT*-like models.  
  Each token attends only to previous tokens (left-to-right).

This masking defines whether the model is generative or contextual.

---

## 3.10 Computational Advantages

| Feature | RNN | Transformer |
|:--|:--|:--|
| Parallelization | ❌ sequential | ✅ full parallel |
| Long dependencies | Weak | Strong |
| Memory of past context | Compressed state | Direct access |
| Complexity | $O(n)$ time | $O(n^2)$ attention (can be reduced) |

---

## 3.11 Scaling Transformers

Variants for efficiency:
- **Sparse Attention:** reduces quadratic cost (e.g., Longformer, BigBird)
- **Perceiver / Performer:** linear attention mechanisms
- **Transformer-XL:** adds recurrence across segments
- **ALiBi / RoPE:** improved positional encodings

---

## 3.12 The Big Picture

Transformers underpin nearly all modern language models:
- **BERT** (bidirectional)
- **GPT series** (autoregressive)
- **T5 / FLAN** (sequence-to-sequence)
- **LLama, Mistral, Claude, Gemini, etc.**

They are not only for text — also used in vision (ViT), speech, protein folding, and multimodal systems.

---

## 🧾 References

- Vaswani, A. et al. (2017). *Attention Is All You Need.*  
- Bahdanau, D. et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.*  
- Shaw, P. et al. (2018). *Self-Attention with Relative Position Representations.*  
- Beltagy, I. et al. (2020). *Longformer: The Long-Document Transformer.*

---

## ✅ Summary

> Attention replaces recurrence by allowing direct interaction between any tokens.  
> The Transformer architecture operationalizes this idea at scale — using self-attention, residuals, and positional encoding —  
> enabling massive pretraining and transfer across tasks.

