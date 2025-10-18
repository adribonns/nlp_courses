# 🧠 1. From n-grams to Transformers

Language Modeling (LM) is the task of assigning a probability to a sequence of words.
It forms the mathematical foundation of most NLP systems — from predictive text to GPT-style models.

---

## 1.1 What Is a Language Model?

A language model estimates the probability of a word sequence:

P(w₁, w₂, ..., wₙ)

By the chain rule of probability:

P(w₁, w₂, ..., wₙ) = Π (from i=1 to n) P(wᵢ | w₁, ..., wᵢ₋₁)

This is the **generative view of language**: predicting the next word given the previous context.

---

## 1.2 The n-gram Model

Early language models assumed that only the last n−1 words matter (Markov assumption):

P(wᵢ | w₁, ..., wᵢ₋₁) ≈ P(wᵢ | wᵢ₋₍ₙ₋₁₎, ..., wᵢ₋₁)

### Example (bigram model)
P("the cat sat") = P("the") × P("cat" | "the") × P("sat" | "cat")

### Key idea
- Build frequency counts from a corpus.  
- Estimate probabilities using relative frequencies:
  P(wᵢ | wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)

---

## 1.3 The Problem of Sparsity

Language is vast — many word sequences never appear in training data.

To deal with unseen n-grams, **smoothing** techniques are used:
- Additive (Laplace) smoothing
- Good-Turing discounting
- Kneser-Ney smoothing (state-of-the-art for n-grams)

Even so, n-gram models:
- Need huge storage for large n
- Struggle with long-range dependencies
- Treat words as discrete symbols (no notion of meaning)

---

## 1.4 The Neural Language Model (Bengio et al., 2003)

Introduced the idea of **learning distributed word representations** and predicting with a neural network.

Instead of discrete symbols, each word wᵢ is mapped to a continuous vector e(wᵢ).  
These vectors capture **semantic similarity**.

Network architecture (simplified):

Input: [e(wᵢ₋₃), e(wᵢ₋₂), e(wᵢ₋₁)]  
→ Hidden layer → Softmax output over vocabulary

### Advantages
- Generalizes to unseen n-grams via embeddings  
- Learns smooth probability distributions  
- First step toward **deep learning in NLP**

**Reference:** Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model.*

---

## 1.5 Recurrent Neural Networks (RNNs)

To model longer dependencies, we need recurrence:

At each time step t:
- hₜ = f(Wₓxₜ + Wₕhₜ₋₁)
- yₜ = softmax(Wₒhₜ)

Here, hₜ is a **hidden state** summarizing all previous context.

**Strengths**
- Can, in theory, capture unbounded context.
- Works well for sequential data (text, speech).

**Weaknesses**
- Hard to train on long sequences (vanishing gradients).
- Sequential computation prevents parallelization.

---

## 1.6 LSTM and GRU: Gated RNNs

To mitigate vanishing gradients, gated architectures were proposed:

### LSTM (Hochreiter & Schmidhuber, 1997)
Adds memory cells and gates:
- Input gate: controls how much new info enters.
- Forget gate: controls what to discard.
- Output gate: controls what to emit.

This allows **long-term dependency tracking**.

### GRU (Cho et al., 2014)
Simpler alternative: merges input and forget gates.

---

## 1.7 The Rise of Attention

RNNs process tokens sequentially — slow and limited.  
In 2014, Bahdanau et al. introduced **attention** for machine translation.

Key idea: at each step, the model “attends” to relevant parts of the input sequence.

Attention weight for token i at step t:
αₜᵢ = softmax(score(hₜ₋₁, hᵢ))

Context vector:
cₜ = Σ αₜᵢ · hᵢ

This allows dynamic focusing on different parts of the sequence — a major breakthrough.

---

## 1.8 Transformers (Vaswani et al., 2017)

Transformers replaced recurrence entirely with **self-attention**.

Each token attends to all others in parallel, using the **Scaled Dot-Product Attention** mechanism.

For a query Q, key K, and value V:

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

Where dₖ is the key dimension.

### Multi-head Attention
Multiple attention “heads” run in parallel, allowing the model to capture different types of relationships.

Output = Concat(head₁, ..., headₕ) × Wₒ

---

## 1.9 Advantages of Transformers

- Fully parallelizable (fast training)
- Capture long-range dependencies efficiently
- Scale well with data and parameters
- Foundation for large pre-trained models (BERT, GPT, T5)

---

## 1.10 Historical Summary

| Era | Model | Key Innovation |
|:--|:--|:--|
| 1980s–1990s | n-gram | Statistical co-occurrence |
| 2003 | Neural LM | Embeddings + NN prediction |
| 2013–2015 | RNN/LSTM/GRU | Sequential modeling |
| 2014 | Attention | Dynamic context weighting |
| 2017 | Transformer | Parallel self-attention |

---

## 1.11 References

- Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model.*  
- Bahdanau, D. et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.*  
- Vaswani, A. et al. (2017). *Attention is All You Need.*  
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.*

---

## ✅ Summary

> The path from n-grams to transformers represents the evolution of NLP itself:  
> from discrete counts to continuous representations, from local to global context, and from statistical models to massive pretrained architectures.  
> Understanding this journey is key to grasping how modern language models like GPT actually work.

