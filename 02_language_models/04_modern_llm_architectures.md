# 🧩 4. Modern LLM Architectures

Large Language Models (LLMs) are built upon the Transformer architecture, but over time, a variety of **architectural paradigms and training objectives** have emerged.  
Understanding these distinctions is essential to reason about model capabilities, efficiency, and behavior.

---

## 4.1 Core Transformer Variants

Modern Transformer-based LMs fall into **three main architectural families**:

| Type | Example Models | Description | Training Objective |
|:--|:--|:--|:--|
| **Encoder-only** | BERT, RoBERTa, DeBERTa | Bidirectional, context-understanding | Masked Language Modeling (MLM) |
| **Decoder-only** | GPT, LLaMA, Falcon, Mistral | Autoregressive generation | Causal Language Modeling (CLM) |
| **Encoder–Decoder** | T5, BART, FLAN-T5 | Seq2seq tasks (input → output) | Denoising / Translation / Summarization |

---

### Encoder-only (BERT-style)

- Processes the entire sentence bidirectionally.
- Trained to reconstruct masked tokens.

**Objective:**

$$
\mathcal{L}_{MLM} = - \sum_{i \in M} \log P(w_i | w_{\setminus M})
$$

where $M$ is the set of masked positions.

**Use cases:** text classification, NER, sentence embeddings, retrieval.

---

### Decoder-only (GPT-style)

- Processes text left-to-right.
- Predicts next token given all previous ones.

**Objective:**

$$
\mathcal{L}_{CLM} = - \sum_{t} \log P(w_t | w_{<t})
$$

**Use cases:** text generation, reasoning, dialogue, coding.

---

### Encoder–Decoder (T5-style)

- Encoder reads input (source sequence).
- Decoder generates output (target sequence), attending to encoder output.

Used for translation, summarization, question answering, etc.

**Objective:**

$$
\mathcal{L}_{seq2seq} = - \sum_{t} \log P(y_t | y_{<t}, x)
$$

---

## 4.2 Training Objectives and Their Impact

| Objective | Directionality | Typical Use | Example Model |
|:--|:--|:--|:--|
| **CLM (Causal)** | Left-to-right | Generation | GPT, LLaMA |
| **MLM (Masked)** | Bidirectional | Understanding | BERT |
| **Seq2Seq** | Conditional | Translation, Summarization | T5 |
| **Denoising Autoencoding** | Bidirectional | Robust text recovery | BART |

**Key insight:**  
Autoregressive models (CLM) generate; masked models (MLM) understand.

---

## 4.3 Scaling Laws and Pretraining

Empirical studies (Kaplan et al., 2020) show that model performance scales predictably with:
- Parameters ($N$)
- Dataset size ($D$)
- Compute budget ($C$)

Roughly:

$$
\text{Loss} \propto N^{-\alpha_N} D^{-\alpha_D} C^{-\alpha_C}
$$

Hence, LLMs are typically trained with trillions of tokens and hundreds of billions of parameters.

---

## 4.4 Mixture of Experts (MoE)

A modern trend for scaling efficiently.

### Idea:
Instead of activating the full model, route tokens through a **subset of experts**.

Architecture:
- A **router network** selects $k$ experts out of $N$ for each token.
- Each expert is a small feed-forward layer.

**Computation:**

$$
y = \sum_{i=1}^N p_i(x) \cdot \text{Expert}_i(x)
$$

### Benefits
- Much larger capacity with same compute per token.
- Enables trillion-parameter models (e.g., Switch Transformer, Mixtral).

**Examples:** GLaM, Switch-Transformer, Mixtral-8x7B, DeepSeek-V2.

---

## 4.5 State-Space Models: Mamba, S4, RetNet

Transformers struggle with long sequences due to $O(n^2)$ attention cost.  
**State-space models (SSMs)** provide a continuous-time alternative.

### Core principle
Maintain a latent state updated recursively:

$$
h_t = A h_{t-1} + B x_t
$$

$$
y_t = C h_t
$$

Where matrices $(A, B, C)$ define a dynamic system over the sequence.

### Mamba (Gu & Dao, 2023)
Introduces **selective state updates** with gating:
- Linear-time sequence modeling ($O(n)$)
- Comparable performance to Transformers
- Efficient for long-context inference

**Examples:** Mamba, RetNet, Hyena, RWKV (RNN-inspired hybrids)

---

## 4.6 Architectural Hybrids and Trends

Modern LLMs often mix paradigms:
- **Retrieval-augmented models** (RAG, RETRO, Atlas)
- **Memory-augmented models** (MemGPT, Recurrent Transformers)
- **Multimodal Transformers** (LLaVA, Gemini, GPT-4V)
- **Tool-using models** (function calling, code execution)

---

## 4.7 Comparison Summary

| Architecture | Directionality | Strength | Limitation |
|:--|:--|:--|:--|
| Encoder-only | Bidirectional | Understanding | Cannot generate |
| Decoder-only | Unidirectional | Generation, reasoning | No full context |
| Encoder–Decoder | Conditional | Flexible tasks | More parameters |
| MoE | Sparse compute | Efficiency, capacity | Routing complexity |
| State-space | Recurrent-like | Long context, speed | Training stability |

---

## 4.8 Training Strategies and Objectives

### 1. Pretraining (Foundation)
Objective: next-token prediction (CLM) or masked recovery (MLM).

### 2. Supervised Fine-Tuning (SFT)
Use labeled datasets (e.g., instruction–response pairs).

### 3. Preference Optimization / Alignment
Adjust model behavior to follow human intent.

---

## 4.9 The Role of Pretraining Objective

- **CLM** → generation, completion, reasoning  
- **MLM** → understanding, classification  
- **Seq2Seq** → instruction following, translation  
- **Contrastive / Multimodal** → alignment across modalities

---

## 4.10 Future Directions

Emerging directions in architecture:
- **Mixture of Experts + Retrieval** (hybrid scaling)
- **Recurrent Transformers** (longer memory)
- **State-space hybrids** (Mamba++)
- **Multimodal pretraining** (text–image–audio fusion)
- **Neural-symbolic integration** (reasoning + knowledge graphs)

---

## 🧾 References

- Vaswani et al. (2017). *Attention Is All You Need.*  
- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.*  
- Fedus et al. (2021). *Switch Transformers.*  
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*  
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).*  
- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.*

---

## ✅ Summary

> Modern LLMs extend the Transformer into multiple paradigms — decoder-only for generation, encoder-only for understanding, and encoder–decoder for conditional tasks.  
> New architectures like Mixture of Experts and Mamba push efficiency and scalability, while multimodal and retrieval-augmented systems redefine the capabilities of foundation models.

