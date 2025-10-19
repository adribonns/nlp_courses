# 🔍 1. Retrieval-Augmented Generation (RAG): Principles and Pipeline

Large Language Models (LLMs) are powerful, but their **knowledge is static** — fixed at training time.  
Retrieval-Augmented Generation (RAG) bridges this limitation by allowing models to **access external information** dynamically.

---

## 1.1 Motivation

> “Don’t retrain your model — retrieve what it needs.”

Traditional LLMs rely solely on **parametric memory** (weights).  
RAG adds **non-parametric memory** — external data sources that the model can query at runtime.

### Why RAG?
- Overcomes *knowledge cutoff*  
- Reduces hallucination  
- Enables domain adaptation (e.g., legal, medical)  
- Maintains interpretability (retrieved evidence)

---

## 1.2 Core Architecture

      +--------------------+
      |   User Query (Q)   |
      +--------------------+
                  ↓
        [Embedding Model]
                  ↓
          Dense Vector q
                  ↓
      +--------------------+
      |   Retriever (ANN)  |
      +--------------------+
                  ↓
        Top-k Documents D*
                  ↓
      +--------------------+
      |  Generator (LLM)   |
      +--------------------+
                  ↓
         Context-aware Answer


---

## 1.3 Mathematical Formulation

Given:
- Query $q$
- Knowledge corpus $\mathcal{D} = \{d_1, d_2, ..., d_N\}$
- Retriever $p_\eta(d | q)$
- Generator $p_\theta(y | q, d)$

We want to maximize:

$$
\hat{y} = \arg\max_y \sum_{d \in \mathcal{D}} p_\eta(d | q) \, p_\theta(y | q, d)
$$

In practice, the retriever selects top-$k$ documents:
$$
D^* = \text{TopK}_{d \in \mathcal{D}} \, \text{sim}(E_q(q), E_d(d))
$$
and the generator conditions on $D^*$.

---

## 1.4 Two Core Components

### (1) Retriever
Finds relevant passages given a query.

**Types:**
- **Sparse retrieval:** TF-IDF, BM25 (keyword-based)
- **Dense retrieval:** Embeddings + similarity search

#### Dense Retrieval
Encode queries and documents:
$$
q = f_\text{enc}(Q), \quad d = f_\text{enc}(D)
$$
Compute cosine similarity:
$$
\text{sim}(q, d) = \frac{q \cdot d}{\|q\|\|d\|}
$$

Efficient indexing with **Approximate Nearest Neighbor (ANN)** methods (e.g., FAISS, ScaNN, Milvus).

---

### (2) Generator
Produces a response conditioned on retrieved context.

$$
P(y | Q, D^*) = \prod_t P(y_t | y_{<t}, Q, D^*)
$$

Typically a **decoder-only LLM** (GPT, LLaMA, Mistral).

---

## 1.5 Variants of RAG Architectures

| Type | Description | Example |
|:--|:--|:--|
| **Naive RAG** | Retrieve top-k documents, concatenate to prompt | LangChain “stuff” |
| **RAG-Sequence** | Retrieve at each decoding step | Lewis et al. (2020) |
| **RAG-Token** | Each token attends to different retrieved doc | Advanced (Facebook RAG) |
| **Iterative / Re-ranking RAG** | Retrieve → generate → refine → retrieve | Self-RAG, ReAct |
| **Tool-Augmented RAG** | Integrate APIs or knowledge graphs | Toolformer, GraphRAG |

---

## 1.6 Chunking and Indexing

Documents must be chunked before retrieval.

### Heuristic Approaches:
- Fixed-size windows (e.g., 512 tokens)
- Sliding window with overlap
- Sentence or paragraph-based
- Semantic segmentation (using embeddings)

### Trade-offs:
| Chunk size | Pros | Cons |
|:--|:--|:--|
| Small | Precise retrieval | Fragmented context |
| Large | Richer context | Noisy, slower search |

> Common practice: 300–700 tokens per chunk with 20–30% overlap.

---

## 1.7 Context Construction and Prompting

Retrieved passages are concatenated before generation:

[System Prompt]
You are a helpful assistant. Use the context below.

[Context]

Passage A ...

Passage B ...

[Question]
Q: What are the benefits of RAG?


LLMs are prompted to **ground** their answer in retrieved text.

---

## 1.8 End-to-End RAG Pipelines

| Framework | Core Idea | Example |
|:--|:--|:--|
| **LangChain** | Chain-of-components (retriever, LLM, memory) | Pythonic workflows |
| **LlamaIndex** | Graph-based document indexing | Multi-level RAG |
| **Haystack** | Modular retriever–reader pipeline | Enterprise-grade |
| **HuggingFace RAG** | Integrated retriever + generator | RAG-Token, RAG-Sequence |

---

## 1.9 Retrieval Models

| Model | Base | Description |
|:--|:--|:--|
| **DPR** | BERT | Dense Passage Retrieval (dual encoder) |
| **Contriever** | RoBERTa | Self-supervised contrastive retrieval |
| **E5 / GTE / BGE** | Transformer-based | Universal text embeddings |
| **ColBERT / SPLADE** | Hybrid dense-sparse | Token-level matching |
| **OpenAI Text-Embedding-3** | LLM-based | State-of-the-art embeddings |

---

## 1.10 Knowledge Update via RAG

Unlike fine-tuning, RAG allows **instant knowledge injection**:

| Method | Updates Knowledge? | Requires Retraining? |
|:--|:--|:--|
| Fine-tuning | ✅ | ✅ |
| Prompting | ⚠️ (limited) | ❌ |
| **RAG** | ✅ (dynamic retrieval) | ❌ |

---

## 1.11 Evaluation Metrics

| Aspect | Metric | Description |
|:--|:--|:--|
| Retrieval | Recall@K, Precision@K | Relevance of retrieved docs |
| Generation | BLEU, ROUGE, F1 | Surface-level similarity |
| Faithfulness | FActScore, TruthfulQA | Alignment with context |
| End-to-end | Human eval, groundedness score | Overall quality |

---

## 1.12 Typical RAG Workflow (Textual Schema)

┌─────────────────────────────┐
│ Documents │
└──────────────┬──────────────┘
│ Chunking
▼
┌─────────────────────┐
│ Vector Index (DB) │
└─────────────────────┘
│ Embedding + ANN
▼
┌─────────────────────┐
│ Retriever │
└─────────────────────┘
│ Top-k docs
▼
┌─────────────────────┐
│ LLM (Generator)│
└─────────────────────┘
│
▼
┌─────────────────────┐
│ Grounded Answer │
└─────────────────────┘


---

## 1.13 RAG vs Fine-Tuning

| Feature | RAG | Fine-Tuning |
|:--|:--|:--|
| Knowledge update | Dynamic | Static |
| Training cost | None | High |
| Adaptability | High | Medium |
| Control | Explicit (retrieval) | Implicit (weights) |
| Hallucination | Lower (if retrieval relevant) | Often higher |

---

## 1.14 Advanced Topics (Preview)

- **Self-RAG:** Model decides *when* and *what* to retrieve.  
- **Graph-RAG:** Use structured graphs for retrieval (Microsoft 2024).  
- **Memory-Augmented RAG:** Persistent cache for iterative context.  
- **RAG with Multi-Agent Systems:** Specialized retrieval per subdomain.  
- **ColPali / RAG-Vision:** Integration of visual retrieval (next lesson 🔜).

---

## ✅ Summary

> RAG enhances language models with external retrieval.  
> It decouples *knowledge access* from *model parameters*, enabling dynamic, grounded, and explainable reasoning.

Mathematically:
$$
P(y|q) = \sum_{d \in D^*} p_\eta(d|q) \, p_\theta(y|q,d)
$$

Practically:
> Chunk → Embed → Index → Retrieve → Generate → Evaluate

---

## 📚 References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*  
- Izacard et al. (2022). *Self-RAG: Learning to Retrieve, Generate, and Refine.*  
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain QA.*  
- Gao et al. (2021). *SimCSE: Simple Contrastive Learning for Sentence Embeddings.*  
- Sun et al. (2023). *BGE and E5 Embedding Models.*  
- Microsoft (2024). *GraphRAG: Leveraging Knowledge Graphs for RAG.*


