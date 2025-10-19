# ⚙️ 2. Core Components and Strategies in RAG Pipelines

While the RAG concept is simple — *retrieve, then generate* — the actual performance depends on how you handle each component:  
**chunking**, **retrieval**, **re-ranking**, and **generation**.

This section reviews common techniques and trade-offs for each step.

---

## 2.1 Chunking (Document Splitting)

Chunking transforms large documents into manageable, semantically coherent pieces.  
Good chunking is crucial: too small → loss of context; too large → irrelevant retrieval.

### 2.1.1 Fixed-size Splitters

- Divide text into chunks of `N` tokens (e.g., 512 or 1000).
- Often use sliding window with overlap to preserve coherence.

**Parameters:**
- `chunk_size`: typical values between 300–700 tokens.
- `chunk_overlap`: 10–30% overlap between chunks.

**Pros:** simple, fast  
**Cons:** ignores semantic or structural boundaries

---

### 2.1.2 Recursive Splitters

Break documents hierarchically:
1. Split by sections → paragraphs → sentences → tokens.
2. Recombine until size constraint met.

**Example (LangChain RecursiveCharacterTextSplitter):**

["### Title", "Paragraph 1...", "Paragraph 2...", ...]


Preserves context hierarchy — ideal for technical or academic texts.

---

### 2.1.3 Semantic Splitters

Use **embedding similarity** or **sentence transformers** to detect semantic boundaries:
- Compute embeddings for consecutive sentences.
- Split when cosine similarity drops below a threshold.

$$
\text{sim}(E(s_i), E(s_{i+1})) < \delta \implies \text{new chunk}
$$

**Tools:** SemanticTextSplitter, GTE-embeddings, Cohere splitter.

**Pros:** natural boundaries, meaningful retrieval  
**Cons:** slower, needs embeddings

---

### 2.1.4 Domain-specific Splitters

| Domain | Strategy |
|:--|:--|
| **Code** | Split by function/class (AST-based) |
| **Markdown** | Split by heading levels |
| **PDFs** | Use layout-based segmenters (e.g., PyMuPDF) |
| **Tables/CSV** | Split row-wise or by logical groups |
| **Conversations** | Split by turn or speaker |

---

## 2.2 Retrieval Techniques

Retrieval determines what knowledge enters the model.  
RAG quality is **upper-bounded by retrieval relevance**.

### 2.2.1 Sparse Retrieval (Lexical)

Based on keyword overlap.

**BM25** (Okapi formula):

$$
\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \frac{f(t,d) (k_1 + 1)}{f(t,d) + k_1(1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
$$

**Tools:** Elasticsearch, Lucene, Whoosh.

**Pros:** transparent, interpretable, fast  
**Cons:** no semantic understanding

---

### 2.2.2 Dense Retrieval (Vector-based)

Encode queries and documents into dense vectors:

$$
E_q = f_{\text{enc}}(q), \quad E_d = f_{\text{enc}}(d)
$$

Similarity via cosine:

$$
\text{sim}(E_q, E_d) = \frac{E_q \cdot E_d}{\|E_q\|\|E_d\|}
$$

**Models:** DPR, E5, BGE, GTE, OpenAI Text-Embedding-3.

**Indexes:** FAISS, Milvus, Qdrant, Weaviate.

---

### 2.2.3 Hybrid Retrieval

Combine sparse + dense signals:

$$
\text{score}(q,d) = \alpha \, \text{BM25}(q,d) + (1 - \alpha) \, \text{sim}(E_q, E_d)
$$

**Example:** ColBERT, SPLADE, HyDE (Hybrid Dense + Expansion).

---


### 2.2.4 Ensemble Retrievers

While *hybrid retrieval* combines sparse and dense scores **at the vector level**,  
an **ensemble retriever** combines the *outputs of multiple retrievers* at the *ranking level*.

Each retriever produces its own list of top documents with a score, and these are **merged or re-weighted**.

---

#### 🧩 Motivation

No single retriever is perfect:

| Retriever Type | Strengths | Weaknesses |
|:--|:--|:--|
| BM25 | precise for keyword overlap | ignores semantics |
| Dense (e.g. DPR, E5) | semantic generalization | may miss exact matches |
| Clustering / Summary | broad coverage | low granularity |
| Domain retrievers | specialized vocab | poor generalization |

Ensembling allows combining **complementary recall patterns**.

---

#### ⚙️ General Formula

Let $S_i(q, d)$ be the normalized score given by retriever $i$ for query $q$ and document $d$.  
The ensemble score is a weighted combination:

$$
S_{\text{ensemble}}(q, d) = \sum_i w_i \, S_i(q, d)
$$

where $\sum_i w_i = 1$.

Documents are then ranked by $S_{\text{ensemble}}(q, d)$.

---

#### 💡 Example — BM25 + Dense Retriever

Suppose you query:  
> "how to fine-tune a transformer model?"

| Document | BM25 score | Dense sim | Normalized scores | Weighted (0.6 BM25 / 0.4 dense) |
|:--|--:|--:|--:|--:|
| D1: “Fine-tuning BERT tutorial” | 0.80 | 0.60 | 0.89 / 0.75 | **0.83** |
| D2: “Transformer model structure” | 0.75 | 0.55 | 0.83 / 0.69 | 0.78 |
| D3: “Neural nets training steps” | 0.50 | 0.40 | 0.56 / 0.50 | 0.53 |

Final ranking → **D1 > D2 > D3**

---

#### 🧠 Normalization & Fusion Methods

Before combination, scores from different retrievers must be made *comparable*.  
Common normalization:

$$
S'_i(q, d) = \frac{S_i(q, d) - \min S_i}{\max S_i - \min S_i}
$$

Then combine via:
- **Weighted sum** (as above)
- **Reciprocal Rank Fusion (RRF)**:
  $$
  RRF(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}
  $$
- **Rank aggregation** (Borda count, Condorcet)

---

#### 🧰 Implementation Patterns

| Framework | Method |
|:--|:--|
| **LangChain** | `EnsembleRetriever([bm25, faiss], weights=[0.6, 0.4])` |
| **LlamaIndex** | `QueryFusionRetriever` |
| **Haystack** | `EnsembleRetriever` (supports RRF, weighted sum) |

---

#### 🧭 Best Practices

- Start with **BM25 + Dense** ensemble → strongest baseline.  
- Normalize scores before fusion.  
- Tune weights empirically (e.g., 0.6 lexical / 0.4 dense).  
- Use **Cross-Encoder** re-ranking *after* ensemble fusion for final precision.

---

**Result:**  
Ensemble retrievers achieve **higher recall** than any individual retriever  
and maintain good precision when paired with a re-ranker.

---


### 2.2.5 Clustering and Summary-based Retrieval

Used for **large-scale corpora**.

- Cluster chunks by embedding similarity (K-Means, HDBSCAN)
- Store **cluster centroids** or **summaries**
- Retrieve cluster → expand to original documents

**Benefits:**
- Lower latency
- Fewer false positives
- Better diversity in retrieved docs

---

### 2.2.6 Hierarchical Retrieval

Multi-level retrieval pipeline:

Doc-level → Section-level → Chunk-level


Example: retrieve top 10 documents, then fine-grained chunks within them.

---

## 2.3 Re-ranking

After retrieval, **re-ranking** selects the *most relevant* passages for generation.

### 2.3.1 Bi-encoder vs Cross-encoder

| Type | Architecture | Description |
|:--|:--|:--|
| **Bi-encoder** | Two encoders (query/doc) → similarity | Fast, scalable (used in retrieval) |
| **Cross-encoder** | Jointly encode (query, doc) pair → relevance score | Slow, high precision (used in re-ranking) |

**Cross-encoder model:**

$$
s(q, d) = f_{\text{cross}}([q; d])
$$

**Models:** MonoT5, MiniLM CrossEncoder, Cohere Rerank, bge-reranker.

---

### 2.3.2 Rank Fusion

Combine rankings from multiple retrievers:
- **Reciprocal Rank Fusion (RRF):**
  
$$
  RRF(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}
  
$$

- **Borda count**, **Condorcet voting**, etc.

Useful for hybrid or multi-database retrieval.

---

### 2.3.3 Learned Re-ranking

Train a lightweight neural ranker on top of retrieval results using pairwise losses:

$$
\mathcal{L}_{\text{rank}} = -\log \sigma(s(q,d^+) - s(q,d^-))
$$

---

## 2.4 Reader / Generator Strategies

The “Reader” or “Generator” consumes the selected passages and produces the final answer.  
Different *composition strategies* exist for feeding context into the LLM.

---

### 2.4.1 Stuff

Concatenate all retrieved passages into one prompt.

**Pros:** simplest, easy to implement  
**Cons:** limited by context window, redundant context

---

### 2.4.2 Map–Reduce

**Map:** Summarize each chunk independently.  
**Reduce:** Combine summaries into final synthesis.

Map: LLM(chunk_i) → summary_i
Reduce: LLM(summary_1...n) → final answer


**Useful for:** long docs, scalability

---

### 2.4.3 Refine (Iterative Summarization)

Sequentially refine an answer using each chunk:

Answer_0 = ""
For each chunk_i:
$Answer_i = LLM(Answer_{i-1} + chunk_i)$


Captures context incrementally — better than simple concatenation.

---

### 2.4.4 Compact / Condense

Before feeding to LLM, merge overlapping or redundant chunks using embedding similarity:

$$
\text{sim}(E(d_i), E(d_j)) > \tau \implies \text{merge}(d_i, d_j)
$$

Reduces prompt size and improves signal-to-noise ratio.

---

### 2.4.5 Multi-hop RAG

Retrieve iteratively for complex reasoning:
1. First-hop: retrieve supporting facts  
2. Second-hop: refine query using intermediate answer

**Example:** HotpotQA, Self-RAG (Meta, 2024)

---

## 2.5 End-to-End Optimization

### 2.5.1 Balancing Recall and Precision

| Goal | Strategy |
|:--|:--|
| Maximize recall | Large `k`, small chunk size |
| Maximize precision | Smaller `k`, re-ranking, hybrid retrieval |

### 2.5.2 Latency vs Quality

- ANN search → trade precision for speed  
- Cache frequent queries  
- Precompute embeddings asynchronously

### 2.5.3 Adaptive Retrieval

Dynamic `k` and context selection based on query type:
- factual → small `k`  
- exploratory → large `k`

---

## 2.6 Practical Pipeline Example

1️⃣ Split → Recursive + Semantic splitter (chunk_size=512)
2️⃣ Embed → OpenAI text-embedding-3-large
3️⃣ Index → FAISS + BM25 hybrid
4️⃣ Retrieve → Top 10 chunks
5️⃣ Rerank → Cross-encoder (bge-reranker-large)
6️⃣ Compact → Merge redundant
7️⃣ Generate → LLM (map-reduce w/ refine)
8️⃣ Evaluate → Recall@K + groundedness


---

## ✅ Summary

> The quality of RAG is not limited by the LLM,  
> but by *how effectively* you retrieve, filter, and present information.

Key takeaways:
- Chunk meaningfully → retrieve precisely → rerank intelligently → generate efficiently.
- Balance **recall vs latency**.
- Choose **semantic-aware splitters** and **cross-encoders** for high-stakes domains.

---

## 📚 References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP.*  
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain QA.*  
- Gao et al. (2021). *SimCSE: Contrastive Learning for Sentence Embeddings.*  
- Pradeep et al. (2023). *Beyond BM25: Hybrid Retrieval Models.*  
- Thakur et al. (2021). *BEIR: Benchmarking Information Retrieval Models.*  
- Asai et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Refine.*


