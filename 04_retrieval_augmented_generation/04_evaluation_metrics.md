# 📏 4. Evaluation and Metrics for RAG Systems

> *"A RAG pipeline is only as good as its weakest stage — evaluate each component separately."*

RAG systems involve multiple stages:  
(1) **retrieval**, (2) **re-ranking**, (3) **generation**, and (4) **grounding evaluation**.  
Each stage has its own metrics, but the final goal is to measure **answer correctness, faithfulness, and relevance**.

---

## 4.1 Why Evaluation Is Hard

Unlike classic NLP tasks, RAG combines:
- **information retrieval (IR)** → objective metrics like Recall@k  
- **natural language generation (NLG)** → subjective metrics like coherence  
- **grounding** → whether the model’s answer is *based on retrieved evidence*

Therefore, evaluation must be **multi-dimensional**:
1. Retrieval quality  
2. Generation quality  
3. Attribution / grounding  
4. Overall factuality

---

## 4.2 Retrieval Metrics

Retrieval evaluation compares retrieved documents \( \{d_1, ..., d_k\} \) to a set of *relevant documents* \( R \) for a query \( q \).

---

### 4.2.1 Recall@k

Measures how many relevant documents were retrieved among top-k.

$$
\text{Recall@k} = \frac{|R \cap D_k|}{|R|}
$$

- High recall → good coverage  
- Too high may increase noise for generator

---

### 4.2.2 Precision@k

Fraction of retrieved docs that are relevant.

$$
\text{Precision@k} = \frac{|R \cap D_k|}{|D_k|}
$$

Used to assess retriever focus; usually trades off against recall.

---

### 4.2.3 Mean Average Precision (MAP)

Average precision over all queries, considering rank order.

$$
MAP = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{|R_q|} \sum_{k=1}^{N} P_q(k) \cdot rel_q(k)
$$

where \( rel_q(k) = 1 \) if the k-th doc is relevant, else 0.

---

### 4.2.4 nDCG (Normalized Discounted Cumulative Gain)

Rewards retrieving relevant docs earlier in the ranking.

$$
nDCG@k = \frac{1}{Z_k} \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

Used in large-scale retrieval benchmarks like **BEIR**.

---

### 4.2.5 Embedding-based Evaluation

For open-domain retrieval (no gold references),  
measure average cosine similarity between retrieved and ground-truth contexts.

---

## 4.3 Re-ranking Evaluation

Re-rankers are evaluated by:
- **Mean Reciprocal Rank (MRR)** — reciprocal of rank of first relevant doc.  
  $$
  MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}
  $$
- **AUC / ROC** — if trained as binary relevance classifier.
- **Cross-encoder accuracy** — classification accuracy on relevant vs. irrelevant pairs.

---

## 4.4 Generation Metrics

The generator is evaluated like a classic text generation model —  
but with a focus on factuality and grounding.

---

### 4.4.1 ROUGE, BLEU, METEOR

Compare lexical overlap with reference answers.

| Metric | Description | Weakness |
|:--|:--|:--|
| **ROUGE-L** | Longest common subsequence | Lexical bias |
| **BLEU** | n-gram overlap precision | Poor for paraphrasing |
| **METEOR** | BLEU + stemming + synonymy | Better recall |

Useful for benchmarks (e.g., NaturalQuestions) but limited for open-ended QA.

---

### 4.4.2 Embedding-based Metrics (Semantic Similarity)

Use embedding cosine similarity between model output and reference answer.

$$
\text{sim}(A_{\text{model}}, A_{\text{ref}}) = \frac{E(A_m) \cdot E(A_r)}{\|E(A_m)\| \|E(A_r)\|}
$$

Models: `sentence-transformers`, `text-embedding-3-large`, `GTE-large`.

---

### 4.4.3 LLM-based Evaluation (Self-critique)

Use a separate evaluator model to score correctness and coherence.

Prompt example:

> "Given context and answer, rate factual accuracy (0–1) and relevance."

Open frameworks:
- `ragas` (OpenAI/DeepEval)
- `TruLens`
- `LangSmith Evaluation`
- `LLM-as-a-Judge` (GPT-4T, Claude 3.5)

---

## 4.5 Grounding and Faithfulness

Grounding evaluates *how much of the generated text is supported by retrieved evidence*.

### 4.5.1 Faithfulness Score (RAGAS)

Checks if answer statements are entailed by retrieved context.

$$
\text{Faithfulness} = \frac{\text{\# factual claims supported by evidence}}{\text{\# total factual claims}}
$$

### 4.5.2 Context Precision & Context Recall

From RAGAS (2024):

- **Context Precision:** fraction of used evidence that is relevant.  
- **Context Recall:** fraction of relevant evidence actually used.  

---

### 4.5.3 Attribution Score

Measures how well the model cites retrieved context.

$$
\text{Attribution} = \frac{\text{\# sentences correctly attributed}}{\text{total sentences}}
$$

Often estimated via QA-over-answers:
- ask the LLM which source supports each claim  
- check overlap with retrieved docs.

---

## 4.6 Factual Consistency and Hallucination Detection

A *hallucination* is information in the answer not supported by retrieval or knowledge.

### 4.6.1 Automatic Detection

- **FactCC** (Kryscinski et al., 2020)  
- **TRUE** (Honovich et al., 2022)  
- **G-Eval / LLM Eval:** judge if each statement is factually entailed.

### 4.6.2 Quantitative Measures

| Metric | Description |
|:--|:--|
| **Factual Consistency (FC)** | 1 − hallucination rate |
| **Support Rate** | Fraction of statements grounded in retrieval |
| **Non-factual rate** | % unsupported or contradicted claims |

---

## 4.7 Holistic Metrics (End-to-End RAG)

### 4.7.1 Answer Groundedness (AG)

Combines correctness and support:

$$
AG = \text{Correctness} \times \text{Faithfulness}
$$

High AG means: correct *and* well-grounded.

---

### 4.7.2 RAGAS Composite Metrics

[RAGAS](https://github.com/explodinggradients/ragas) evaluates:
- Faithfulness  
- Answer relevancy  
- Context precision  
- Context recall  
- Factual consistency  

and computes an overall **RAG Score**.

---

### 4.7.3 Human Evaluation

Despite automation, **human judgment** remains the gold standard.

Evaluators rate:
- Relevance
- Correctness
- Fluency
- Faithfulness
- Use of retrieved context

Scale: 1–5 Likert or binary (True / Partial / False).

---

## 4.8 Visualization and Monitoring

Use dashboards (LangSmith, Arize, Weights & Biases) to monitor:
- retrieval hit rates
- answer grounding
- latency and cost per query
- qualitative failure cases

Example metrics table:

| Query | Recall@5 | Faithfulness | Factuality | Final Score |
|:--|:--|:--|:--|:--|
| "When was ColBERT introduced?" | 1.0 | 0.95 | 1.0 | 0.98 |
| "How does ColPali differ from CLIP?" | 0.9 | 0.88 | 0.92 | 0.90 |

---

## ✅ Summary

> Evaluating RAG = Evaluating *Retrieval + Generation + Grounding.*

| Stage | Metrics | Tools |
|:--|:--|:--|
| **Retriever** | Recall@k, nDCG, MRR | BEIR, Haystack eval |
| **Re-ranker** | MAP, MRR | PyTerrier, HuggingFace IR |
| **Generator** | ROUGE, BLEU, BERTScore | Evaluate, Ragas |
| **Grounding** | Faithfulness, Attribution, Context Recall | Ragas, TruLens |
| **End-to-end** | Answer Groundedness, Human eval | LangSmith, Ragas |

---

## 📚 References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP.*  
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Information Retrieval.*  
- Honovich et al. (2022). *TRUE: Task of Robustness and Factuality Evaluation.*  
- Kryscinski et al. (2020). *Evaluating the Factual Consistency of Summaries.*  
- Gupta et al. (2024). *RAGAS: Automated Evaluation of RAG Pipelines.*  
- Arora et al. (2024). *Evaluating RAG Systems: Beyond Faithfulness.*


