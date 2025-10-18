# 📏 4. Evaluation Metrics in NLP

Evaluation is a critical step in NLP — it tells us how well a model performs a given language task.  
Metrics depend on **the type of problem** (classification, generation, retrieval, etc.) and on the **nature of the data** (balanced vs imbalanced, structured vs free text).

---

## 4.1 Why Evaluation Matters

A model that "sounds good" may not actually perform well.  
We need **quantitative and objective measures** to compare models, optimize them, and detect overfitting.

### Key Evaluation Aspects
- **Accuracy:** How often predictions are correct  
- **Precision/Recall/F1:** Quality of positive predictions  
- **BLEU / ROUGE:** Quality of generated text  
- **Perplexity:** Fluency of probabilistic language models

---

## 4.2 Classification Metrics

For many NLP tasks (e.g., sentiment analysis, spam detection, NER), evaluation follows standard classification metrics.

### Confusion Matrix
|                | Predicted Positive | Predicted Negative |
|----------------|-------------------:|-------------------:|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Basic Metrics
$$\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Interpretation:**
- **Precision**: correctness of positive predictions.  
- **Recall**: coverage of true positives.  
- **F1-score**: harmonic mean balancing both.

### Example

| Model | Precision | Recall | F1-score |
|:--|--:|--:|--:|
| Logistic Regression | 0.87 | 0.76 | 0.81 |
| Transformer | 0.92 | 0.91 | **0.91** |

---

## 4.3 Multi-class and Sequence Evaluation

### Macro vs Micro Averaging
When dealing with multiple classes (e.g., emotion detection):
- **Macro average:** compute metric per class, then average.
- **Micro average:** aggregate all TP/FP/FN before computing.

### Sequence Labeling (e.g., NER, POS-tagging)
We use token-level or entity-level F1.  
Entity-level metrics consider full spans rather than single tokens.

**Reference:** Tjong Kim Sang & De Meulder (2003) — *Introduction to the CoNLL-2003 Shared Task.*

---

## 4.4 Metrics for Language Generation

Tasks like translation, summarization, or dialogue require measuring **textual similarity** rather than exact matches.

---

### 4.4.1 BLEU (Bilingual Evaluation Understudy)

**Purpose:** Evaluate machine translation by comparing n-gram overlap with reference translations.

Formula:
$$\text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)$$

where:
- \(p_n\): precision for n-grams  
- \(BP\): brevity penalty = \( \min(1, e^{1 - r/c}) \),  
  with \(r\) = reference length, \(c\) = candidate length  

**Interpretation:**
- High BLEU → similar to reference  
- Ignores synonyms and semantics (surface-level)

**Typical range:** 0–1 (or 0–100 when scaled)

**Reference:** Papineni et al. (2002), *BLEU: a Method for Automatic Evaluation of Machine Translation.*

---

### 4.4.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Used for **summarization**.  
Measures overlap between system and reference summaries.

Most common variant:  
$$\text{ROUGE-1, ROUGE-2} = \text{Recall of unigrams/bigrams overlap}$$
$$\text{ROUGE-L} = \text{Longest Common Subsequence Recall}$$

**Reference:** Lin (2004), *ROUGE: A Package for Automatic Evaluation of Summaries.*

---

### 4.4.3 METEOR, BERTScore, and Beyond

- **METEOR**: Considers stemming, synonyms, and alignment.  
- **BERTScore**: Uses contextual embeddings (BERT) for semantic similarity —  
  captures meaning, not just token overlap.

$$\text{BERTScore} = \text{mean cosine similarity between token embeddings}$$

**Reference:** Zhang et al. (2019), *BERTScore: Evaluating Text Generation with BERT.*

---

## 4.5 Perplexity — Evaluating Language Models

Perplexity measures how well a model predicts text (likelihood of a test corpus).

$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1}) \right)$$

- Lower perplexity → better predictive power.  
- Used for traditional LMs and autoregressive models.

**Caveats:**  
Not ideal for open-ended tasks like dialogue; may not correlate with human judgment.

---

## 4.6 Human Evaluation

For open-ended generation tasks (dialogue, summarization, storytelling):
- **Fluency** — grammaticality and naturalness.  
- **Coherence** — logical consistency.  
- **Faithfulness** — factual accuracy.  
- **Relevance** — task appropriateness.

Evaluators often rate outputs on a **Likert scale (1–5)**.

Hybrid approaches combine automatic metrics (BLEU, ROUGE) with selective human checks.

---

## 4.7 Error Analysis and Explainability

Metrics alone do not tell *why* a model fails.  
Qualitative analysis helps:
- Examine confusion matrices  
- Visualize embeddings (t-SNE, PCA)  
- Use explainability tools like **LIME** or **SHAP**

---

## 4.8 Summary Table

| Task Type | Metric | Focus |
|:--|:--|:--|
| Classification | Accuracy, F1 | Discrete labels |
| Sequence labeling | Entity-level F1 | Structure prediction |
| Translation | BLEU, METEOR | N-gram overlap |
| Summarization | ROUGE-L | Content recall |
| Language modeling | Perplexity | Predictive fluency |

---

## 4.9 Key References

- Papineni, K. et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.*  
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*  
- Zhang, T. et al. (2019). *BERTScore: Evaluating Text Generation with BERT.*  
- Tjong Kim Sang, E. F., & De Meulder, F. (2003). *CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.*

---

## ✅ Summary

> Evaluation is the lens through which we understand model performance.  
> The right metric depends on the task: from discrete label accuracy to semantic similarity and human-rated quality.  
> A good NLP practitioner combines **quantitative metrics** with **qualitative insights** for meaningful model assessment.

