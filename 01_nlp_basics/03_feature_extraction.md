# 🧮 3. Feature Extraction in NLP

Once text is preprocessed, the next step is to **represent it numerically** so that algorithms can process it.  
This transformation — called **feature extraction** or **vectorization** — converts text into mathematical features while preserving as much semantic meaning as possible.

---

## 3.1 Why Representation Matters

Machine learning models operate on **vectors**, not words.  
Thus, the quality of textual representations directly influences model performance.

Desirable properties:
- Capture **semantic similarity** (e.g., “dog” ≈ “puppy”)  
- Preserve **contextual information**  
- Be **efficient and scalable**

---

## 3.2 Bag of Words (BoW)

The simplest form of text representation.

### Idea:
Each document is represented by a vector of word counts.

**Example:**
Corpus → {“NLP is fun”, “NLP is powerful”}

| Word | Doc1 | Doc2 |
|:--|:--:|:--:|
| NLP | 1 | 1 |
| is | 1 | 1 |
| fun | 1 | 0 |
| powerful | 0 | 1 |

Each document becomes a vector:  
$$\text{Doc1} = [1, 1, 1, 0], \quad \text{Doc2} = [1, 1, 0, 1]$$

### Limitations
- Ignores word order and context.  
- Sparse high-dimensional vectors.  
- Treats all words as equally important.

---

## 3.3 Term Frequency–Inverse Document Frequency (TF-IDF)

Improves on BoW by weighting words by their **importance** in a corpus.

### Formula:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

where  
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}}, \quad
\text{IDF}(t) = \log\frac{N}{1 + n_t}$$

- \(f_{t,d}\): frequency of term *t* in document *d*  
- \(N\): total number of documents  
- \(n_t\): number of documents containing *t*

**Intuition:**  
Common words across documents get low weights, rare yet significant words get high weights.

**Example:**
If “NLP” appears in every document, its IDF is low.  
If “transformer” appears rarely, its IDF is high → more informative.

**Reference:** Salton & Buckley (1988), *Term-weighting approaches in automatic text retrieval.*

---

## 3.4 N-grams and Phrase Features

**Definition:** Consecutive sequences of *n* words.

Examples:
- 1-gram: “language”  
- 2-gram: “natural language”  
- 3-gram: “language processing system”

**Purpose:** Capture short-range context and collocations.  
However, the vocabulary grows exponentially with *n* → sparsity increases.

$$|\text{Vocab}_{n\text{-gram}}| \approx |V|^n$$

**Use case:** Text classification and sentiment analysis (e.g., “not good” vs “good”).

---

## 3.5 Word Embeddings — Distributed Representations

To overcome sparsity and capture meaning, we map words to **dense vectors** in a continuous space.

### Conceptual shift:
Instead of “counting” words → we **learn** representations through training objectives.

**Goal:** Similar words have similar vectors.  
$$\text{sim}(\mathbf{w}_i, \mathbf{w}_j) = \frac{\mathbf{w}_i \cdot \mathbf{w}_j}{\|\mathbf{w}_i\|\|\mathbf{w}_j\|}$$

---

### 3.5.1 Word2Vec

Proposed by Mikolov et al. (2013).  
Two main architectures:
- **CBOW (Continuous Bag of Words):** Predicts a word from its context.  
- **Skip-gram:** Predicts context words from a target word.

**Training Objective:**
$$\max_\theta \sum_{t=1}^T \sum_{-c \le j \le c, j \ne 0} 
\log P(w_{t+j} | w_t; \theta)$$

**Key insight:** Context defines semantics.  
Words appearing in similar contexts → similar embeddings.

**Famous result:**  
$$\text{vec}("king") - \text{vec}("man") + \text{vec}("woman") \approx \text{vec}("queen")$$

**Reference:** Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space.*

---

### 3.5.2 GloVe (Global Vectors)

**Idea:** Use global co-occurrence statistics rather than local context windows.

Objective:
$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$
where \(X_{ij}\) is the co-occurrence count of words *i* and *j*.

**Reference:** Pennington, Socher, & Manning (2014), *GloVe: Global Vectors for Word Representation.*

---

## 3.6 Contextual Embeddings (Transition to Neural NLP)

Traditional embeddings (Word2Vec, GloVe) are **static**:  
each word has one vector, regardless of context.

Example:  
“bank” → same vector in “river bank” and “central bank”.

**Contextual embeddings (e.g., ELMo, BERT)** dynamically generate vectors based on surrounding words — a key leap toward **language models**.

---

## 3.7 Visualizing Embeddings (Conceptually)

Imagine a 2-D projection of high-dimensional embeddings:  
- Clusters of semantically similar words: `cat`, `dog`, `puppy`.  
- Linear relationships: gender (“king”→“queen”), tense (“run”→“ran”).  

Dimensionality reduction methods: PCA, t-SNE, UMAP.

---

## 3.8 Summary Table

| Method | Representation Type | Context-Aware | Sparsity | Interpretability |
|:--|:--|:--:|:--:|:--:|
| Bag of Words | Frequency counts | ❌ | High | ✅ |
| TF-IDF | Weighted counts | ❌ | High | ✅ |
| Word2Vec / GloVe | Dense embeddings | ❌ | Low | ⚙️ |
| Contextual (BERT, GPT) | Neural contextual embeddings | ✅ | Low | ⚙️⚙️ |

---

## 3.9 Key References

- Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.*  
- Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space.*  
- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.*  
- Peters, M. E. et al. (2018). *Deep contextualized word representations (ELMo).*

---

## ✅ Summary

> Feature extraction is the bridge between raw text and machine understanding.  
> The field has evolved from **count-based** to **dense vector-based** representations, setting the stage for contextual embeddings and large language models.
