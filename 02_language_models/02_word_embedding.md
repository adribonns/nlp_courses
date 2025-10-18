# 🧩 2. Word Embeddings

Traditional NLP models treated words as discrete symbols (e.g., "cat" ≠ "dog" ≠ "house"), with no notion of similarity.  
**Word embeddings** revolutionized this by mapping words to continuous vectors that encode **semantic and syntactic relationships**.

---

## 2.1 Motivation

The **distributional hypothesis** (Harris, 1954):

> “Words that occur in similar contexts tend to have similar meanings.”

### Example

| Word | Context words |
|:--|:--|
| cat | pet, animal, fur, purr |
| dog | pet, animal, fur, bark |

→ Vectors for *cat* and *dog* should be **close** in embedding space.

---

## 2.2 From One-Hot to Dense Representations

### One-Hot Encoding
- Each word = vector of length $|V|$ (vocabulary size)
- Exactly one “1” and the rest “0”

Example for a 5-word vocab:  
“cat” → [0, 0, 1, 0, 0]

**Problems:**
- Sparse and memory-inefficient  
- No semantic relationship (cosine similarity = 0 between all pairs)

### Embedding Representation
Instead, each word is a **dense vector** of dimension $d$ (typically 100–1000):
$$
\text{cat} \rightarrow \mathbf{v}_{\text{cat}} = [0.15, -0.27, ..., 0.73]
$$

These vectors are **learned** from data such that similar words have nearby vectors.

---

## 2.3 Word2Vec (Mikolov et al., 2013)

Word2Vec introduced two architectures to learn embeddings efficiently:

1. **Continuous Bag of Words (CBOW):**
   - Predict target word from context words.
   - Example: input = “the ___ sat on the mat” → predict “cat”

   Objective:
   $$
   \max \sum_{t} \log P(w_t | w_{t−c}, ..., w_{t+c})
   $$

2. **Skip-Gram:**
   - Predict context words given the target.
   - Example: input = “cat” → predict “the”, “sat”, “on”, etc.

   Objective:
   $$
   \max \sum_{t} \sum_{−c \le j \le c, j \ne 0} \log P(w_{t+j} | w_t)
   $$

### Output Probability
Both models use a softmax layer:
$$
P(w_o | w_i) = \frac{e^{v_{w_o} \cdot v_{w_i}}}{\sum_{w=1}^{|V|} e^{v_w \cdot v_{w_i}}}
$$

To avoid large vocabulary costs, **negative sampling** or **hierarchical softmax** is used.

---

## 2.4 The Geometry of Meaning

Word2Vec embeddings capture linguistic structure via **vector arithmetic**:

$$
\text{King} - \text{Man} + \text{Woman} \approx \text{Queen}
$$

Cosine similarity is commonly used:
$$
\text{cosine\_sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}
$$

---

## 2.5 GloVe (Global Vectors, Pennington et al., 2014)

Word2Vec is local (context-based), whereas **GloVe** leverages **global co-occurrence statistics**.

### Core Idea
Learn embeddings that reproduce observed co-occurrence ratios:
$$
w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log X_{ij}
$$

Where:
- $X_{ij}$ = number of times word $j$ appears in context of $i$

Optimized via weighted least squares:
$$
J = \sum_{i,j} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

---

## 2.6 Evaluation of Embeddings

Common evaluation tasks:
- **Word similarity:** correlation with human-rated similarity (e.g., WordSim-353)
- **Word analogy:** “man : king :: woman : ?”
- **Downstream tasks:** sentiment analysis, NER, etc.

---

## 2.7 Limitations of Static Embeddings

1. **Polysemy:** one vector per word → no sense disambiguation  
   - “bank” (river vs finance)  
2. **Context ignorance:** same embedding for all uses  
3. **Bias propagation:** embeddings reflect data biases  
   - e.g., “doctor – man + woman ≈ nurse”

---

## 2.8 Contextual Embeddings

Modern models (ELMo, BERT, GPT) produce **context-dependent vectors**.

For example:
- “bank” in “river bank” vs “bank loan” → different embeddings

These are derived from **hidden states** of deep language models, capturing syntax and semantics dynamically.

---

## 2.9 Visualization and Interpretation

Visualizing embeddings via **t-SNE** or **UMAP** reveals clusters:
- Animals, professions, emotions, etc.
- Semantic axes (e.g., gender, tense, country–capital)

---

## 2.10 Summary Table

| Model | Type | Context | Key Idea |
|:--|:--|:--|:--|
| One-Hot | Sparse | None | Discrete representation |
| Word2Vec (CBOW/Skip-gram) | Dense | Local window | Predict context or target |
| GloVe | Dense | Global | Co-occurrence statistics |
| Contextual (BERT, GPT) | Dynamic | Sentence-level | Contextualized meaning |

---

## 🧾 References

- Harris, Z. (1954). *Distributional Structure.*  
- Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space.*  
- Pennington, J. et al. (2014). *GloVe: Global Vectors for Word Representation.*  
- Peters, M. et al. (2018). *Deep Contextualized Word Representations (ELMo).*

---

## ✅ Summary

> Word embeddings transformed NLP from symbolic processing to geometric reasoning.  
> By learning distributed representations, models gained the ability to reason over similarity, analogy, and meaning — paving the way for contextual embeddings and transformers.

