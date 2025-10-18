# 🔤 2. Text Preprocessing

Before any model can understand or generate text, raw language must be **cleaned, structured, and normalized**.  
Text preprocessing transforms unstructured text into a form suitable for computational analysis.

---

## 2.1 Why Preprocessing Matters

Raw text contains noise: punctuation, capitalization, typos, and linguistic variations.  
Effective preprocessing improves both the **accuracy** and **efficiency** of NLP models.

Typical goals:
- Standardize input  
- Reduce vocabulary size  
- Improve feature extraction consistency

---

## 2.2 NLP Preprocessing Pipeline (Conceptual Diagram)

→ Tokenization
→ Normalization
→ Stopword Removal
→ Stemming / Lemmatization
→ Vectorization


Each step refines the representation of text to help algorithms capture meaning more effectively.

---

## 2.3 Tokenization

**Definition:** Splitting text into meaningful units called *tokens* (words, subwords, or characters).

### Types of tokenization
| Type | Example Input | Example Tokens |
|:--|:--|:--|
| **Word-level** | “Natural Language Processing” | ["Natural", "Language", "Processing"] |
| **Subword-level** | “unhappiness” | ["un", "##happiness"] |
| **Character-level** | “NLP” | ["N", "L", "P"] |
| **Sentence-level** | “Hello world. How are you?” | ["Hello world.", "How are you?"] |

### Notes
- Tokenization is **language-dependent**.  
- Modern LLMs (e.g., GPT, BERT) use **subword tokenization** like BPE (Byte-Pair Encoding) or WordPiece to handle rare words and balance vocabulary size.

**Formula (approximate vocabulary compression):**
$$V_{new} = V_{old} - n + 1$$
where \(n\) is the number of symbol pairs merged during BPE training.

**Key references:**  
- Sennrich et al. (2016) — *Neural Machine Translation of Rare Words with Subword Units.*

---

## 2.4 Normalization

Normalization reduces variation in the text without altering meaning.

### Common steps:
- Lowercasing: “Apple” → “apple”  
- Removing punctuation and special characters  
- Handling numbers: “2025” → “<NUM>”  
- Unicode normalization (e.g., removing diacritics: “café” → “cafe”)  
- Expanding contractions: “don’t” → “do not”

> ⚠️ Be careful: excessive normalization can remove meaningful distinctions (e.g., “US” vs “us”).

---

## 2.5 Stopword Removal

Stopwords are common words (e.g., *the, is, of, and*) that usually carry limited semantic content.

- Common libraries: `NLTK`, `spaCy`, or custom domain-specific lists.  
- However, **contextual models** like BERT or GPT no longer need explicit stopword removal — they learn importance implicitly.

---

## 2.6 Stemming and Lemmatization

### 🔹 **Stemming**
Heuristic process that chops off word endings to reduce them to their base form.  
Example:  
`running → run`, `flies → fli`

**Algorithms:** Porter Stemmer, Snowball Stemmer.

Stemming formula (simplified):
$$f_{stem}(word) = prefix(word, k)$$
where \(k\) truncates suffixes according to rule-based heuristics.

---

### 🔹 **Lemmatization**
Linguistic process using vocabulary and morphology to find the canonical form (*lemma*).

Example:  
`am, are, is → be`  
`better → good`

Lemmatization considers:
- Part-of-speech tagging  
- Morphological analysis  

**Tools:** WordNetLemmatizer (`NLTK`), spaCy lemmatizer.

---

## 2.7 Handling Noise and Special Tokens

| Issue | Example | Typical Solution |
|:--|:--|:--|
| URLs | `https://openai.com` | Replace with `<URL>` |
| Numbers | `123` | `<NUM>` |
| Emojis | 🙂 😢 | Map to sentiment or remove |
| Typos | “goood” | Spell correction models |
| Repeated chars | “soooo” | Normalization rules |

---

## 2.8 Advanced Text Cleaning

- **Named Entity Replacement:** Replace entities with placeholders → `John works at Google.` → `<PERSON> works at <ORG>.`  
- **Handling code-mixed text:** Separate tokens from multiple languages.  
- **Language detection:** Detect and normalize multilingual inputs.

---

## 2.9 Summary Diagram (Textually Described)

1. Input: “The cats are running faster than the dogs!”  
2. Tokenization → ["The", "cats", "are", "running", "faster", "than", "the", "dogs", "!"]  
3. Normalization → ["cats", "running", "faster", "dogs"]  
4. Lemmatization → ["cat", "run", "fast", "dog"]

---

## 2.10 Key References

- Porter, M. F. (1980). *An algorithm for suffix stripping.*  
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.*  
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.*

---

## ✅ Summary

> Preprocessing transforms raw linguistic data into structured, analyzable input.  
> While early NLP relied heavily on manual cleaning, modern neural models integrate tokenization and normalization internally — yet understanding these steps remains essential for data quality and interpretability.
