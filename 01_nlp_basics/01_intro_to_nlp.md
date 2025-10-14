# 🩵 1. Introduction to Natural Language Processing (NLP)

Natural Language Processing (NLP) is the field of Artificial Intelligence concerned with enabling computers to understand, interpret, and generate human language.  
It lies at the intersection of **linguistics**, **computer science**, and **machine learning**.

---

## 1.1 What Is NLP?

NLP involves designing algorithms and models that can process text or speech to perform tasks such as:

- **Understanding** → sentiment analysis, named entity recognition (NER)  
- **Generation** → chatbots, text summarization  
- **Translation** → automatic language translation  
- **Information retrieval** → search engines, question answering

In short, NLP seeks to **bridge the gap between human communication and machine representation**.

---

## 1.2 Core Challenges

1. **Ambiguity** — Words or phrases can have multiple meanings.  
   - *Example:* “I saw the man with the telescope.”  
2. **Context-dependence** — Meaning often depends on sentence or world context.  
3. **World knowledge** — Understanding text often requires external or implicit knowledge.  
4. **Pragmatics** — Intention, tone, and indirect meaning are hard to model.  

---

## 1.3 Evolution of NLP

| Era | Approach | Key Characteristics |
|:--|:--|:--|
| **1950–1980s** | Symbolic / Rule-based | Grammar rules, handcrafted lexicons |
| **1990s** | Statistical NLP | Probabilistic models, n-grams, HMMs |
| **2010s** | Neural NLP | Word embeddings, RNNs, CNNs |
| **2020s** | Large Language Models (LLMs) | Transformers, pretraining, in-context learning |

**Schematic description:**  
Imagine a timeline from left to right:
- Early symbolic systems (rules → logic trees).  
- Then probabilistic models (Markov chains → HMMs).  
- Then neural architectures (layers → embeddings → attention).  
- Finally, LLMs as unified, pretrained text generators.

---

## 1.4 Applications of NLP

- **Information Extraction (IE)** — Extracting entities, relations, and facts from text.  
- **Machine Translation (MT)** — Converting text between languages.  
- **Question Answering (QA)** — Finding relevant answers to user queries.  
- **Text Summarization** — Generating concise summaries.  
- **Conversational AI** — Chatbots, virtual assistants.  
- **Speech-related NLP** — ASR (automatic speech recognition) and TTS (text-to-speech).

---

## 1.5 NLP Pipeline Overview

**Conceptual flow diagram (described):**  
Raw Text → Preprocessing → Feature Extraction → Modeling → Evaluation → Deployment

### Steps:
1. **Text preprocessing:** tokenization, normalization, stopword removal.  
2. **Feature extraction:** TF-IDF, embeddings.  
3. **Modeling:** classification, sequence labeling, or generation.  
4. **Evaluation:** metrics such as accuracy, F1, BLEU, ROUGE.  
5. **Deployment:** serving, monitoring, feedback loops.

---

## 1.6 Symbolic vs Statistical vs Neural NLP

| Aspect | Symbolic | Statistical | Neural |
|:--|:--|:--|:--|
| **Knowledge source** | Handcrafted rules | Data-driven probabilities | Learned representations |
| **Interpretability** | High | Medium | Low |
| **Scalability** | Low | Medium | High |
| **Data requirement** | Low | Medium | Very high |

---

## 1.7 Key References

- Chomsky, N. (1957). *Syntactic Structures.*  
- Jurafsky & Martin (2024). *Speech and Language Processing (3rd Ed. draft).*  
- Manning & Schütze (1999). *Foundations of Statistical NLP.*  
- Hirschberg & Manning (2015). *Advances in Natural Language Processing.*

---

## 1.8 Summary

> **NLP** is the foundation of modern language technology.  
> It evolved from symbolic rules to statistical models, and now to large pretrained neural networks.  
> Understanding this evolution helps interpret current trends in LLMs and RAG systems.
