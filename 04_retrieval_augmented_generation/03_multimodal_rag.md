# 🖼️ 3. Multimodal Retrieval-Augmented Generation (RAG-Vision and ColPali)

> *“RAG doesn’t have to be text-only. When knowledge lives in images, charts, or PDFs — multimodal retrieval becomes essential.”*

Traditional RAG retrieves *text chunks*.
But in many domains (medicine, science, documents, UX, robotics), **visual data** carries key context.
This motivates **multimodal RAG (MM-RAG)** — systems that can *retrieve and reason* over **text + images (and sometimes video, audio, tables)**.

---

## 3.1 Motivation

**Problem:**  
A legal document, a scientific article, or an invoice often encodes information visually —  
layout, figures, tables, diagrams.

Text-only retrieval ignores these modalities → missing critical evidence.

**Solution:**  
Multimodal RAG =  
→ *multimodal retriever* (text + vision embeddings)  
→ *multimodal generator* (vision-language LLM, VLM).

---

## 3.2 Architecture Overview

Query (text or image)
↓
1️⃣ Multimodal embedding (e.g., CLIP, SigLIP, ColPali)
↓
2️⃣ Multimodal index (vector DB with text + image vectors)
↓
3️⃣ Retriever: find semantically similar items
↓
4️⃣ Generator (VLM): fuse retrieved visual + textual context to produce grounded answer


Equation form:
$$
\text{Answer} = f_{\text{VLM}}(q, \{r_i\}_{i=1..k})
$$
where \( r_i \) are multimodal retrieved contexts (text, image, or both).

---

## 3.3 Multimodal Embedding Models

| Model | Type | Objective | Reference |
|:--|:--|:--|:--|
| **CLIP** | Dual-encoder (image–text) | Contrastive: align image & caption embeddings | Radford et al., 2021 |
| **ALIGN** | Dual-encoder (noisy web data) | Large-scale contrastive learning | Jia et al., 2021 |
| **BLIP / BLIP-2** | Vision–language (encoder–decoder) | Pretrain contrastive + captioning | Li et al., 2022 |
| **SigLIP** | Similar to CLIP with sigmoid loss | Better gradient stability | Zhai et al., 2023 |
| **ColPali** | ColBERT-based multimodal retriever | Late-interaction retrieval for text–image pairs | Barlas et al., 2024 |

---

## 3.4 ColPali: Dense Retrieval for Vision-Language

**ColPali** (2024) introduces **late-interaction retrieval** for multimodal documents.  
It extends the **ColBERT** architecture to **images + text**.

### 🧠 Idea
Instead of pooling the entire embedding into one vector, it keeps **per-token (or per-patch)** embeddings and compares them at query time.

For a query \( q \) and document \( d \):

$$
S(q, d) = \sum_{i \in q} \max_{j \in d} \langle E_q[i], E_d[j] \rangle
$$

This allows **fine-grained matching** — e.g., a question about a chart can align with a specific region of an image.

### ⚙️ Components
- **Visual encoder**: Vision Transformer (ViT)
- **Text encoder**: BERT-like transformer
- **Late interaction layer**: token-wise similarity matrix
- **Index**: FAISS or ColBERT-specific ANN structure

### 💡 Advantage
- More precise alignment between text query and image content  
- Retains efficiency compared to full cross-encoder models  
- Works on multimodal PDFs, figures, diagrams

---

## 3.5 Building a Multimodal RAG Pipeline

Typical steps:

1️⃣ **Chunk multimodal data**  
   - Extract text via OCR (e.g., `Tesseract`, `LayoutLMv3`)  
   - Extract images or visual regions (bounding boxes, screenshots)  
   - Associate text + image pairs as single “multimodal chunks”.

2️⃣ **Embed**  
   - Use CLIP or SigLIP to encode both query and document sides.  
   - Optionally project into shared embedding space.

3️⃣ **Index**  
   - Vector DB (FAISS, Milvus, Qdrant) stores multimodal vectors.  
   - Use metadata tags for modality (`"image"`, `"caption"`, `"layout"`, etc.).

4️⃣ **Retrieve**  
   - Retrieve multimodal candidates using cosine similarity:
     $$
     \text{sim}(E_q, E_d) = \frac{E_q \cdot E_d}{\|E_q\|\|E_d\|}
     $$
   - Possibly ensemble with textual retriever for fallback.

5️⃣ **Re-rank**  
   - Apply cross-modal re-ranker (e.g., BLIP-2, ColPali cross-encoder).

6️⃣ **Generate**  
   - Feed retrieved visuals + captions to a **Vision–Language LLM (VLM)** such as:
     - GPT-4V / Claude 3 Opus  
     - LLaVA / MiniGPT-4  
     - Kosmos-2 / Flamingo

---

## 3.6 Examples of Vision–RAG Applications

| Domain | Example Use |
|:--|:--|
| **Document AI** | Retrieve visual evidence from PDFs (invoices, forms) |
| **Scientific QA** | Extract information from charts, plots, equations |
| **E-commerce** | Visual search: “show me jackets similar to this image” |
| **UX / product QA** | Answer questions from screenshots |
| **Medical** | Image-assisted reasoning (X-rays, histopathology) |

---

## 3.7 Key Models and Systems

| System | Description | Reference |
|:--|:--|:--|
| **ColPali** | Multimodal late-interaction retriever for docQA | Barlas et al., 2024 |
| **V-RAG** | Unified retrieval from vision + text sources | Han et al., 2024 |
| **MM-RAG (Meta)** | Multimodal knowledge retrieval for LLaVA-Next | Meta AI, 2024 |
| **Visual-RAG (Microsoft)** | Vision-aware retrieval pipeline with SigLIP + BLIP-2 | 2024 |
| **LLaVA-RAG** | Combines LLaVA as generator with CLIP retriever | 2024 |

---

## 3.8 Design Considerations

### 🧩 Modality Fusion

Two main strategies:
- **Early fusion** — fuse image and text embeddings before retrieval.  
  → Simpler but harder to tune.
- **Late fusion** — retrieve separately and merge results (similar to ensemble retrievers).  
  → More modular, better interpretability.

### ⚖️ Trade-offs

| Aspect | Text-only RAG | Multimodal RAG |
|:--|:--|:--|
| Context length | Shorter | Larger (includes visual tokens) |
| Embedding size | Small (768–1024) | High (1024–4096) |
| Latency | Lower | Higher |
| Recall on visual info | Poor | Excellent |

---

## 3.9 Multimodal RAG in Practice (Example)

User: "What is the correlation between GDP and CO₂ in this chart?"
↓
1️⃣ Encode query (text)
2️⃣ Retrieve similar charts (image embeddings)
3️⃣ Retrieve text describing them (captions)
4️⃣ Feed to VLM (e.g., LLaVA) with chart image + retrieved context
↓
Answer: "The chart shows a strong positive correlation between GDP and CO₂ emissions."


This pipeline uses **cross-modal retrieval** + **visual reasoning**.

---

## 3.10 Challenges and Research Directions

- Efficient multimodal indexing (joint text+image compression)
- Cross-domain generalization (e.g., medical vs. scientific vs. everyday images)
- Grounding answers visually (faithfulness metrics)
- Evaluation frameworks for visual RAG (few exist today)
- Integrating video and temporal reasoning (e.g., Video-RAG)

---

## ✅ Summary

> **Multimodal RAG** extends retrieval to images and structured visuals.  
> It blends **retrieval**, **vision-language understanding**, and **generation** into one reasoning loop.

Key ideas:
- Use **contrastive embedding** (CLIP, ColPali) for image–text alignment.  
- Index multimodal chunks together.  
- Use **VLMs** as readers/generators.  
- Fusion and re-ranking are crucial for multimodal fidelity.

---

## 📚 References

- Barlas et al. (2024). *ColPali: Efficient Multimodal Late Interaction Retrieval.*  
- Han et al. (2024). *V-RAG: Vision-Retrieval-Augmented Generation.*  
- Li et al. (2022). *BLIP: Bootstrapped Language–Image Pretraining.*  
- Zhai et al. (2023). *SigLIP: Improving CLIP with Sigmoid Loss.*  
- Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision (CLIP).*  
- Meta AI (2024). *MM-RAG: Multimodal Retrieval-Augmented Generation.*


