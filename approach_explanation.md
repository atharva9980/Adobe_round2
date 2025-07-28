
# `approach_explanation.md`

## 📘 Methodology for Persona-Driven Document Intelligence

Our solution is engineered to win by transforming static PDFs into dynamic, intelligent resources through a sophisticated, two-stage pipeline. The first stage (Round 1A) achieves unparalleled accuracy in structural analysis to create a reliable document outline. The second stage (Round 1B) leverages this precise structure to perform a persona-driven semantic analysis, extracting and ranking the most relevant content to directly address the user's job-to-be-done.

---

## 🔷 Round 1A: High-Precision Outline Extraction via Structural Analysis

To overcome the brittleness of traditional methods, we developed a novel multi-pass, structural analysis pipeline for outline extraction. Instead of analyzing documents line-by-line, our system understands structure much like a human would, by identifying the relationship between headings and the body text they introduce.

### ✅ Goal:
Convert a PDF into a reliable, hierarchical document outline using structure-aware logic.

### 🧠 Key Design:
Unlike brittle, line-by-line approaches, our method emulates human reading by understanding relationships between headings and their body content.

### 🔄 Four-Pass Pipeline:

1. **Logical Block Reconstruction**
   - Splits the PDF into paragraph-like blocks.
   - Preserves structural distinctions by isolating styled or fragmented text.

2. **High-Confidence Body Text Identification**
   - Identifies dominant font style based on **total word count** in paragraph blocks.
   - Robust against noise from forms or mixed formatting.

3. **Structural Heading Identification**
   - Strict criteria: short length and visual distinction (e.g., larger, bolder font).
   - Ensures only genuine headings are retained.

4. **Hierarchical Classification**
   - Classifies heading levels (H1–H4) based on font size and numbering patterns like `2.1 Topic`.

### 🏁 Scoring Criteria (1A):

| Metric | Justification |
|--------|---------------|
| **Heading Detection Accuracy (25 pts)** | Structural cues + context-aware filtering removes false positives. |
| **Performance (10 pts)** | Fast, CPU-bound with **0MB model size** using PyMuPDF. |
| **Multilingual Support (10 bonus pts)** | Language-agnostic; uses font/structure, not semantics. `ensure_ascii=False` enables Unicode. |



---

## 🔷 Round 1B: Persona-Driven Semantic Analysis

The accurate outlines from Round 1A are the bedrock of our intelligent analysis engine, directly enabling us to excel in the Round 1B scoring criteria.

### ✅ Goal:
Rank document sections by their relevance to a user's **persona and job-to-be-done**.

### 📥 Pipeline:

1. **Section & Subsection Extraction**
   -  The extracted outline is used to precisely segment each document into meaningful sections. The full, granular text content under each heading is captured.

2. **Semantic Embedding**
   - Converts both **user query** and **document sections** to vector embeddings.
   - Model: `all-MiniLM-L6-v2` (small, fast, offline, <100MB).

3. **Relevance Ranking**
   - Computes **cosine similarity** between query and sections.
   - Produces a **stack-ranked list** of relevant content.

### 🏁 Scoring Criteria (1B):

| Metric | Justification |
|--------|---------------|
| **Section Relevance (60 pts)** | Persona-contextual embeddings + precise sectioning enable high match accuracy. |
| **Subsection Relevance (40 pts)** | Clean outlines = high-quality, granular subsections for semantic analysis. |

---

## 📊 Performance & Scalability

### ⏱ Time Complexity:
- **1A:** `O(B)` where `B` = number of text blocks
- **1B:** `O(S * L)` where `S` = sections, `L` = avg section length

### 🧠 Space Complexity:
- ~90MB memory (sentence-transformer) + embeddings

### ⚙️ Scalability:
- Pipeline: `Extractor → Sectionizer → Ranker`
- Supports parallelism and microservices (e.g., semantic ranker on GPU).

---

## ⚠️ Limitations & 🔮 Future Work

| Limitation | Solution |
|-----------|----------|
| **Scanned PDFs** | Add OCR (e.g., Tesseract) as a preprocessing layer |
| **Complex layouts (magazines, etc.)** | Use hybrid ML-heuristic: CV model locates headings → passed to structured classifier |
| **Generic semantic model** | Fine-tune on domain-specific corpora (finance, medicine) for deeper context matching |

---

## ✅ Summary

This dual-phase approach brings together **structural precision** and **semantic depth** to deliver a system that is fast, scalable, multilingual, and highly relevant for real-world document intelligence tasks.
