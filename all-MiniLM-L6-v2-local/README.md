# Adobe Hackathon: Persona-Driven Document Intelligence

This project is a solution for the Adobe "Connecting the Dots" Hackathon. It transforms static PDFs into intelligent, queryable resources by first understanding their structure and then analyzing their content based on a user's specific persona and goals.

The system performs two primary tasks:
1. **Round 1A: High-Precision Outline Extraction:** Analyzes a raw PDF and generates a structured, hierarchical outline (Title, H1, H2, H3).
2. **Round 1B: Persona-Driven Intelligence:** Uses the extracted outline to rank document sections according to their relevance to a user's role and task, and then provides granular, summarized insights from the most important sections.

---

## 🚀 Key Features

- **High-Accuracy Outline Extraction:** Multi-pass, heuristic-based pipeline to identify headings without relying solely on font sizes.
- **Precise Document Sectioning:** Uses accurate heading bounding boxes to extract clean sections.
- **Advanced Hybrid Ranking:** Combines:
  - **Semantic Search:** Understands meaning & intent using a `sentence-transformer` model.
  - **Lexical Search (BM25):** Matches important keywords for completeness.
- **Granular Sub-Section Analysis:** Summarizes most relevant sentences into `refined_text`.
- **Fully Offline & Optimized:** Runs on CPU without internet, packaged in Docker.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **PDF Processing:** `PyMuPDF` (fitz)
- **Semantic Analysis:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Lexical Analysis:** `rank_bm25`
- **Text Processing:** `nltk`
- **Other Libraries:** `numpy`, `scikit-learn`
- **Containerization:** Docker

---

## ⚙️ Methodology & Pipeline

### Stage 1: Structural Outline Extraction
- **Body Text Identification:** Detects main body font style by word count.
- **Heading Identification:** Finds headings via font size, boldness, and short length.
- **Hierarchical Classification:** Assigns H1, H2, H3 via font features and numbering schemes.

### Stage 2: Persona-Driven Intelligence
Uses the structured outline to deliver user-specific insights.

```mermaid
graph TD
    A[Input: PDFs, Persona, Job] --> B{Stage 1: Outline & Sectionize};
    B --> C[All Document Sections];
    A --> D{Advanced Query Processing};
    D --> E[Semantic Vector];
    D --> F[Keyword Set];

    subgraph "Hybrid Scoring Engine"
        C & E --> G[Semantic Score (Cosine Similarity)];
        C & F --> H[Lexical Score (BM25)];
        G & H --> I[Weighted Hybrid Score α*S + (1-α)*L];
    end

    I -- Rank Sections by Score --> J[Top-Ranked Sections];
    subgraph "Sub-Section Analysis"
         J -- For each Top Section --> K[Find most similar sentences];
         K --> L[Generate Refined Summary];
    end
    J & L --> M[Format Final JSON Output];

## 🐳 Quick Start (Build & Run with Docker)

To run the project completely offline inside Docker, simply run:

```bash
docker build --platform linux/amd64 -t mysolution:latest . \
  && mkdir -p input output \
  && docker run --rm \
       -v $(pwd)/input:/app/input \
       -v $(pwd)/output:/app/output \
       --network none \
       mysolution:latest
