# Adobe Hackathon: Persona-Driven Document Intelligence

This project is a solution for the Adobe "Connecting the Dots" Hackathon. It transforms static PDFs into intelligent, queryable resources by first understanding their structure and then analyzing their content based on a user's specific persona and goals.

The system performs two primary tasks:
1.  **Round 1A: High-Precision Outline Extraction:** It analyzes a raw PDF and generates a structured, hierarchical outline (Title, H1, H2, H3).
2.  **Round 1B: Persona-Driven Intelligence:** It uses the extracted outline to rank document sections according to their relevance to a user's role and task, and then provides granular, summarized insights from the most important sections.

---

## 🚀 Key Features

* **High-Accuracy Outline Extraction:** Uses a multi-pass, heuristic-based pipeline to robustly identify headings without relying solely on font sizes.
* **Precise Document Sectioning:** Leverages accurate heading bounding boxes to extract clean, meaningful sections.
* **Advanced Hybrid Ranking:** Combines two powerful techniques for relevance scoring:
    * **Semantic Search:** Understands the *meaning* and *intent* behind the user's query using a `sentence-transformer` model.
    * **Lexical Search (BM25):** Matches specific *keywords* to ensure critical terms are not missed.
* **Granular Sub-Section Analysis:** Instead of returning whole sections, it performs extractive summarization to pinpoint and deliver the most relevant sentences, creating a truly `refined_text`.
* **Fully Offline & Optimized:** The entire pipeline runs on a CPU without any network access, with all models and dependencies included in the Docker container.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3
* **PDF Processing:** `PyMuPDF` (fitz) for fast and detailed text/layout extraction.
* **Semantic Analysis:** `sentence-transformers` with the `all-MiniLM-L6-v2` model.
* **Lexical Analysis:** `rank_bm25` for efficient keyword-based scoring.
* **Text Processing:** `nltk` for sentence tokenization and stop word removal.
* **Core Libraries:** `numpy`, `scikit-learn`
* **Containerization:** `Docker`

---

## ⚙️ Methodology & Pipeline

Our solution is a sophisticated pipeline that first establishes a structural foundation and then builds semantic intelligence on top of it.

### Stage 1: Structural Outline Extraction

The system first parses the PDF using a `PyMuPDF`-based pipeline. Unlike simple line-by-line methods, it analyzes logical text blocks to understand the document's typography.

1.  **Body Text Identification:** It determines the primary body text style by finding the font style associated with the highest total word count.
2.  **Heading Identification:** It identifies headings as text blocks that are visually distinct (e.g., larger, bolder) and structurally distinct (shorter, single-line) from the body text.
3.  **Hierarchical Classification:** It assigns H1, H2, and H3 levels based on font size, boldness, and a final refinement pass using numbering schemes (e.g., "2.1 Topic").

### Stage 2: Persona-Driven Intelligence

This stage uses the structured outline to perform a deep analysis tailored to the user.

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

###   Dockerfile Instructions
For the solution to work in an offline Docker container, the NLTK data must be downloaded during the build process
# Before your main CMD or ENTRYPOINT
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

### Building the Docker Image
Navigate to the root directory of the project (where the Dockerfile is located) and run the build command:
docker build --platform linux/amd64 -t mysolution:latest .


### Running the Solution
We will run your solution using the command specified in the hackathon brief. Your solution should automatically process all collection subdirectories inside the /app/input volume and place the corresponding output in /app/output

# Ensure you have an 'input' and 'output' directory in your current path
# Place your test cases (e.g., 'Test Case 1', 'Test Case 2') inside the 'input' directory
mkdir -p input output

# Example command to run the container
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolution:latest