# Hybrid RAG Support Bot (Advanced Retrieval System)

## 📖 Project Overview
This project implements a **Production-Grade RAG (Retrieval-Augmented Generation) System** designed to answer user queries based on technical documentation. 

Unlike standard RAG implementations that treat documents as unstructured text blobs, this system employs **Smart Ingestion** to parse the document's physical structure (Chapters/Sections) and utilizes **Metadata Filtering** to drastically improve retrieval accuracy and reduce hallucinations.

**Core Tech Stack:** Python, LangChain, ChromaDB, Ollama, PDFPlumber.

---

## 🚀 Key Architecture Features

### 1. Smart Ingestion Pipeline (`ingest.py`)
*   **Structure-Aware Parsing:** Uses Regex heuristic logic to detect distinct sections (e.g., "12. TOKEN REMUNERATION", "4. ELIGIBILITY").
*   **Layout Preservation:** Uses `pdfplumber` with `layout=True` to correctly parse complex tabular data (e.g., Lists of Domains/Durations).
*   **Metadata Enrichment:** Chunks are stored with precise metadata tags: `{"chapter": "ELIGIBILITY", "source": "manual.pdf"}`.

### 2. Hybrid Search Logic (`main.py`)
*   **Layer 1: Deterministic Routing (Rule-Based):** 
    *   Maps high-confidence keywords (e.g., "stipend", "salary", "certificate") directly to the specific source chapter.
    *   *Benefit:* 100% Accuracy and near-zero latency for known domain terms.
*   **Layer 2: Semantic Routing (AI-Based):** 
    *   Uses an LLM Router to determine the relevant chapter for complex or ambiguous queries where keywords fail.
*   **Layer 3: Global Fallback:** 
    *   Intelligently searches the entire document context if no specific intent is detected.

### 3. Performance Observability
*   Real-time console logging of **Retrieval Latency** (I/O time) vs. **Generation Latency** (Compute time) for every single query, allowing for backend bottleneck analysis.

---

## 🤖 Model Configuration

This project is currently configured to use **`llama3.2:1b`** (1 Billion Parameters). This decision was made to ensure the system runs smoothly on standard local hardware with limited VRAM, demonstrating capability without requiring enterprise-grade GPUs.

**For Better Results:**
If hardware permits (8GB+ VRAM), it is highly recommended to use the **`llama3`** (8B) model. The larger model size significantly improves reasoning capabilities on complex tabular data and subtle context nuances.

To use the stronger model:
1.  Run `ollama pull llama3`
2.  Update `main.py` (Line 9): `MODEL_NAME = "llama3"`

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   **Python 3.10+**
*   **Ollama** (Must be installed and running in the background)

### 2. Install Dependencies
Use the provided requirements file or install manually:
```bash
pip install -r requirements.txt
```
### 3. Download the Model
Ensure Ollama is running, then pull the default model:
code
```Bash
ollama pull llama3.2:1b
```
## 🏃‍♂️ How to Run
### Step 1: Ingestion (ETL Pipeline)
Run this script once. It parses the PDF, cleans the text, and populates the Vector Database (chroma_db_nic).
code
```Bash
python ingest.py
```
Note: The script looks for dii-summer-guidelines-april-2025.pdf. If not found, it falls back to manual.pdf.
Success Output: ✅ Ingestion Complete. System is ready.

### Step 2: Start the Bot (Runtime)
Launch the interactive query engine.
code
```Bash
python main.py
```
## 🔍 Validation Scenarios

To verify the **Hybrid Routing** logic and **Metadata Filtering**, test the following queries:

| Query | Expected Behavior (Log Output) |
| :--- | :--- |
| **"How much is the stipend?"** | `🎯 Router Decision: Filter Database by [chapter == 'TOKEN REMUNERATION']` |
| **"What are the eligibility criteria?"** | `🎯 Router Decision: Filter Database by [chapter == 'ELIGIBILITY']` |
| **"Where is the place of internship?"** | `🎯 Router Decision: Filter Database by [chapter == 'PLACE OF INTERNSHIP']` |
| **"Who is the Competent Authority?"** | Uses AI/Global Search (General Query) |

---

## 📊 Latency Metrics Output

The console output provides a clear breakdown of performance for backend optimization analysis:

```text
 📊 PERFORMANCE METRICS:
   - Metadata Filter: 12. TOKEN REMUNERATION
   - Retrieval Latency: 0.0220s   (Vector Search Time)
   - Generation Latency: 5.6023s  (LLM Inference Time)
   - Total Latency:      5.6243s
```
## 📂 Project Structure

```text
.
├── ingest.py           # ETL Script: Parsing, Cleaning, Embedding
├── main.py             # Runtime Script: Hybrid Router & RAG Pipeline
├── requirements.txt    # List of python dependencies
├── dii-summer-guidelines-april-2025.pdf    # Source Data (NIC Internship Guidelines)
└── chroma_db_nic/      # Local Vector Store (Auto-generated)
```
## 🔧 Troubleshooting

*   **Error: `Connection refused`**
    *   *Fix:* Make sure Ollama is running in the background. Open a terminal and run `ollama serve`.

*   **Error: `Model not found`**
    *   *Fix:* You have not downloaded the model yet. Run `ollama pull llama3.2:1b`.

*   **Error: `Database not found`**
    *   *Fix:* You must run the ingestion script once before starting the bot. Run `python ingest.py`.
