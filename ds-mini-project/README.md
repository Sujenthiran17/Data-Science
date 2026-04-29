# Data Science Mini-Project: Medical RAG & Data Transformation

This repository contains two primary components: a **Medical Retrieval-Augmented Generation (RAG) Pipeline** and a **Billing & Renewal Data Transformation Pipeline**.

## 🚀 Features

### 1. Medical RAG Pipeline
An end-to-end system to answer medical questions using real-time data from PubMed.
- **PubMed Integration**: Searches and fetches latest scientific abstracts via NCBI E-utilities.
- **Vector Database**: Utilizes **ChromaDB** for persistent local storage of document embeddings.
- **AI-Powered**: Uses **Google Gemini** (`gemini-embedding-001`) for semantic search and (`gemini-flash-latest`) for context-aware answer generation.
- **Anti-Hallucination**: Implementation includes strict prompt engineering to ensure answers are based only on retrieved context.

### 2. Billing & Renewal Transformation
A suite of data processing scripts for cleaning and merging complex financial datasets.
- **Data Cleaning**: Specialized scripts for billings and renewals (`cleaning_billings.ipynb.py`, `clean_joined.ipynb.py`).
- **Complex Joins**: Notebooks for joining datasets based on correlation and multiple column sets.
- **Modular Design**: Structured pipeline stages from ingestion to final transformation.

---

## 📁 Project Structure

```text
ds-mini-project/
├── 00 ingestion/           # Data loading scripts
├── 01 exploration/         # Exploratory Data Analysis (EDA)
├── 02 transformation/      # Data cleaning and joining logic
├── chroma_db_data/         # Local ChromaDB persistent storage
├── medical_rag_pipeline.py  # Main entry point for the RAG system
├── f_renewal.py             # Renewal processing logic
└── README.md                # Project documentation
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Gemini API Key

### Setup
1. Clone the repository.
2. Install required dependencies:
   ```bash
   pip install chromadb google-generativeai requests pandas
   ```
3. Set your Gemini API key in `medical_rag_pipeline.py` (or set as an environment variable).

---

## 📖 Usage

### Running the Medical RAG Pipeline
To ask a medical question and get an answer backed by PubMed research:
```bash
python medical_rag_pipeline.py
```
By default, the script queries for *"Latest treatment for diabetes"*. You can modify the `user_query` variable in the `if __name__ == "__main__":` block.

### Data Transformation
The transformation scripts in `02 transformation/` can be run to process raw billing and renewal data into a joined, clean format suitable for analysis.

---

## ⚠️ Disclaimer
This project is for research and educational purposes. The medical information generated is based on PubMed abstracts and should **not** be considered professional medical advice. Always consult a qualified healthcare provider for medical decisions.