# 🔬 ResearchLens AI

An AI-powered research intelligence system that lets you upload multiple research papers and understand them together — not one at a time.

Built with **LangChain**, **LangGraph**, **LLaMA 3.3 70B (via Groq)**, and **Streamlit**.

---

## ✨ Features

- 📥 **Multi-format ingestion** — Upload PDFs, DOCX files, or paste arXiv IDs
- 💬 **Chat across all papers** — Ask questions, get answers with source citations
- 🔗 **Relationship graph** — Visualise how papers build on, contradict, or share methods with each other
- 📊 **Deep analysis** — Research gaps, common methods, contradictions, timelines
- 🧠 **Corrective RAG** — LangGraph pipeline with retrieve → grade → generate nodes

---

## 🏗️ Architecture

```
User Question
     │
     ▼
LangGraph Pipeline
     ├── Node 1: retrieve_node  → ChromaDB MMR retrieval
     ├── Node 2: grade_node     → LLM relevance filtering
     └── Node 3: generate_node  → LLaMA 3.3 70B answer generation
```

**Stack:**

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Document Loaders | LangChain (PyMuPDF, Docx2txt, ArxivLoader) |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| LLM | LLaMA 3.3 70B via Groq API |
| RAG Workflow | LangGraph StateGraph |
| Graph Viz | NetworkX + PyVis |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Open `app.py` and replace line 9:
```python
GROQ_API_KEY = "your_groq_api_key_here"
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py              # Main Streamlit app + LangGraph pipeline
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from git
└── README.md           # This file
```

---

## ⚠️ Important

Never commit your Groq API key to GitHub. The `.gitignore` excludes `.env` files — consider moving your key there using `python-dotenv` before making the repo public.
