# ⚛️ Quantum Network RAG Engine

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant text chunks from quantum computing literature (scientific ArXiv papers) and generates streaming responses using **Groq Cloud API** or a local **Ollama** server.

Built with **Streamlit**, **Supabase (pgvector)**, and **LangChain**.

---

## 🚀 Features

* **Local Embedding Generation**: Uses Hugging Face (`all-MiniLM-L6-v2`) locally to embed text chunks and queries for 100% free vector indexing.
* **Vector Similarity Search**: Leverages Supabase with `pgvector` to perform cosine similarity searches.
* **High-Speed Cloud LLM**: Uses Groq Cloud API (`llama-3.1-8b-instant`) to stream response generations in real-time.
* **Local Fallback LLM**: Optional integration with local Ollama (`llama3` and `nomic-embed-text`) for full offline operations.
* **Clean UI**: Responsive chat interface styled natively with Streamlit.

---

## 🛠️ Project Structure

```text
├── app.py                # Main Streamlit chat UI and RAG pipeline logic
├── local_ingest.py       # Offline script to parse, chunk, embed, and store PDFs
├── ingest.py             # Optional web scraper and ingestion script
├── quantum_papers/       # Folder containing source PDF papers
├── .env.example          # Reference file for required environment variables
└── README.md             # Project documentation
```

---

## ⚙️ Local Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/khurdiaakshat123/quantum_arxiv_pipeline.git
cd quantum_arxiv_pipeline
```

### 2. Configure Environment Variables
Create a file named `.env` in the root folder (use `.env.example` as a template) and add your keys:
```env
SUPABASE_URL=https://your-project-id.supabase.co/rest/v1/
SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
SUPABASE_SECRET_KEY=your-supabase-secret-key
GROQ_API_KEY=your-groq-api-key
```

### 3. Install Dependencies
```bash
pip install streamlit python-dotenv supabase langchain-openai langchain-ollama langchain-groq langchain-community langchain-core pypdf sentence-transformers
```

### 4. Run the Ingestion Pipeline
To ingest and embed the papers in your `quantum_papers/` directory into Supabase, run:
```bash
python local_ingest.py --provider huggingface
```

### 5. Launch the Streamlit Chatbot
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 💡 RAG Configuration Options
* Select **Groq (Free Cloud API)** as the LLM Provider in the sidebar.
* Under *Embeddings Source for Groq*, select **Hugging Face Local (Free)**.
* Adjust the *Similarity Threshold* and *Top K Chunks* sliders to customize how many documents are retrieved for generation.
