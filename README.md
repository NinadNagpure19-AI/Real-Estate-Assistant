# 🏙️ Real Estate Research Tool

A RAG (Retrieval-Augmented Generation) based research tool that lets you paste URLs from any real estate or mortgage news articles, and ask questions to get precise answers extracted from those articles — powered by LangChain, ChromaDB, Groq (Llama3), and Google Gemini Embeddings.

---

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-estate-assistant-r5wpmfaahnglsrhzuh4xfp.streamlit.app/)

👉 [Click here to try the app](https://real-estate-assistant-r5wpmfaahnglsrhzuh4xfp.streamlit.app/)

## Features

- Paste up to 3 article URLs from any news source
- Fetches and parses webpage content using `requests` + `BeautifulSoup`
- Splits content into chunks and stores as vector embeddings in ChromaDB
- Ask questions in natural language and get answers grounded in the articles
- Displays source URLs alongside every answer
- Built with LangChain's LCEL (LangChain Expression Language) pipeline

---

## How It Works

### Ingestion Pipeline
1. **Load** — Webpage content is fetched using `requests` with a browser User-Agent header and parsed with `BeautifulSoup` to strip noise (scripts, navbars, footers). Each page becomes a LangChain `Document` object.
2. **Split** — `RecursiveCharacterTextSplitter` breaks documents into overlapping chunks of 1000 characters using separators `["\n\n", "\n", ".", " "]` to preserve semantic boundaries.
3. **Embed** — Each chunk is converted into a vector using `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`).
4. **Store** — Vectors are stored in a local `ChromaDB` collection (`real_estate`) persisted to `resources/vectorstore/`.

### Retrieval Pipeline
1. **Retrieve** — On a user query, ChromaDB performs a **similarity search** to find the most relevant chunks from the stored vectors.
2. **Format** — Retrieved chunks are concatenated into a single context string (stuff method).
3. **Generate** — The context + question are passed to `ChatGroq` (Llama 3.3 70B) via a `ChatPromptTemplate`.
4. **Parse** — `StrOutputParser` extracts the final answer string.

### LangChain LCEL Chain
```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | PROMPT | llm | StrOutputParser()
)
```
The chain is composed declaratively using the `|` pipe operator — retriever feeds into the prompt, which feeds into the LLM, which feeds into the output parser.

---

## Tech Stack

| Component | Tool |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 3.3 70B via Groq |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | ChromaDB |
| Orchestration | LangChain (LCEL) |
| Web Scraping | requests + BeautifulSoup |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/real-estate-research-tool.git
cd real-estate-research-tool
```

### 2. Create and activate a virtual environment
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

### 4. Create a `.env` file
```text
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```
Get your Groq API key at https://console.groq.com  
Get your Google API key at https://aistudio.google.com

### 5. Run the app
```bash
streamlit run main.py
```

---

## Usage

1. Paste 1–3 article URLs in the sidebar
2. Click **Process URLs** and wait for all steps to complete
3. Type your question in the text box
4. Get your answer with source citations

### Sample URLs to try
```
https://www.cnbc.com/2026/05/19/mortgage-rates-closing-in-on-7percent.html
https://www.bankrate.com/mortgages/rate-trends/
https://fortune.com/article/current-mortgage-rates-03-25-2026
```

### Sample Questions
- What is the current 30-year fixed mortgage rate?
- Why are mortgage rates rising in 2026?
- How does inflation affect mortgage rates?
- What are experts predicting for mortgage rates by end of 2026?
- How has the war in Iran affected mortgage rates?

---

## Note on URL Loader

This project uses `requests` + `BeautifulSoup` instead of LangChain's `UnstructuredURLLoader`. `UnstructuredURLLoader` works well on Linux/Mac but requires native C libraries (`libmagic`, `onnxruntime`) that cause segfaults on Windows. The replacement produces identical LangChain `Document` objects — the rest of the RAG pipeline is completely unchanged.

---

## Project Structure

```
├── main.py              # Streamlit frontend
├── rag.py               # Ingestion + retrieval pipeline
├── prompt.py            # LangChain prompt template
├── requirements.txt     # Dependencies
├── .env                 # API keys (not committed)
├── .gitignore
└── resources/
    └── vectorstore/     # ChromaDB persisted storage
```

