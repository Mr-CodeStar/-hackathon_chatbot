# ⚡ hackathon_chatbot: Intelligent Multi-Mode AI Agent

An advanced, full-stack AI orchestrator prototype designed to provide seamless contextual interactions. The platform features an intelligent, multi-channel processing router that dynamically targets tasks into three operational paths:

* Standard Conversational Chat
* Real-Time Live Web Tools
* Persistent Vector-Based RAG Document Processing

---

# 🚀 Key Features

### Intelligent Dynamic Routing

Automatically analyzes incoming prompts and routes them into the appropriate workflow:

* Standard conversational reasoning
* Live data retrieval
* Document retrieval pipelines

---

### Persistent RAG Workspaces

Upload PDFs and automatically trigger:

* Text extraction
* Semantic chunking
* Embedding generation
* Vector persistence

Uses:

* BAAI/bge-small-en-v1.5 embeddings
* FAISS vector indexing

All processing remains scoped to the active chat session.

---

### Live Data Tooling Integration

Supports real-time data retrieval through:

* DuckDuckGo Search
* Live Stock Data
* Weather Information
* News Feeds

---

### Local Session Management

Maintains conversation history using:

* SQLite persistence
* Structured session storage
* Historical prompt recovery

---

### Responsive UI Sandbox

Frontend includes:

* Dark mode interface
* Markdown rendering
* Streaming simulation
* Sidebar contextual memory

---

# 🛠️ Tech Stack & Requirements

## Backend Framework

* Python 3.10+
* FastAPI
* Uvicorn
* LangChain Community
* FAISS
* SQLite3
* HuggingFace Transformers

### Core Backend Libraries

* PyPDFLoader
* RecursiveCharacterTextSplitter
* FAISS Vector Store
* HuggingFace Embeddings

---

## External Foundations & SDKs

### Google GenAI

Used for:

* Routing Logic
* Conversations
* Search Integration

Model:

* gemini-2.5-flash

### Groq

Used for:

* Fast Document Q&A
* Summarization

Model:

* llama-3.1-8b-instant

### Additional Integrations

* duckduckgo_search
* yfinance
* feedparser
* requests

---

## Frontend Stack

* HTML5
* CSS3
* TailwindCSS v4
* Marked.js
* FontAwesome v6

---

# 🗂️ Project Structure

```text
hackathon_chatbot/
│
├── database.py
│   └── SQLite schema setup and database initialization
│
├── live_data.py
│   └── Real-time search pipelines and tool routing
│
├── main.py
│   └── FastAPI application router and orchestration logic
│
├── rag.py
│   └── PDF extraction, embeddings, FAISS indexing
│
├── gen_ques.py
│   └── Experimental / standalone prototyping workspace
│
└── index.html
    └── Frontend dashboard interface
```

## Configure Environment Variables

Create a `.env` file inside the root folder.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key

GROQ_API_KEY=your_groq_api_key
```

---

# Future Improvements

* Authentication System
* Multi-user Workspace Support
* Cloud Vector Storage
* Voice Interface Integration
* Multi-agent Routing Pipelines
* Deployment Containers

---

Built for  hackathons
