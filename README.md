# 🤖 AI Chat Bot — Document-Aware Conversational Assistant

A **RAG-based (Retrieval-Augmented Generation) AI chatbot** that answers questions from uploaded PDF documents using LLM-powered retrieval and streaming responses. Built with Flask, LangChain, and a modern web UI.

---

## ✨ Features

- 📄 **PDF Knowledge Base** — Upload PDFs and the bot builds a searchable vector index from their content.
- 🔍 **Semantic Search** — Uses sentence-transformer embeddings for accurate document retrieval.
- 💬 **Streaming Responses** — Answers are streamed token-by-token for a real-time chat experience.
- 💡 **Smart Suggestions** — Auto-generates relevant follow-up questions as you type.
- 🔄 **One-Click Refresh** — Rebuild the knowledge base without restarting the server.
- 🎨 **Modern Chat UI** — Glassmorphism design with animations and responsive layout.

---

## 📂 Project Structure

```
Ai Chat Bot/
│
├── v1/                              # Version 1 — Enhanced (Jarvis)
│   ├── app.py                       # Flask backend (ChromaDB + OpenRouter)
│   ├── samarth_prototype.py         # Streamlit-based agri-data Q&A prototype
│   ├── .env                         # API keys (OpenRouter)
│   ├── chatbot.db                   # SQLite database (chat history & logs)
│   ├── chroma_db/                   # ChromaDB vector store (auto-generated)
│   ├── docs/                        # PDF documents for the knowledge base
│   └── templates/
│       ├── index.html               # Main chat interface
│       └── chat_embedded.html       # Embeddable chat widget
│
├── chatbot(sonali version)/         # Version 2 — Lightweight (JotBot)
│   ├── app.py                       # Flask backend (FAISS + OpenAI)
│   ├── .env                         # API keys (OpenAI)
│   ├── requriments.txt              # Python dependencies
│   ├── docs/                        # PDF documents for the knowledge base
│   └── templates/
│       └── index.html               # Main chat interface
│
└── README.md
```

---

## 🔀 Version Comparison

| Feature | **v1 (Jarvis)** | **Sonali Version (JotBot)** |
|---|---|---|
| **LLM Provider** | OpenRouter (GPT-3.5 Turbo) | OpenAI Direct (GPT-4o-mini) |
| **Vector Store** | ChromaDB | FAISS |
| **Database** | SQLite (chat history + logs) | None |
| **Refresh** | Async with progress bar | Synchronous |
| **UI Theme** | Dark mode + glassmorphism | Light mode + clean design |
| **CORS** | Enabled | Not configured |
| **Extras** | Chat history page, logs page, embeddable widget | Lightweight, minimal setup |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, LangChain
- **LLMs:** OpenAI GPT-4o-mini / GPT-3.5 Turbo (via OpenRouter)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Stores:** ChromaDB (v1) / FAISS (Sonali version)
- **Frontend:** HTML, CSS, JavaScript, TailwindCSS
- **Database:** SQLite with Flask-SQLAlchemy (v1 only)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An API key from [OpenAI](https://platform.openai.com/) or [OpenRouter](https://openrouter.ai/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ai-chat-bot.git
   cd ai-chat-bot
   ```

2. **Choose a version and install dependencies**

   **For v1 (Jarvis):**
   ```bash
   cd v1
   pip install flask flask-sqlalchemy flask-cors python-dotenv langchain langchain-huggingface langchain-chroma langchain-openai langchain-community sentence-transformers pypdf chromadb
   ```

   **For Sonali Version (JotBot):**
   ```bash
   cd "chatbot(sonali version)"
   pip install -r requriments.txt
   ```

3. **Configure the environment**

   Create or edit the `.env` file in the chosen version's directory:

   **v1:**
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   DATABASE_URL=sqlite:///chatbot.db
   ```

   **Sonali Version:**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Add your PDF documents**

   Place your PDF files in the `docs/` folder inside the chosen version's directory.

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

---

## 💡 How It Works

```
User Question
     │
     ▼
┌────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Flask API │────▶│  Multi-Query     │────▶│  Vector     │
│  /chat     │     │  Retriever       │     │  Store      │
└────────────┘     └──────────────────┘     │  (ChromaDB/ │
     │                                       │   FAISS)    │
     │              ┌──────────────────┐     └─────────────┘
     │              │  Retrieved Docs  │◀────────┘
     │              └──────────────────┘
     │                      │
     ▼                      ▼
┌─────────────────────────────────┐
│  LLM generates answer using    │
│  retrieved context (streaming) │
└─────────────────────────────────┘
     │
     ▼
  Streamed Response → Chat UI
```

1. PDFs are loaded and split into chunks during startup.
2. Chunks are embedded using `all-MiniLM-L6-v2` and stored in a vector database.
3. User questions trigger a multi-query retrieval to find the most relevant chunks.
4. The LLM generates an answer grounded in the retrieved context.
5. The response is streamed back to the browser in real time.

---

## 📸 Screenshots

> _Add screenshots of the chat interface here._

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ using Flask, LangChain & LLMs
</p>
