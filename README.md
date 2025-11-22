AI Document Chatbot (RAG)

A full-stack AI application that allows users to upload PDF documents and ask questions about them in natural language.

🚀 Features

Document Ingestion: Uploads and processes PDF files using pypdf.

RAG Pipeline: Uses LangChain and ChromaDB to create vector embeddings and retrieve relevant context.

AI Response: Generates accurate answers using an LLM (e.g., OpenAI/Llama) based only on the provided document.

API: Built with Flask (or FastAPI) to serve requests.

🛠️ Tech Stack

Language: Python 3.10

Frameworks: LangChain, Flask

Database: ChromaDB (Vector Store)

** LLM:** OpenAI GPT-3.5 / HuggingFace Hub

⚙️ How to Run

Clone the repo

Install dependencies: pip install -r requirements.txt

Add your API key to .env

Run the app: python app.py
