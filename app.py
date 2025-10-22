import os
import json
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response
from flask_sqlalchemy import SQLAlchemy
# ADDED: This line is new
from flask_cors import CORS 
import datetime
import re
import shutil
import threading
import time
import gc

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- Configuration & Globals ---
load_dotenv()
BASE_DIR = Path(__file__).parent.resolve()
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# ADDED: This line is new. It allows your PDF editor to connect.
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000"}}) 

app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-super-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{BASE_DIR / "chatbot.db"}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Global variable to track the refresh progress ---
REFRESH_STATUS = {
    "is_running": False,
    "message": "Idle",
    "progress": 0
}

# --- Database Models ---
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class AppLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# --- Logging Helper ---
def log_message(level, message):
    with app.app_context():
        log_entry = AppLog(level=level, message=message)
        db.session.add(log_entry)
        db.session.commit()

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in .env file.")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-3.5-turbo"
DEFAULT_HEADERS = { "HTTP-Referer": "http://localhost:5000", "X-Title": "Jarvis" }

# --- Global Models & Vector Store ---
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
streaming_llm = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=OPENROUTER_API_KEY, openai_api_base=OPENROUTER_BASE_URL, streaming=True, temperature=0.1, default_headers=DEFAULT_HEADERS)
gen_llm = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=OPENROUTER_API_KEY, openai_api_base=OPENROUTER_BASE_URL, streaming=False, temperature=0.0, default_headers=DEFAULT_HEADERS)
vector_store = None
vector_store_ready = threading.Event()

# --- Vector Store Functions ---
def load_or_build_vector_store(force_rebuild=False):
    global vector_store, REFRESH_STATUS
    vector_store_ready.clear()
    
    try:
        if force_rebuild:
            REFRESH_STATUS.update({"is_running": True, "message": "Preparing to refresh...", "progress": 0})
            if vector_store is not None:
                log_message("INFO", "Releasing existing vector store from memory.")
                vector_store = None
                gc.collect()
                time.sleep(1) 

            if CHROMA_DIR.exists():
                REFRESH_STATUS.update({"message": "Deleting old index...", "progress": 10})
                shutil.rmtree(CHROMA_DIR)
                time.sleep(1)

        if CHROMA_DIR.exists():
            log_message("INFO", "Loading existing vector store from disk.")
            vector_store = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings_model)
            log_message("INFO", "Vector store loaded successfully.")
            vector_store_ready.set() # Signal ready only on success
        else:
            log_message("INFO", "Building new vector store...")
            REFRESH_STATUS.update({"message": "Scanning for documents...", "progress": 20})
            loaders = [PyPDFLoader(str(p)) for p in DOCS_DIR.glob("*.pdf")]
            if not loaders:
                log_message("WARNING", "No PDF files found in ./docs directory.")
                REFRESH_STATUS.update({"message": "No documents found to index.", "progress": 100})
                return # Exit without setting the ready event

            REFRESH_STATUS.update({"message": "Loading document content...", "progress": 40})
            docs = [doc for loader in loaders for doc in loader.load()]
            if not docs:
                log_message("WARNING", "Could not load any content from PDF files.")
                REFRESH_STATUS.update({"message": "Could not load content.", "progress": 100})
                return # Exit without setting the ready event

            REFRESH_STATUS.update({"message": "Splitting documents into chunks...", "progress": 60})
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            split_docs = text_splitter.split_documents(docs)

            REFRESH_STATUS.update({"message": f"Creating index from {len(split_docs)} chunks...", "progress": 80})
            vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings_model, persist_directory=str(CHROMA_DIR))
            log_message("INFO", f"Vector store built successfully with {len(split_docs)} document chunks.")
            vector_store_ready.set() # Signal ready only on success

        REFRESH_STATUS.update({"message": "Refresh complete!", "progress": 100})
    except Exception as e:
        vector_store = None # Ensure vector_store is None on error
        error_msg = f"Failed during vector store build/load. Error: {e}"
        log_message("ERROR", error_msg)
        traceback.print_exc() # Print full traceback to console for debugging
        REFRESH_STATUS.update({"message": f"Error: {e}", "progress": 100})
    finally:
        REFRESH_STATUS["is_running"] = False


# --- Flask Endpoints ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/get_initial_data')
def get_initial_data():
    username = "Jarvis"
    greeting = "Good " + ("Morning" if 5 <= datetime.datetime.now().hour < 12 else "Afternoon" if 12 <= datetime.datetime.now().hour < 17 else "Evening")
    return jsonify({"username": username, "greeting": greeting})

@app.route("/refresh_index", methods=["POST"])
def refresh_index_endpoint():
    global REFRESH_STATUS
    if REFRESH_STATUS["is_running"]:
        return jsonify({"status": "error", "message": "A refresh is already in progress."}), 409
    threading.Thread(target=load_or_build_vector_store, kwargs={'force_rebuild': True}).start()
    return jsonify({"status": "accepted", "message": "Knowledge base refresh initiated."}), 202

@app.route("/refresh_status", methods=["GET"])
def refresh_status_endpoint():
    return jsonify(REFRESH_STATUS)

@app.route("/suggest_questions", methods=["POST"])
def suggest_questions_endpoint():
    if not vector_store_ready.is_set() or vector_store is None:
        return jsonify({"questions": []})
    
    data = request.get_json()
    keyword = data.get("keyword", "").strip()
    if not keyword or len(keyword) < 3:
        return jsonify({"questions": []})
    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 10})
        docs = retriever.get_relevant_documents(keyword)
        if not docs: return jsonify({"questions": []})
        
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_template = """Your sole task is to generate up to 3 user questions that can be directly and completely answered using ONLY the provided text CONTEXT.
- Do NOT invent any details, entities, or topics not explicitly mentioned in the text.
- Return ONLY a valid JSON list of strings. If no questions can be formed, return an empty list [].
CONTEXT:
{context}
JSON Question List:"""
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=gen_llm, prompt=prompt)
        output = chain.run(context=context)
        try:
            match = re.search(r'\[.*\]', output, re.DOTALL)
            questions = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            questions = [line.strip("- ").strip() for line in output.split('\n') if '?' in line]
        return jsonify({"questions": questions[:3]})
    except Exception as e:
        log_message("ERROR", f"Error in /suggest_questions: {e}")
        return jsonify({"questions": []})

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_question = data.get("query", "").strip()
    username = "Jarvis" # Hardcoded username for history
    if not user_question: return Response("Please ask a question.", mimetype='text/plain')
    if user_question.lower() in {"hello", "hi", "hey"}: return Response("Hello! How can I help you today?", mimetype='text/plain')
    
    if not vector_store_ready.is_set() or vector_store is None:
        return Response("Knowledge base is still initializing or has failed to load. Please try again shortly or check the logs.", mimetype='text/plain')

    try:
        retriever = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=gen_llm)
        retrieved_docs = retriever.get_relevant_documents(user_question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer_prompt = PromptTemplate.from_template("""Answer the user's QUESTION using ONLY the provided CONTEXT. If the answer is not in the context, say "The answer is not available in the provided documents."
CONTEXT: {context}
QUESTION: {question}
ANSWER:""")
        answer_chain = LLMChain(llm=streaming_llm, prompt=answer_prompt)
        full_response = []
        def generate_stream():
            for chunk in answer_chain.stream({"context": context, "question": user_question}):
                text_chunk = chunk.get("text", "")
                full_response.append(text_chunk)
                yield text_chunk
            final_answer = "".join(full_response)
            with app.app_context():
                history_entry = ChatHistory(username=username, question=user_question, answer=final_answer)
                db.session.add(history_entry)
                db.session.commit()
        return Response(generate_stream(), mimetype='text/plain')
    except Exception as e:
        error_msg = f"Error in /chat: {e}\n{traceback.format_exc()}"; print(error_msg); log_message("ERROR", error_msg)
        return Response("An error occurred on the server.", mimetype='text/plain')

@app.route("/history")
def history():
    history_entries = ChatHistory.query.order_by(ChatHistory.timestamp.desc()).all()
    return render_template('history.html', history=history_entries)

@app.route("/logs")
def logs():
    log_entries = AppLog.query.order_by(AppLog.timestamp.desc()).all()
    return render_template('logs.html', logs=log_entries)

# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    threading.Thread(target=load_or_build_vector_store).start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

