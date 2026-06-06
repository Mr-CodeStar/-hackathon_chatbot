# rag.py
import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Fallback API Key handling
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
client = Groq(api_key=GROQ_API_KEY)

VECTOR_STORE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def load_documents(pdf_files):
    all_documents = []
    for pdf in pdf_files:
        if os.path.exists(pdf):
            loader = PyPDFLoader(pdf)
            all_documents.extend(loader.load())
    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def create_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

def get_vector_store_path(session_id: str) -> str:
    return os.path.join(VECTOR_STORE_DIR, f"faiss_{session_id}")

def load_vector_store_for_session(session_id: str):
    path = get_vector_store_path(session_id)
    if not os.path.exists(path):
        return None
    embeddings = load_embedding_model()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def summarize_document(documents, max_words=300):
    full_text = ""
    for doc in documents:
        full_text += doc.page_content + "\n"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following document in approximately {max_words} words.\n\nDocument:\n{full_text[:6000]}"
            }
        ]
    )
    return response.choices[0].message.content

def answer_question(question, vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"Answer ONLY using the context provided.\n\nContext:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )
    return response.choices[0].message.content

def process_and_save_document(pdf_files, session_id: str):
    documents = load_documents(pdf_files)
    if not documents:
        raise ValueError("No text could be extracted from the document.")
    
    chunks = split_documents(documents)
    embeddings = load_embedding_model()
    vector_db = create_vector_store(chunks, embeddings)
    
    # Save index mapped directly to this session ID
    path = get_vector_store_path(session_id)
    vector_db.save_local(path)
    return vector_db, documents