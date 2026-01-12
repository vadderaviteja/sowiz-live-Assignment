from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- LOAD PDF ----------------

def function():
    loader = PyPDFLoader(r"C:\Users\Windows\Downloads\GEN_AI Interview questions.pdf")
    docs = loader.load()

    # ---------------- SPLIT ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # ---------------- EMBEDDINGS ----------------
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------- VECTOR STORE ----------------
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db"
    )

    print("✅ PDF indexed and persisted successfully")
    return db
    
    