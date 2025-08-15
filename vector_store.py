import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- Configuration ---
TEXT_SOURCE_PATH = "sec_10k_report.txt"
METADATA_SOURCE_PATH = "filing_metadata.json"
VECTOR_STORE_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def build_vector_store():
    """
    Loads text and metadata, splits the text into chunks, creates vector embeddings,
    and builds a persistent Chroma vector store.
    """
    # --- Verification ---
    if not os.path.exists(TEXT_SOURCE_PATH) or not os.path.exists(METADATA_SOURCE_PATH):
        print(f"Error: Required file(s) not found.")
        print("Please run `ingest_data.py` first to generate the report and metadata files.")
        return

    # --- Load Data with Metadata (Temporal Context) ---
    print("Loading documents with metadata...")
    with open(TEXT_SOURCE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    with open(METADATA_SOURCE_PATH, 'r') as f:
        metadata = json.load(f)

    # Create a single LangChain Document object that contains both the text and its metadata.
    documents = [Document(page_content=text, metadata=metadata)]

    # --- Chunk the Document ---
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    # The splitter automatically propagates the metadata to each new chunk.
    docs = text_splitter.split_documents(documents)

    # --- Create Embeddings ---
    print(f"Creating embeddings with '{EMBEDDING_MODEL}' model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # --- Build and Persist Vector Store ---
    print(f"Building and persisting vector store at '{VECTOR_STORE_PATH}'...")
    # This creates a persistent vector store on disk.
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print("Vector store built successfully.")
    print(f"Total chunks processed: {len(docs)}")
    if docs:
        print(f"Metadata check for first chunk: {docs[0].metadata}")

if __name__ == "__main__":
    build_vector_store()