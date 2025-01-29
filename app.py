import streamlit as st
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import json
import os
import sqlite3

# Check SQLite version
sqlite_version = sqlite3.sqlite_version
if tuple(map(int, sqlite_version.split("."))) < (3, 35, 0):
    st.error(f"Your SQLite version ({sqlite_version}) is outdated. ChromaDB requires SQLite >= 3.35.0. Please update SQLite and restart the app.")
    st.stop()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Ensure database directory exists
DB_PATH = "./chroma_db"
os.makedirs(DB_PATH, exist_ok=True)

# Initialize vector database
try:
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection("space_encyclopedia")
except Exception as e:
    st.error(f"ChromaDB initialization failed: {e}")
    chroma_client = chromadb.Client()  # Fallback to in-memory mode
    collection = chroma_client.create_collection("space_encyclopedia")

# Function to load documents into ChromaDB
def load_documents():
    try:
        with open("space_encyclopedia.json", "r") as f:
            documents = json.load(f)
        for doc in documents:
            embedding = embedding_model.encode(doc["text"]).tolist()
            collection.add(documents=[{"id": doc["id"], "text": doc["text"], "embedding": embedding}])
    except FileNotFoundError:
        st.warning("Document file not found. Please add space_encyclopedia.json.")
    except Exception as e:
        st.error(f"Error loading documents: {e}")

# Load documents if database is empty
if len(collection.get()) == 0:
    load_documents()

# Streamlit UI
st.title("RAG Model for Space Encyclopedia")
query = st.text_input("Ask a space-related question:")

if query:
    try:
        # Generate embedding for query
        query_embedding = embedding_model.encode(query).tolist()
        
        # Retrieve similar documents
        results = collection.query(query_embedding, n_results=3)
        retrieved_docs = "\n".join([doc["text"] for doc in results["documents"]])
        
        # Generate response using LLM
        input_text = f"Context: {retrieved_docs}\n\nQuestion: {query}\n\nAnswer:"
        response = generator(input_text, max_length=200, num_return_sequences=1)[0]['generated_text']
        
        st.subheader("Answer:")
        st.write(response)
    except Exception as e:
        st.error(f"Error processing query: {e}")
