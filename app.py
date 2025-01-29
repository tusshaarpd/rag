import streamlit as st
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import json

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize vector database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("space_encyclopedia")

# Function to load documents into ChromaDB
def load_documents():
    with open("space_encyclopedia.json", "r") as f:
        documents = json.load(f)
    for doc in documents:
        embedding = embedding_model.encode(doc["text"]).tolist()
        collection.add(documents=[{"id": doc["id"], "text": doc["text"], "embedding": embedding}])

# Load documents if database is empty
if len(collection.get()) == 0:
    load_documents()

# Streamlit UI
st.title("RAG Model for Space Encyclopedia")
query = st.text_input("Ask a space-related question:")

if query:
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
