# rag
This application is to demostrate capability of a Rag Model
Let's break this down into key components:

Retrieval (FAISS + Sentence Transformers):

Uses all-mpnet-base-v2 model for dense vector embeddings

Creates a vector store using FAISS for efficient similarity search

Generation (FLAN-T5):

Uses Google's FLAN-T5 base model for text generation

Combines retrieved context with question for answer generation

Streamlit Interface:

Simple web interface for user interaction

Displays both generated answer and retrieved context

Shows processing time for transparency
