# rag
This application is to demostrate capability of a Rag Model
Let's break this down into key components:

Document Retrieval: Use an open-source space encyclopedia (e.g., Wikipedia dumps or NASA's open datasets) and store it in a vector database.
Embedding Model: Use an open-source model like sentence-transformers/all-MiniLM-L6-v2 for vectorization.
Vector Database: Use FAISS or ChromaDB for storing and retrieving relevant documents.
LLM for Response Generation: Use Mistral-7B, Llama2, or Phi-2 via transformers and torch (not API-based).
Frontend with Streamlit: Create an interface where users can ask space-related questions, retrieve relevant documents, and generate responses.
