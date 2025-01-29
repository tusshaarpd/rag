import os
from transformers import pipeline
from langchain_community.vectorstores import FAISS  # Changed import
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

class SpaceRAG:
    def __init__(self, data_dir="sample_data"):
        self.data_dir = data_dir
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1  # Use CPU; change to 0 for GPU if available
        )
        self.vector_store = self._create_vector_store()

    def _load_documents(self):
        documents = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".txt"):
                with open(os.path.join(self.data_dir, file), "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(text)
        return documents

    def _create_vector_store(self):
        documents = self._load_documents()
        texts = self.text_splitter.split_text("\n\n".join(documents))
        return FAISS.from_texts(texts, self.embeddings)

    def retrieve(self, query, k=3):
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def generate(self, query, context):
        prompt = f"Answer this question: {query}\n\nContext: {' '.join(context)}"
        result = self.qa_pipeline(
            prompt,
            max_length=512,
            num_return_sequences=1
        )
        return result[0]['generated_text']
