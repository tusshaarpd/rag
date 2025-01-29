import streamlit as st
from rag_pipeline import SpaceRAG
import time

# App configuration
st.set_page_config(page_title="Space RAG Explorer", page_icon="🚀")

@st.cache_resource
def load_rag_system():
    return SpaceRAG()

def main():
    st.title("🚀 Space Encyclopedia RAG System")
    st.markdown("Ask questions about space exploration and astronomy!")

    rag = load_rag_system()

    query = st.text_input("Enter your space-related question:", 
                        placeholder="E.g., What is a black hole?")

    if query:
        start_time = time.time()
        
        with st.spinner("Searching through space knowledge..."):
            context = rag.retrieve(query)
        
        with st.spinner("Generating answer..."):
            answer = rag.generate(query, context)
        
        st.subheader("Answer:")
        st.write(answer)

        st.divider()
        
        st.subheader("Retrieved Context:")
        for i, text in enumerate(context, 1):
            st.markdown(f"**Document {i}:**")
            st.info(text)

        st.caption(f"Process completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
