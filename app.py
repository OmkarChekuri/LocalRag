import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM # For initializing the main LLM
import os

# Import core RAG components from our new modules
from rag_core import get_rag_chain, set_llm_model
from vector_db import get_retrievers, CSV_DB_LOCATION, PDF_DB_LOCATION

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Pizza Restaurant Q&A", layout="centered")
st.title("RAG Chatbot")
st.markdown("""
Ask questions! You can choose to retrieve information from:
- **CSV Reviews**: For questions about pizza restaurant reviews.
- **PDF Documents**: For questions based on general PDF files in the `local_pdf` folder.
""")

# --- LLM and Embeddings Initialization (NO CACHING) ---
# Removed @st.cache_resource as per user request
def load_ollama_models_no_cache():
    """Loads Ollama LLM and Embeddings models without caching."""
    llm_instance = OllamaLLM(model="llama3.2")
    embeddings_instance = OllamaEmbeddings(model="mxbai-embed-large")
    return llm_instance, embeddings_instance

llm, embeddings = load_ollama_models_no_cache()

# Set the LLM model in the rag_core module
set_llm_model(llm)

# --- Retrievers and Initializing Vector Stores (NO CACHING) ---
# Removed @st.cache_resource as per user request
def load_and_get_retrievers_no_cache(embeddings_instance_param): # Renamed for clarity
    """
    Loads documents and initializes vector stores/retrievers without caching.
    """
    with st.spinner("Loading and processing documents... This may take a moment if new files are found."):
        # Pass the embeddings_instance directly
        csv_retriever, pdf_retriever = get_retrievers(embeddings_instance_param) 
    return csv_retriever, pdf_retriever

# Pass the embeddings instance to the retriever loader
csv_retriever, pdf_retriever = load_and_get_retrievers_no_cache(embeddings)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_source" not in st.session_state:
    st.session_state.selected_source = "CSV" # Default source

# --- Sidebar for Options ---
st.sidebar.header("Options")
source_choice = st.sidebar.radio(
    "Choose data source:",
    ("CSV Reviews", "PDF Documents"), # Changed label for clarity
    index=0 if st.session_state.selected_source == "CSV" else 1,
    key="source_radio"
)
if source_choice == "CSV Reviews":
    st.session_state.selected_source = "CSV"
    retriever = csv_retriever
else: # If "PDF Documents" is selected
    st.session_state.selected_source = "PDF"
    retriever = pdf_retriever

st.sidebar.markdown("---")
st.sidebar.subheader("Vector Store Status")
csv_db_exists = os.path.exists(CSV_DB_LOCATION) and len(os.listdir(CSV_DB_LOCATION)) > 0
pdf_db_exists = os.path.exists(PDF_DB_LOCATION) and len(os.listdir(PDF_DB_LOCATION)) > 0

st.sidebar.info(f"CSV Vector Store: {'Loaded' if csv_db_exists else 'Not yet populated (add files to local_csv)'}")
st.sidebar.info(f"PDF Vector Store: {'Loaded' if pdf_db_exists else 'Not yet populated (add files to local_pdf)'}")
st.sidebar.markdown("---")


# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
# Updated prompt text based on selected source
chat_input_placeholder = "Ask a question about pizza restaurant reviews..." if st.session_state.selected_source == "CSV" else "Ask a question about the PDF documents..."
if prompt := st.chat_input(chat_input_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner(f"Retrieving from {st.session_state.selected_source} and generating response..."):
        try:
            # Get the RAG chain with the selected retriever
            rag_chain = get_rag_chain(retriever)
            
            # Invoke the chain
            full_response = rag_chain.invoke(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"An error occurred: {e}. Please check your Ollama server and data files."
            st.error(error_message)
            with st.chat_message("assistant"):
                st.markdown("Sorry, an error occurred. Please try again.")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error occurred."})

# --- Clear Chat Button ---
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
