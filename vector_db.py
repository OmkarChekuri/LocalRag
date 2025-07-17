from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json
import pandas as pd # Import pandas here as it's used in load_csv_documents

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None
    print("Warning: PyPDFLoader not found. PDF support will be disabled. Run: pip install pypdf langchain-community")

def find_files_recursive(root_folder, extension):
    """Recursively find files with the given extension in root_folder."""
    file_paths = []
    if not os.path.exists(root_folder):
        print(f"Warning: Folder '{root_folder}' not found. Skipping file search.")
        return []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(extension):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def load_processed_files(record_path):
    """Loads a set of already processed file paths from a JSON record."""
    if os.path.exists(record_path):
        with open(record_path, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_files(record_path, processed_files):
    """Saves a set of processed file paths to a JSON record."""
    with open(record_path, "w") as f:
        json.dump(list(processed_files), f)

def load_csv_documents(csv_folder, processed_files):
    """Loads documents from CSV files in a given folder."""
    documents = []
    ids = []
    csv_files = find_files_recursive(csv_folder, ".csv")
    for csv_path in csv_files:
        filename = os.path.relpath(csv_path, csv_folder)
        if filename not in processed_files:
            try:
                df = pd.read_csv(csv_path)
                for i, row in df.iterrows():
                    # Combine 'Title' and 'Review' for page_content
                    content = ""
                    if "Title" in row and pd.notna(row["Title"]):
                        content += str(row["Title"])
                    if "Review" in row and pd.notna(row["Review"]):
                        if content: content += " "
                        content += str(row["Review"])
                    
                    if content: # Only add if content is not empty
                        document = Document(
                            page_content=content,
                            metadata={"rating": row.get("Rating", ""), "date": row.get("Date", ""), "source": filename},
                            id=f"{filename}_{i}"
                        )
                        ids.append(f"{filename}_{i}")
                        documents.append(document)
                processed_files.add(filename)
                print(f"Loaded CSV: {filename}")
            except Exception as e:
                print(f"Error loading CSV file {csv_path}: {e}")
    return documents, ids, processed_files

def load_pdf_documents(pdf_folder, processed_files):
    """Loads documents from PDF files in a given folder."""
    documents = []
    ids = []
    if PyPDFLoader is None:
        print("PDF support requires langchain_community and pypdf. Skipping PDF loading.")
        return documents, ids, processed_files
    
    pdf_files = find_files_recursive(pdf_folder, ".pdf")
    for pdf_path in pdf_files:
        filename = os.path.relpath(pdf_path, pdf_folder)
        if filename not in processed_files:
            try:
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                for i, doc in enumerate(pdf_docs):
                    document = Document(
                        page_content=doc.page_content,
                        metadata={"source": filename, "page": i+1},
                        id=f"{filename}_{i+1}"
                    )
                    ids.append(f"{filename}_{i+1}")
                    documents.append(document)
                processed_files.add(filename)
                print(f"Loaded PDF: {filename}")
            except Exception as e:
                print(f"Error loading PDF file {pdf_path}: {e}")
    return documents, ids, processed_files

# Collection/database locations
CSV_DB_LOCATION = "./chroma_csv_db"
PDF_DB_LOCATION = "./chroma_pdf_db"
CSV_RECORD_PATH = "./csv_files_record.json"
PDF_RECORD_PATH = "./pdf_files_record.json"

def get_retrievers(embeddings_model): # This function is called from app.py
    """Initializes and returns CSV and PDF retrievers."""
    # Load processed files record
    csv_processed_files = load_processed_files(CSV_RECORD_PATH)
    pdf_processed_files = load_processed_files(PDF_RECORD_PATH)

    # CSV collection
    csv_vector_store = Chroma(
        collection_name="csv_reviews",
        persist_directory=CSV_DB_LOCATION,
        embedding_function=embeddings_model, # embeddings_model is passed in
    )
    csv_documents, csv_ids, csv_processed_files = load_csv_documents("local_csv", csv_processed_files)
    if csv_documents:
        csv_vector_store.add_documents(documents=csv_documents, ids=csv_ids)
        save_processed_files(CSV_RECORD_PATH, csv_processed_files)
        print(f"Added {len(csv_documents)} new CSV documents to vector store.")
    else:
        print("No new CSV documents to add.")

    # PDF collection
    pdf_vector_store = Chroma(
        collection_name="pdf_reviews",
        persist_directory=PDF_DB_LOCATION,
        embedding_function=embeddings_model, # embeddings_model is passed in
    )
    pdf_documents, pdf_ids, pdf_processed_files = load_pdf_documents("local_pdf", pdf_processed_files)
    if pdf_documents:
        pdf_vector_store.add_documents(documents=pdf_documents, ids=pdf_ids)
        save_processed_files(PDF_RECORD_PATH, pdf_processed_files)
        print(f"Added {len(pdf_documents)} new PDF documents to vector store.")
    else:
        print("No new PDF documents to add.")

    # Retrievers for each collection
    csv_retriever = csv_vector_store.as_retriever(search_kwargs={"k": 5})
    pdf_retriever = pdf_vector_store.as_retriever(search_kwargs={"k": 5})

    return csv_retriever, pdf_retriever

