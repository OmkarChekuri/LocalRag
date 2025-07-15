from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None

def find_files_recursive(root_folder, extension):
    """Recursively find files with the given extension in root_folder."""
    file_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(extension):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def load_processed_files(record_path):
    if os.path.exists(record_path):
        with open(record_path, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_files(record_path, processed_files):
    with open(record_path, "w") as f:
        json.dump(list(processed_files), f)

def load_csv_documents(csv_folder, processed_files):
    import pandas as pd
    documents = []
    ids = []
    csv_files = find_files_recursive(csv_folder, ".csv")
    for csv_path in csv_files:
        filename = os.path.relpath(csv_path, csv_folder)
        if filename not in processed_files:
            df = pd.read_csv(csv_path)
            for i, row in df.iterrows():
                document = Document(
                    page_content=str(row.get("Title", "")) + " " + str(row.get("Review", "")),
                    metadata={"rating": row.get("Rating", ""), "date": row.get("Date", ""), "source": filename},
                    id=f"{filename}_{i}"
                )
                ids.append(f"{filename}_{i}")
                documents.append(document)
            processed_files.add(filename)
    return documents, ids, processed_files

def load_pdf_documents(pdf_folder, processed_files):
    documents = []
    ids = []
    if PyPDFLoader is None:
        print("PDF support requires langchain_community. Run: pip install langchain-community")
        return documents, ids, processed_files
    pdf_files = find_files_recursive(pdf_folder, ".pdf")
    for pdf_path in pdf_files:
        filename = os.path.relpath(pdf_path, pdf_folder)
        if filename not in processed_files:
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
    return documents, ids, processed_files

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Collection/database locations
csv_db_location = "./chroma_csv_db"
pdf_db_location = "./chroma_pdf_db"
csv_record_path = "./csv_files_record.json"
pdf_record_path = "./pdf_files_record.json"

# Load processed files record
csv_processed_files = load_processed_files(csv_record_path)
pdf_processed_files = load_processed_files(pdf_record_path)

# CSV collection
csv_vector_store = Chroma(
    collection_name="csv_reviews",
    persist_directory=csv_db_location,
    embedding_function=embeddings,
)
csv_documents, csv_ids, csv_processed_files = load_csv_documents("local_csv", csv_processed_files)
if csv_documents:
    csv_vector_store.add_documents(documents=csv_documents, ids=csv_ids)
    save_processed_files(csv_record_path, csv_processed_files)

# PDF collection
pdf_vector_store = Chroma(
    collection_name="pdf_reviews",
    persist_directory=pdf_db_location,
    embedding_function=embeddings,
)
pdf_documents, pdf_ids, pdf_processed_files = load_pdf_documents("local_pdf", pdf_processed_files)
if pdf_documents:
    pdf_vector_store.add_documents(documents=pdf_documents, ids=pdf_ids)
    save_processed_files(pdf_record_path, pdf_processed_files)

# Retrievers for each collection
csv_retriever = csv_vector_store.as_retriever(search_kwargs={"k": 5})
pdf_retriever = pdf_vector_store.as_retriever(search_kwargs={"k": 5})