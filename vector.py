from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# Load the restaurant reviews dataset from a CSV file
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize the embedding model for document embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Set the location for the Chroma vector database
db_location = "./chroma_langchain_db"

# Determine if documents need to be added (only if DB doesn't exist)
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []  # List to store Document objects
    ids = []        # List to store unique document IDs

    # Iterate over each row in the DataFrame to create Document objects
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],  # Combine title and review text
            metadata={"rating": row["Rating"], "date": row["Date"]},  # Add metadata
            id=str(i)  # Assign a unique string ID
        )
        ids.append(i)
        documents.append(document)

# Initialize the Chroma vector store with the specified collection and embedding function
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# If adding documents, store them in the vector database
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# create a retriever from the vector store to fetch relevant documents
retriever = vector_store.as_retriever(
        search_kwargs={"k": 5} # Retrieve top 5 relevant documents
)