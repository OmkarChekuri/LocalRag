# LocalRAG: Flexible Retrieval-Augmented Generation Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that can answer questions using information from both CSV and PDF files. It leverages [LangChain](https://python.langchain.com/) with Ollama LLM and Chroma vector store for semantic search and response generation.

## Features

- **Semantic Search:** Retrieves the most relevant information for a given question from CSV or PDF files.
- **LLM-Powered Answers:** Uses a local Ollama LLM to generate expert answers based on retrieved data.
- **Interactive CLI:** Ask questions in a terminal loop and get instant, context-aware responses.
- **Flexible Data Sources:** Supports CSV files (in `local_csv/`) and PDF files (in `local_pdf/`), including recursive folder search.
- **No Duplicate Embeddings:** Tracks processed files to avoid re-embedding the same data.
- **General Purpose:** Not limited to restaurant reviews—add any CSV or PDF files and ask questions about their content.

## Project Structure

```
.
├── main.py                # Main chatbot loop
├── vector.py              # Vector store and retriever setup
├── local_csv/             # Folder for CSV files (can contain subfolders)
├── local_pdf/             # Folder for PDF files (can contain subfolders)
├── chroma_csv_db/         # Chroma vector database for CSV files
├── chroma_pdf_db/         # Chroma vector database for PDF files
├── csv_files_record.json  # Tracks processed CSV files
├── pdf_files_record.json  # Tracks processed PDF files
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/LocalRAG.git
   cd LocalRAG
   ```

2. **Create and activate a virtual environment**
   ```sh
   python -m venv rag_evn
   .\rag_evn\Scripts\activate
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   # For PDF support:
   pip install langchain-community
   ```

4. **Prepare your data files**
   - Place CSV files (with columns like `Title`, `Review`, `Rating`, `Date` or any other structure) in the `local_csv/` folder (subfolders allowed).
   - Place PDF files in the `local_pdf/` folder (subfolders allowed).

5. **Run the chatbot**
   ```sh
   python main.py
   ```

## How It Works

- **vector.py:**  
  Recursively loads documents from CSV and PDF files, embeds them, and stores them in separate Chroma vector databases. Tracks processed files to avoid duplicates. Provides retrievers for both collections.

- **main.py:**  
  Prompts the user to select the data source (CSV or PDF), retrieves relevant documents using the chosen retriever, and generates an answer using the LLM.

## Example Usage

```
Choose data source for retrieval:
1. CSV reviews
2. PDF reviews
Enter 1 or 2: 2

-----------------------------New prompt-----------------------------
Please enter your question (q to quit): What does the contract say about payment terms?
[LLM responds with a summary based on relevant PDF content]
```

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally with the `llama3.2` and `mxbai-embed-large` models
- [LangChain](https://python.langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [langchain-community](https://pypi.org/project/langchain-community/) (for PDF support)

## Notes

- The first run will create local vector databases from your CSV and/or PDF files.
- New files added to `local_csv/` or `local_pdf/` will be automatically embedded on the next run.
- You can customize the prompt in `main.py` for different domains or styles.