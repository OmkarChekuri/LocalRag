# Local RAG Chatbot (CSV & PDF) with Upload, Memory & UI

This is a local chatbot application that lets you ask questions about your own CSV and PDF files. It runs entirely offline and uses LangChain, ChromaDB, Ollama, and Streamlit. You can upload files, chat with your data, and keep separate chat memory for CSVs and PDFs.

## What You Can Do

- Upload CSV and PDF files through the web interface
- Ask questions about the content of your files
- Keep chat memory separate for CSV and PDF modes (with an option to clear memory)
- See a list of uploaded files and debug information
- Use the app completely offline—no data leaves your computer

## Features

- **File Upload:** Add new PDFs and CSVs at any time using the UI.
- **Chat Memory:** The app keeps separate chat histories for PDFs and CSVs. You can clear them whenever you want.
- **Universal CSV Support:** Works with any CSV file by merging all columns into searchable text.
- **ChromaDB:** Stores document embeddings locally for fast retrieval.
- **Chat Interface:** Uses Streamlit for a simple, interactive chat experience.
- **Debugging Panel:** Shows internal logs, including your last input and prompt.

## Technology Stack

- Streamlit for the user interface
- LangChain for prompt chaining and retrieval
- Chroma for the local vector database
- Ollama for running local LLM and embedding models

## Requirements

### Install Ollama and Models

1. Download and install Ollama: https://ollama.com/download
2. Pull the required models:
   ```sh
   ollama pull llama3
   ollama pull mxbai-embed-large
   ```

### Python Dependencies

Install all dependencies with:
```sh
pip install -r requirements.txt
```

Your `requirements.txt` should include:
```
streamlit
pandas
langchain
langchain-core
langchain-community
langchain-chroma
ollama
pypdf
```

## How It Works

- Choose either PDF or CSV mode from the sidebar.
- Upload your documents through the interface, or add them directly to the `local_pdf/` or `local_csv/` folders.
- The app will index your files and store their embeddings in local ChromaDB folders.
- Ask questions in the chat. The app retrieves relevant content and generates a response.
- Chat memory is saved separately for CSV and PDF modes and can be cleared at any time.

## Folder Structure

```
.
├── app.py                  # Main app file
├── local_csv/              # Uploaded CSV files
├── local_pdf/              # Uploaded PDF files
├── chroma_csv_db/          # Chroma vector DB for CSVs
├── chroma_pdf_db/          # Chroma vector DB for PDFs
├── csv_files_record.json   # Tracks CSV files
├── pdf_files_record.json   # Tracks PDF files
└── README.md
```

## Running the App

Start the app with:
```sh
streamlit run app.py
```

## What You See in the UI

- Upload PDFs or CSVs
- See a list of all indexed files
- Ask questions and get context-aware answers
- Clear chat memory for either mode
- View debug logs for troubleshooting

## Example Use Cases

- Analyze customer reviews or feedback forms
- Ask questions about user manuals or product sheets
- Summarize internal documents or reports
- Explore structured datasets in a conversational way

## Clearing Memory

Use the "Clear Chat Memory" button to erase the chat history for either CSV or PDF mode.

## Privacy

This app runs entirely on your computer. No files or data are sent anywhere. You have full control over your models and your data.
