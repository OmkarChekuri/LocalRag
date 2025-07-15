# LocalRAG: Pizza Restaurant Review Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about a pizza restaurant using real customer reviews. It leverages [LangChain](https://python.langchain.com/) with Ollama LLM and Chroma vector store for semantic search and response generation.

## Features

- **Semantic Search:** Retrieves the most relevant reviews for a given question.
- **LLM-Powered Answers:** Uses a local Ollama LLM to generate expert answers based on retrieved reviews.
- **Interactive CLI:** Ask questions in a terminal loop and get instant, context-aware responses.

## Project Structure

```
.
├── main.py                # Main chatbot loop
├── vector.py              # Vector store and retriever setup
├── realistic_restaurant_reviews.csv  # Restaurant reviews dataset
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
   ```

4. **Download or prepare your `realistic_restaurant_reviews.csv` file**  
   Ensure it has columns: `Title`, `Review`, `rating`, `date`.

5. **Run the chatbot**
   ```sh
   python main.py
   ```

## How It Works

- **vector.py:**  
  Loads the reviews, embeds them, and stores them in a Chroma vector database. Provides a retriever to fetch the most relevant reviews for a query.

- **main.py:**  
  Loads the retriever and LLM, prompts the user for questions, retrieves relevant reviews, and generates an answer using the LLM.

## Example Usage

```
Please enter your question (q to quit): What do people think about the crust?
[LLM responds with a summary based on relevant reviews]
```

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally with the `llama3.2` and `mxbai-embed-large` models
- [LangChain](https://python.langchain.com/)
- [Chroma](https://www.trychroma.com/)

## Notes

- The first run will create a local vector database from your CSV reviews.
- You can customize the prompt in `main.py` for different domains or styles.

---

**License:** MIT