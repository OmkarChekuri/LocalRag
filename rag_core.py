from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Placeholder for LLM model (will be set by app.py)
model = None

def set_llm_model(llm_instance):
    """Sets the LLM instance for the RAG chain."""
    global model
    model = llm_instance

def get_rag_chain(retriever):
    """
    Creates and returns the RAG chain.
    Assumes 'model' has been set via set_llm_model.
    """
    if model is None:
        raise ValueError("LLM model has not been set in rag_core.py. Call set_llm_model first.")

    template = """
    You are an expert in answering questions about a pizza restaurant.
    Use the following reviews to answer the question. If the reviews do not contain
    enough information, say "I don't have enough information to answer that question."

    Here are some relevant reviews:
    {reviews}

    Here is the question to answer: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define the RAG chain
    rag_chain = (
        {"reviews": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain

