from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 

# Import both retrievers from vector.py
from vector import csv_retriever, pdf_retriever

# Initialize the Ollama LLM with the specified model
model = OllamaLLM(model = "llama3.2")

# Define the prompt template for the chatbot
template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
# Create a ChatPromptTemplate object from the template string
prompt = ChatPromptTemplate.from_template(template)

# Create a chain that connects the prompt to the model
chain =  prompt | model

# Ask user which source to use
print("Choose data source for retrieval:")
print("1. CSV reviews")
print("2. PDF reviews")
source_choice = input("Enter 1 or 2: ").strip()
if source_choice == "2":
    retriever = pdf_retriever
else:
    retriever = csv_retriever

# Start an interactive loop to accept user questions
while True:
    print("\n\n-----------------------------New prompt-----------------------------")
    question = input("Please enter your question (q to quit): ")
    print("\n")
    if question == "q":
        break  # Exit the loop if the user enters 'q'
    reviews = retriever.invoke(question)  # Retrieve relevant reviews
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)  # Print the result from the chain
    print("\n\n")