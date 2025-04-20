from langchain_groq import ChatGroq
import os

def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    return llm
