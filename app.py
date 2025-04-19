import sys
import torch
import hashlib
import re
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
import concurrent.futures

# Load environment variables
load_dotenv()

# Environment variables for API keys
groq_api_key = os.getenv("GROQ_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

# Initialize LLM and embeddings
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2",  model_kwargs={"device": "cpu"})

@st.cache_resource
def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
        model_kwargs={"device": device, "torch_dtype": "float32" }
    )

embeddings = load_embeddings()

# 🧠 Connect each collection with specific retriever
syllabus_vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace=None,
    collection_name="syllabus"
)
chapters_vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace=None,
    collection_name="chapters"
)
previous_vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace=None,
    collection_name="previous_papers"
)

# 🎯 Define retrievers with weights
weighted_retrievers = [
    (previous_vectorstore.as_retriever(search_kwargs={"k": 15}), 0.5),
    (chapters_vectorstore.as_retriever(search_kwargs={"k": 10}), 0.5),
    (syllabus_vectorstore.as_retriever(search_kwargs={"k": 5}), 0.2),
]

# Weighted retriever function
def weighted_multi_retriever(query):
    all_docs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Retrieve documents in parallel from all retrievers
        future_to_retriever = {
            executor.submit(retriever.get_relevant_documents, query): (retriever, weight)
            for retriever, weight in weighted_retrievers  # capture weight here
        }
        for future in concurrent.futures.as_completed(future_to_retriever):
            retriever, weight = future_to_retriever[future]  # retrieve the retriever and weight
            docs = future.result()
            for doc in docs:
                doc.metadata["weight"] = weight  # assign the weight to the document
            all_docs.extend(docs)
    
    # Sort the documents by score * weight
    sorted_docs = sorted(all_docs, key=lambda d: d.metadata.get("score", 1.0) * d.metadata.get("weight", 1.0), reverse=True)
    return sorted_docs[:30]


# 🧠 Helper to tag and extract question info from retrieved documents
def preprocess_context(docs):
    cleaned = []
    for doc in docs:
        text = doc.page_content

        # Remove noisy filler instructions
        text = re.sub(r"(?i)(attempt any|write short note|note:|instructions:)[^\n]*", "", text)
        text = re.sub(r"\n{2,}", "\n", text).strip()

        q_type = "theory"
        if re.search(r"\bcalculate|find|compute|determine|solve\b", text, re.I):
            q_type = "numerical"
        elif re.search(r"\bMCQ|multiple choice|\(a\)|\(b\)", text, re.I):
            q_type = "MCQ"

        marks_match = re.search(r"(\d{1,2})\s*marks", text, re.I)
        marks = marks_match.group(1) if marks_match else "NA"

        source = doc.metadata.get("source", "Unknown Source")
        tag = f"[SOURCE: {source}] [TYPE: {q_type.upper()}] [MARKS: {marks}]\n"
        cleaned.append(tag + text)
    return "\n\n".join(cleaned)

# Dynamic prompt template selection based on question intent
def choose_prompt_template(question: str) -> PromptTemplate:
    trend_keywords = [
        "trend", "important", "predict", "strategy", "frequent", "mostly asked",
        "chapter wise", "topic wise", "score", "focus areas", "expected questions"
    ]
    if any(word in question.lower() for word in trend_keywords):
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
                You are a CA Exam Strategy Assistant specialized in Goods and Services Tax (GST).
Using the provided document sections and the user query, generate a strategy-oriented, predictive, and insight-rich response.

Return:

1. **Important & Frequent Topics**
   - Topic name
   - Frequency (e.g., 5/7 papers)
   - Last appearance (month/year)
   - Type (theory/numerical/MCQ)
   - Source reference (paper name/year/module)

2. **Predicted Questions**
   - Likely question phrasing
   - Confidence %
   - Last seen year and marks
   - Suggested answer (if available)

3. **Strategy Table**
   | Topic | Importance | Frequency | Difficulty | Appearance % | Avg Marks | Next Exam Prediction % |

4. **Study Tips**
   - Order of topics
   - Time required per topic
   - Focus type: Theory / Practical / MCQs
   - Tips for tough areas

5. **Exclusions**
   - List irrelevant or repeated content worth skipping

6. **Optional Graphs** (Streamlit only)

Context:
{context}

Question:
{question}

Final strategy-rich response:
"""
        )
    else:
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful assistant for CA exam preparation specialized in Goods and Services Tax (GST).
Use the provided content to generate a focused and informative answer to the user's query with in depth explaination of underlying topics.
Also try to use common world analogies while generating the answer for better understanding. Try to use numerical explainations if possible for a topic.

Respond clearly and use markdown formatting if needed. Exclude any noisy or irrelevant info.

Context:
{context}

Question:
{question}

Answer:


"""
        )

CACHE = {}

def get_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()

def build_chain_for_question(question: str):
    prompt_template = choose_prompt_template(question)

    def retrieve_and_clean(inputs):
        docs = weighted_multi_retriever(inputs["question"])
        print(f"🔎 Retrieved {len(docs)} documents")
        return preprocess_context(docs)

    return (
        {"context": retrieve_and_clean, "question": lambda x: x["question"]} 
        | prompt_template 
        | llm
        | StrOutputParser()
    )

# Streamlit-specific logic
def run_streamlit_app():
    import streamlit as st

    st.title("CA Exam Strategy Assistant")

    query = st.text_input("Ask your question about GST syllabus, chapters or papers:")

    if query:
        cache_key = get_cache_key(query)
        if cache_key in CACHE:
            st.write("\n✅ Cached Result:")
            st.write(CACHE[cache_key])
        else:
            qa_chain = build_chain_for_question(query)
            result = qa_chain.invoke({"question": query})
            CACHE[cache_key] = result
            st.write("\n📍 Answer:")
            st.write(result)

# Main logic to detect Streamlit vs CLI mode
if __name__ == "__main__":
    try:
        import streamlit as st
        if st.runtime.exists():
            run_streamlit_app()
        else:
            raise RuntimeError
    except (ImportError, RuntimeError):
        # CLI mode
        print("Ask your question about GST syllabus, chapters or papers:")
        while True:
            query = input("\n> ")
            if query.lower() in ["exit", "quit"]:
                break
            cache_key = get_cache_key(query)
            if cache_key in CACHE:
                print("\n✅ Cached Result:")
                print(CACHE[cache_key])
            else:
                qa_chain = build_chain_for_question(query)
                result = qa_chain.invoke({"question": query})
                CACHE[cache_key] = result
                print("\n📍 Answer:")
                print(result)
