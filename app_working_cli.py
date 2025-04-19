from langchain_astradb import AstraDBVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import os
import hashlib
import re
import asyncio

# 🌱 Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

# 🧠 Load LLM and Embeddings once
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# 🔗 Connect to multiple vector stores
def get_vectorstore(collection_name):
    return AstraDBVectorStore(
        embedding=embeddings,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        namespace=None,
        collection_name=collection_name
    ).as_retriever

syllabus_retriever = get_vectorstore("syllabus")(search_kwargs={"k": 5})
chapters_retriever = get_vectorstore("chapters")(search_kwargs={"k": 10})
previous_retriever = get_vectorstore("previous_papers")(search_kwargs={"k": 15})

weighted_retrievers = [
    (previous_retriever, 0.5),
    (chapters_retriever, 0.3),
    (syllabus_retriever, 0.2),
]

# 🧠 Document Cleaner & Tagger
def preprocess_context(docs):
    cleaned = []
    for doc in docs:
        text = doc.page_content

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

# 🚀 Async Weighted Retriever for better speed
async def weighted_multi_retrieve_async(query: str):
    all_docs = []

    async def get_docs(retriever, weight):
        docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
        for doc in docs:
            doc.metadata["weight"] = weight
        return docs

    tasks = [get_docs(r, w) for r, w in weighted_retrievers]
    results = await asyncio.gather(*tasks)

    for docs in results:
        all_docs.extend(docs)

    sorted_docs = sorted(all_docs, key=lambda d: d.metadata.get("score", 1.0) * d.metadata.get("weight", 1.0), reverse=True)
    return sorted_docs[:30]

# 🧠 Smart prompt selector
def choose_prompt_template(question: str) -> PromptTemplate:
    strategy_keywords = [
        "trend", "important", "predict", "strategy", "frequent", "mostly asked",
        "chapter wise", "topic wise", "score", "focus areas", "expected questions"
    ]
    if any(word in question.lower() for word in strategy_keywords):
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a CA Exam Strategy Assistant specialized in Goods and Services Tax (GST).
Using the provided document sections and the user query, generate a strategy-oriented, predictive, and insight-rich response.

Return:

1. **Important & Frequent Topics**
2. **Predicted Questions**
3. **Strategy Table**
4. **Study Tips**
5. **Exclusions**
6. **Optional Graphs** (in Streamlit)

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
Use the provided content to generate a focused and informative answer to the user's query.

Respond clearly and use markdown formatting if needed. Exclude any noisy or irrelevant info.

Context:
{context}

Question:
{question}

Answer:
"""
        )

# 🧠 Cache Layer
CACHE = {}
def get_cache_key(query): return hashlib.md5(query.encode()).hexdigest()

# 🧠 Chain Builder
async def run_chain(query: str) -> str:
    cache_key = get_cache_key(query)
    if cache_key in CACHE:
        return f"✅ Cached Result:\n{CACHE[cache_key]}"

    docs = await weighted_multi_retrieve_async(query)
    print(f"🔎 Retrieved {len(docs)} documents")

    cleaned_context = preprocess_context(docs)
    prompt_template = choose_prompt_template(query)
    final_prompt = prompt_template.format(context=cleaned_context, question=query)

    result = StrOutputParser().invoke(llm.invoke(final_prompt))
    CACHE[cache_key] = result
    return f"📍 Answer:\n{result}"

# 🧪 Entry Point
if __name__ == "__main__":
    print("🔰 Ask your question about GST syllabus, chapters or past papers:")
    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit"]:
            break
        response = asyncio.run(run_chain(query))
        print(response)
