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

# Global subject variable (used in choose_prompt_template)
SUBJECT = None

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2",  model_kwargs={"device": "cpu"})

# print(torch.__version__)
# print(torch.cuda.is_available())


# @st.cache_resource
# def load_embeddings():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     return HuggingFaceEmbeddings(
#         model_name="intfloat/e5-large-v2",
#         model_kwargs={"device": device}
#     )

# embeddings = load_embeddings()

# Create retrievers dynamically based on selected subject
def create_retrievers(subject):
    retrievers = {}
    if subject.lower() == "gst":
        retrievers["syllabus"] = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="gst_syllabus",
        ).as_retriever(search_kwargs={"k": 5})

        retrievers["previous_papers"] = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="gst_question_papers",
        ).as_retriever(search_kwargs={"k": 7})
    
    elif subject.lower() == "audit":
        retrievers["syllabus"] = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="audit_syllabus",
        ).as_retriever(search_kwargs={"k": 5})

        retrievers["previous_papers"] = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="audit_question_papers",
        ).as_retriever(search_kwargs={"k": 7})

    return retrievers


# Dynamic prompt template selection based on question intent
def choose_prompt_template(question: str) -> PromptTemplate:
    is_trend_query = any(word in question.lower() for word in [
        "trend", "important", "predict", "strategy", "frequent", "mostly asked",
        "chapter wise", "topic wise", "score", "focus areas", "expected questions",
        "paper analysis", "weightage", "frequency", "marks distribution", "past year"
    ])

    if SUBJECT == "GST":
        if is_trend_query:
            system_template = """
You are a strategic CA Exam Assistant focused exclusively on **GST** paper analysis.

Task:
- Provide chapter-wise and topic-wise frequency of questions from previous GST exams.
- Highlight trends, most asked topics, and mark distributions.
- Predict important topics and likely questions in upcoming exams based on past data.

Context:
{context}

Question:
{question}

Format:
- Use bullet points or tables.
- Mention paper name, year, marks, and type (MCQ/Theory/Numerical).
- Give motivational tips and smart revision strategies.

Present the result in **tabular format** (if possible) with the following columns:
| Chapter | Topic | Year | Paper | Marks | Type (MCQ/Theory/Numerical) | Frequency |
|---------|-------|------|-------|-------|-----------------------------|-----------|

After the table, give a brief summary with 1-2 preparation tips.

"""
        else:
            system_template = """
You are a highly knowledgeable CA Exam Assistant focused exclusively on the **GST** paper.

Task:
- Answer syllabus/topic-related questions clearly.
- Provide summaries, study notes, and brief conceptual explanations.
- If the topic is related to previous questions, also include any similar exam questions with their details.

Context:
{context}

Question:
{question}


Format:
- Use bullet points or short paragraphs.
- Include references to past questions (if any) with year, marks, and type.
- Suggest smart preparation tips for each topic.

Present the result in **tabular format** (if possible) with the following columns:
| Year | Paper | Question Summary | Marks | Type |
|------|-------|------------------|-------|------|

End with preparation tips or memory techniques.

"""
    elif SUBJECT == "AUDIT":
        if is_trend_query:
            system_template = """
You are a strategic CA Exam Assistant focused exclusively on **Audit** paper trends.

Task:
- Analyze previous Audit exam papers for frequently asked questions.
- Provide chapter-wise and topic-wise question frequency and marks distribution.
- Predict the most likely upcoming questions and important scoring areas.

Context:
{context}

Question:
{question}


Format:
- Use tables/bullets.
- Include paper name, exam session, marks, and question type.
- Add motivational tips and time-efficient study strategies.
"""
        else:
            system_template = """
You are an expert CA Exam Assistant for the **Audit** paper.

Task:
- Help students understand concepts from the Audit syllabus.
- Provide summaries, conceptual notes, and brief answers.
- If the topic is present in past questions, also include those examples with details.

Format:
- Use simple explanations and bullet points.
- Reference relevant past questions with marks and exam session (if available).
- Include preparation tips for better scoring.
"""
    else:
        system_template = """
            You are a helpful assistant for CA exam preparation who can give out answer in easy explanatory format for ca aspirants.

            Context:
            {context}

            Question:
            {question}
        
        """

    # return PromptTemplate(input_variables=["question", "context"], template=system_template + "\n\nContext:\n{context}\n\nQuestion: {question}")

    return PromptTemplate(
        input_variables=["question", "context"],
        template=system_template.strip()
    )


# Build RetrievalQA chain for a given question
def build_chain_for_question(question: str, retrievers: dict):
    prompt_template = choose_prompt_template(question)

    # Use previous_papers retriever if trend-based, else syllabus
    is_trend_query = any(word in question.lower() for word in [
        "trend", "important", "predict", "strategy", "frequent", "mostly asked",
        "chapter wise", "topic wise", "score", "focus areas", "expected questions",
        "paper analysis", "weightage", "frequency", "marks distribution", "past year", "previous years", 
        "previous papers"
    ])

    retriever = retrievers["previous_papers"] if is_trend_query else retrievers["syllabus"]

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return chain

# CLI Interface
def run_cli():
    global SUBJECT
    print("🌟 Welcome to the CA Exam Assistant (GST / Audit)")
    while True:
        SUBJECT = input("Enter the subject (GST/Audit): ").strip().upper()
        if SUBJECT in ["GST", "AUDIT"]:
            break
        else:
            print("❌ Invalid input. Please enter GST or Audit.")

    retrievers = create_retrievers(SUBJECT)

    print(f"\n🔍 Now chatting with {SUBJECT} assistant. Type 'exit' to quit.\n")
    while True:
        query = input("🧠 Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Good luck with your studies!")
            break

        chain = build_chain_for_question(query, retrievers)
        response = chain.invoke({"question": query})
        print("\n📝 Answer:\n", response["result"])
        print("-" * 50)

# Entry point
if __name__ == "__main__":
    run_cli()


# # 🧠 Connect each collection with specific retriever
# syllabus_vectorstore = AstraDBVectorStore(
#     embedding=embeddings,
#     token=ASTRA_DB_APPLICATION_TOKEN,
#     api_endpoint=ASTRA_DB_API_ENDPOINT,
#     namespace=None,
#     collection_name="syllabus"
# )
# chapters_vectorstore = AstraDBVectorStore(
#     embedding=embeddings,
#     token=ASTRA_DB_APPLICATION_TOKEN,
#     api_endpoint=ASTRA_DB_API_ENDPOINT,
#     namespace=None,
#     collection_name="chapters"
# )
# previous_vectorstore = AstraDBVectorStore(
#     embedding=embeddings,
#     token=ASTRA_DB_APPLICATION_TOKEN,
#     api_endpoint=ASTRA_DB_API_ENDPOINT,
#     namespace=None,
#     collection_name="previous_papers"
# )

# # 🎯 Define retrievers with weights
# weighted_retrievers = [
#     (previous_vectorstore.as_retriever(search_kwargs={"k": 15}), 0.5),
#     (chapters_vectorstore.as_retriever(search_kwargs={"k": 10}), 0.5),
#     (syllabus_vectorstore.as_retriever(search_kwargs={"k": 5}), 0.2),
# ]

# # Weighted retriever function
# def weighted_multi_retriever(query):
#     all_docs = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Retrieve documents in parallel from all retrievers
#         future_to_retriever = {
#             executor.submit(retriever.get_relevant_documents, query): (retriever, weight)
#             for retriever, weight in weighted_retrievers  # capture weight here
#         }
#         for future in concurrent.futures.as_completed(future_to_retriever):
#             retriever, weight = future_to_retriever[future]  # retrieve the retriever and weight
#             docs = future.result()
#             for doc in docs:
#                 doc.metadata["weight"] = weight  # assign the weight to the document
#             all_docs.extend(docs)
    
#     # Sort the documents by score * weight
#     sorted_docs = sorted(all_docs, key=lambda d: d.metadata.get("score", 1.0) * d.metadata.get("weight", 1.0), reverse=True)
#     return sorted_docs[:30]


# # 🧠 Helper to tag and extract question info from retrieved documents
# def preprocess_context(docs):
#     cleaned = []
#     for doc in docs:
#         text = doc.page_content

#         # Remove noisy filler instructions
#         text = re.sub(r"(?i)(attempt any|write short note|note:|instructions:)[^\n]*", "", text)
#         text = re.sub(r"\n{2,}", "\n", text).strip()

#         q_type = "theory"
#         if re.search(r"\bcalculate|find|compute|determine|solve\b", text, re.I):
#             q_type = "numerical"
#         elif re.search(r"\bMCQ|multiple choice|\(a\)|\(b\)", text, re.I):
#             q_type = "MCQ"

#         marks_match = re.search(r"(\d{1,2})\s*marks", text, re.I)
#         marks = marks_match.group(1) if marks_match else "NA"

#         source = doc.metadata.get("source", "Unknown Source")
#         tag = f"[SOURCE: {source}] [TYPE: {q_type.upper()}] [MARKS: {marks}]\n"
#         cleaned.append(tag + text)
#     return "\n\n".join(cleaned)

# # Dynamic prompt template selection based on question intent
# def choose_prompt_template(question: str) -> PromptTemplate:
#     trend_keywords = [
#         "trend", "important", "predict", "strategy", "frequent", "mostly asked",
#         "chapter wise", "topic wise", "score", "focus areas", "expected questions"
#     ]
#     if any(word in question.lower() for word in trend_keywords):
#         return PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
#                 You are a CA Exam Strategy Assistant specialized in Goods and Services Tax (GST).
# Using the provided document sections and the user query, generate a strategy-oriented, predictive, and insight-rich response.

# Return:

# 1. **Important & Frequent Topics**
#    - Topic name
#    - Frequency (e.g., 5/7 papers)
#    - Last appearance (month/year)
#    - Type (theory/numerical/MCQ)
#    - Source reference (paper name/year/module)

# 2. **Predicted Questions**
#    - Likely question phrasing
#    - Confidence %
#    - Last seen year and marks
#    - Suggested answer (if available)

# 3. **Strategy Table**
#    | Topic | Importance | Frequency | Difficulty | Appearance % | Avg Marks | Next Exam Prediction % |

# 4. **Study Tips**
#    - Order of topics
#    - Time required per topic
#    - Focus type: Theory / Practical / MCQs
#    - Tips for tough areas

# 5. **Exclusions**
#    - List irrelevant or repeated content worth skipping

# 6. **Optional Graphs** (Streamlit only)

# Context:
# {context}

# Question:
# {question}

# Final strategy-rich response:
# """
#         )
#     else:
#         return PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
# You are a helpful assistant for CA exam preparation specialized in Goods and Services Tax (GST).
# Use the provided content to generate a focused and informative answer to the user's query with in depth explaination of underlying topics.
# Also try to use common world analogies while generating the answer for better understanding. Try to use numerical explainations if possible for a topic.

# Respond clearly and use markdown formatting if needed. Exclude any noisy or irrelevant info.

# Context:
# {context}

# Question:
# {question}

# Answer:


# """
#         )

# CACHE = {}

# def get_cache_key(query):
#     return hashlib.md5(query.encode()).hexdigest()

# def build_chain_for_question(question: str):
#     prompt_template = choose_prompt_template(question)

#     def retrieve_and_clean(inputs):
#         docs = weighted_multi_retriever(inputs["question"])
#         print(f"🔎 Retrieved {len(docs)} documents")
#         return preprocess_context(docs)

#     return (
#         {"context": retrieve_and_clean, "question": lambda x: x["question"]} 
#         | prompt_template 
#         | llm
#         | StrOutputParser()
#     )

# # Streamlit-specific logic
# def run_streamlit_app():
#     import streamlit as st

#     st.title("CA Exam Strategy Assistant")

#     query = st.text_input("Ask your question about GST syllabus, chapters or papers:")

#     if query:
#         cache_key = get_cache_key(query)
#         if cache_key in CACHE:
#             st.write("\n✅ Cached Result:")
#             st.write(CACHE[cache_key])
#         else:
#             qa_chain = build_chain_for_question(query)
#             result = qa_chain.invoke({"question": query})
#             CACHE[cache_key] = result
#             st.write("\n📍 Answer:")
#             st.write(result)

# # Main logic to detect Streamlit vs CLI mode
# if __name__ == "__main__":
#     try:
#         import streamlit as st
#         if st.runtime.exists():
#             run_streamlit_app()
#         else:
#             raise RuntimeError
#     except (ImportError, RuntimeError):
#         # CLI mode
#         print("Ask your question about GST syllabus, chapters or papers:")
#         while True:
#             query = input("\n> ")
#             if query.lower() in ["exit", "quit"]:
#                 break
#             cache_key = get_cache_key(query)
#             if cache_key in CACHE:
#                 print("\n✅ Cached Result:")
#                 print(CACHE[cache_key])
#             else:
#                 qa_chain = build_chain_for_question(query)
#                 result = qa_chain.invoke({"question": query})
#                 CACHE[cache_key] = result
#                 print("\n📍 Answer:")
#                 print(result)
