import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

load_dotenv()

os.environ['ASTRA_DB_API_ENDPOINT']=os.getenv('ASTRA_DB_API_ENDPOINT')
os.environ['ASTRA_DB_APPLICATION_TOKEN']=os.getenv('ASTRA_DB_APPLICATION_TOKEN')
os.environ['ASTRA_DB_KEYSPACE']=os.getenv('ASTRA_DB_KEYSPACE')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

llm = ChatGroq(temperature=0.1, groq_api_key = os.environ['GROQ_API_KEY'], model_name="llama3-8b-8192")

embedding = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2"
)


# --- Load All Retrievers ---
collections = ["syllabus", "chapters", "previous_papers"]
retrievers = []

for name in collections:
    vstore = AstraDBVectorStore(
        collection_name=name,
        embedding=embedding,
        token=os.environ['ASTRA_DB_APPLICATION_TOKEN'],
        api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT'],
        namespace=os.environ['ASTRA_DB_KEYSPACE'],
    )
    retrievers.append(vstore.as_retriever())

# --- Combine All Retrievers ---
combined_retriever = EnsembleRetriever(retrievers=retrievers, weights=[1, 1, 2]) 

trend_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    
You are an expert CA Exam Strategy Assistant and Analyst powered by advanced data from syllabus, chapters, and previous papers (with suggested answers) stored in Astra DB.

🎯 Your goal: Help students **top the exam** using deep insights and predictions. Stick to **GST-related topics only**.

Here’s your approach:

1. 📚 **Topic/Chapter Analysis**  
   - Focus only on GST-related chapters and questions  
   - Highlight frequency of appearance (e.g., 6/7 times)  
   - Calculate trend score (%)  
   - Rate importance (Low / Medium / High / Critical)  
   - Mention source: `syllabus`, `chapters`, `previous_papers`

2. 📊 **Tabular Summary**  
   - Topic/Chapter  
   - Total Mentions  
   - Last Appeared  
   - Appearance %  
   - Prediction for Next Exam (%)  
   - Difficulty Level  
   - Study Priority

3. 🧠 **Top 3 Study Strategies**  
   - Topic flow (e.g., A ➝ B ➝ C)  
   - Time required  
   - Score boost prediction (%)  
   - Strategy for difficult areas  
   - Focus type (Theory/MCQ/Case-based)

4. 🎯 **Next Likely Questions**  
   - Predict 3–5 questions  
   - Tag with Confidence %, Last Seen, Type (MCQ/Descriptive)  
   - Source file + year  

5. 💡 **Suggested Answers (if present)**  
   - If asked, display detailed solutions from previous_papers

6. 📉 **Exclude Irrelevant Topics**  
   - Suggest noise topics not worth focusing on

7. 📈 **Optional Graph Instructions**  
   - (CLI can print summaries; Streamlit will show graphs later)

8. ⚡ **Caching**  
   - Mark cached responses for performance

9. 🧾 **Metadata**  
   - Include source file, paper name, year, module number

10. 🧠 **Closing Motivation**  
    - Tips, confidence boosters, or career context in CA

Only extract **topic-relevant**, **non-noisy**, **GST-specific** data.
Context:
{context}

Question:
{question}
"""
)

normal_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant for CA exam preparation. Use the provided content to give a clear and informative response to the user query.

Context:
{context}

Question:
{question}
"""
)


def choose_prompt(question):
    trend_keywords = ["trend", "important", "predict", "strategy", "frequent", "mostly asked", "chapter wise", "topic wise"]
    if any(word in question.lower() for word in trend_keywords):
        return trend_prompt_template
    return normal_prompt_template

# Cache Key
CACHE = {}

def get_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()

# Get Embedding for Text
def get_embedding(text):
    return embedding.embed_query(text)

# Visualize Topic Frequencies (CLI version)
def visualize_topic_frequencies(previous_papers_docs, chapters_docs, top_k=10):
    chapter_topic_texts = [doc.page_content[:250] for doc in chapters_docs]  # Topic from chapter heading
    chapter_embeddings = np.array([get_embedding(txt) for txt in chapter_topic_texts])

    qa_texts = [doc.page_content for doc in previous_papers_docs]
    qa_embeddings = np.array([get_embedding(txt) for txt in qa_texts])

    similarity_scores = cosine_similarity(qa_embeddings, chapter_embeddings)

    topic_matches = []
    for i, scores in enumerate(similarity_scores):
        best_idx = np.argmax(scores)
        topic_matches.append(chapter_topic_texts[best_idx])

    topic_counts = Counter(topic_matches)
    top_topics = topic_counts.most_common(top_k)

    for topic, count in top_topics:
        print(f"Topic: {topic[:50]}... - Frequency: {count}")

    return topic_counts

# Main CLI Interaction
def main():
    query = input("Ask your question about GST syllabus, chapters or papers: ")
    cache_key = get_cache_key(query)

    if cache_key in CACHE:
        print("✅ Using cached response")
        print(CACHE[cache_key])
    else:
        selected_prompt = trend_prompt_template if "trend" in query else normal_prompt_template
        chain = RetrievalQA(
            combine_documents_chain=StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=selected_prompt),
                document_variable_name="context"
            ),
            retriever=combined_retriever,
            return_source_documents=True
        )
        result = chain({"query": query})
        CACHE[cache_key] = result["result"]
        print(f"📍 Answer: {result['result']}")

        if "trend" in query or "frequent" in query:
            print("### 🔍 Chapter/Topic Frequency Chart")
            visualize_topic_frequencies(result["source_documents"])

if __name__ == "__main__":
    main()