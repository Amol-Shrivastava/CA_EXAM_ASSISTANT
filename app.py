import os
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

prompt_template = PromptTemplate(
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


# ✅ Step 2: LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# ✅ Step 3: StuffDocumentsChain (this will stuff the retrieved docs into context)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

qa_chain = RetrievalQA(
    combine_documents_chain=stuff_chain,
    retriever=combined_retriever,  # Your merged retriever over syllabus, chapters, previous_papers
    return_source_documents=True
)


query = "Give me list of topics that are very important for the GST syllabus."
response = qa_chain({"query": query})

# ✅ View the result
print(response["result"])