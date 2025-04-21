import logging
import streamlit as st
import speech_recognition as sr
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.prompt import Prompt
from llm import initialize_llm
from PIL import Image
import pytesseract
import tempfile
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import sounddevice as sd
import numpy as np
import speech_recognition as sr

# Setup logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

console = Console()

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={"device": "cpu"})

def create_retrievers(subject):
    retrievers, stores = {}, {}
    try:
        if subject.lower() == "gst":
            syllabus_store = AstraDBVectorStore(embedding=embeddings, collection_name="chapters")
            previous_papers_store = AstraDBVectorStore(embedding=embeddings, collection_name="previous_papers")
        elif subject.lower() == "audit":
            syllabus_store = AstraDBVectorStore(embedding=embeddings, collection_name="audit_syllabus")
            previous_papers_store = AstraDBVectorStore(embedding=embeddings, collection_name="audit_question_papers")
        else:
            raise ValueError("Subject must be 'gst' or 'audit'")

        retrievers["syllabus"] = syllabus_store.as_retriever(search_kwargs={"k": 5})
        retrievers["previous_papers"] = previous_papers_store.as_retriever(search_kwargs={"k": 7})
        stores.update({"syllabus": syllabus_store, "previous_papers": previous_papers_store})

    except Exception as e:
        logger.error("Error in create_retrievers: %s", e, exc_info=True)
        raise

    return retrievers, stores

def choose_prompt_template(subject: str, is_trend_query: bool) -> PromptTemplate:
    try:
        if subject == "GST":
            if is_trend_query:
                system_template = """
You are a strategic CA Exam Assistant focused exclusively on **GST** paper analysis.

🎯 **Your Objectives**:
- Provide **chapter-wise** and **topic-wise** frequency of questions from previous GST exams.
- Highlight **trends**, most asked topics, and **mark distributions**.
- Predict **important topics** and likely questions in upcoming exams based on past data.

**Context:**
{context}

**Question:**
{question}

**Response:**
- Use **bullet points** or **tables** for clarity.
- Provide **detailed references** from past papers:
    - **Year**
    - **Marks**
    - **Question Type**
- Conclude with:
    - **Preparation Tips**
    - **Motivational Strategies**
    - **Importance of Frequently Asked Topics**
"""
            else:
                system_template = """
You are an expert CA Exam Assistant focused exclusively on the **GST** paper.

🎯 **Your Objectives**:
- Deliver syllabus-aligned, detailed answers in **bullet-point** format.
- Maintain original technical phrasings; provide **clarified, layered explanations**.
- Refer to **exact past exam questions**, when applicable.
- If no match, **explain relevant concepts** with clarity and structure.

**Context:**
{context}

**Question:**
{question}

**Response:**
- **Direct Answer:** Elaborate explanation using original terms + layered clarity.
- **Detailed Breakdown:** Define key elements and technical terms.
- **Past Question Reference (if applicable):**
    - **Question Text**: [Exact past exam question]
    - **Marks**: [e.g., 4, 6, 8]
    - **Exam Attempt**: [e.g., May 2022]
    - **Question Type**: [Theory/Practical]
- **Study Tips:** Examples, memory tricks, or use-case-based explanation to aid understanding.
"""
            return PromptTemplate(input_variables=["context", "question"], template=system_template)

        elif subject == "Audit":
            if is_trend_query:
                system_template = """
You are a strategic CA Exam Assistant focused exclusively on **Audit** paper trends.

🎯 **Your Objectives**:
- Provide **chapter-wise** and **topic-wise** frequency of questions from previous Audit exams.
- Predict the **most important upcoming questions** based on past data.
- Analyze **question patterns** and highlight **frequently tested concepts**.

**Context:**
{context}

**Question:**
{question}

**Response:**
- Present **tables or bullet points** showing frequencies.
- Reference past exams using:
    - **Exam Year**
    - **Marks**
    - **Question Type**
- Include:
    - **Trend-based Predictions**
    - **Topic Importance Analysis**
    - **Focus Area Suggestions**
"""
            else:
                system_template = """
You are a **CA Exam Strategy Assistant for the Audit subject**.

🎯 **Your Objectives**:
- Provide **syllabus-aligned**, rich, structured responses.
- **Retain technical phrasing**, but build layered explanation on top of it.
- Refer to **exact past questions** if relevant.
- **Avoid hallucination** — keep it accurate, contextual, and relevant.

**Context:**
{context}

**Question:**
{question}

**Response:**
- **Direct Answer:** Clear, layered, technically correct.
- **In-depth Explanation:** Clarify complex parts in plain terms with examples.
- **Past Question Reference (if applicable):**
    - **Question Text**
    - **Marks**
    - **Attempt**
    - **Question Type**
- **Study Tips:** Real-world analogies or memory aids to reinforce understanding.
"""
            return PromptTemplate(input_variables=["context", "question"], template=system_template)

        else:
            system_template = """
You are an **intelligent assistant for CA exam preparation**.

🎯 **Your Objectives:**
- Deliver **syllabus-aligned**, **technically accurate** responses.
- Keep original **technical terms**, add clarity through structured explanation.
- Refer to past questions only if **directly relevant**.

**Context:**
{context}

**Question:**
{question}

**Response:**
- **Direct Answer:** Thorough, correct, and educational.
- **In-depth Explanation:** Break technical content with structured insight.
- **Past Question Reference (if applicable):**
    - **Question Text**
    - **Marks**
    - **Exam Attempt**
    - **Question Type**
- **Study Tips:** Personalized, contextual, and practical strategies.
"""
            return PromptTemplate(input_variables=["context", "question"], template=system_template)

    except Exception as e:
        logging.error("❌ Error in choose_prompt_template: %s", str(e))
        raise

strategy_prompt = PromptTemplate.from_template("""
You are a CA Exam Strategy Assistant for the subject: {subject}.
Based on the user's query, follow this format:
1. **Concept Explanation**
2. **Related Chapter/Topic**
3. **Past Exam Appearance (with year, paper)**
4. **Suggested Answer (if any)**
5. **Study Strategy**
6. **Prediction of Future Importance (% chance)**
7. **Passing Tips**

Query: {query}
""")


# def transcribe_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak now")
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             return recognizer.recognize_google(audio)
#         except sr.UnknownValueError:
#             st.error("Could not understand audio")
#         except sr.RequestError as e:
#             st.error(f"Speech recognition error: {e}")
#         except Exception as e:
#             logger.error("Speech-to-text error: %s", e, exc_info=True)

# Parameters for sounddevice
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds

def transcribe_speech():
    recognizer = sr.Recognizer()

    # Recording with sounddevice
    st.info("Listening... Speak now")
    try:
        # Record audio using sounddevice
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()  # Wait for the recording to finish
        
        # Convert the audio data to an audio file format for speech recognition
        audio = sr.AudioData(audio_data.tobytes(), SAMPLE_RATE, 2)  # Convert to audio data

        # Recognize speech using Google's API
        return recognizer.recognize_google(audio)
    
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
    except Exception as e:
        logger.error("Speech-to-text error: %s", e, exc_info=True)

def generate_pdf(response_text):
    # Create a BytesIO buffer to hold the PDF data
    buffer = BytesIO()
    
    # Create a canvas object and set up the PDF page
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Set the font and size
    c.setFont("Helvetica", 10)
    
    # Add the response text to the PDF (split by lines if it's too long)
    width, height = letter
    text_object = c.beginText(40, height - 40)
    text_object.setFont("Helvetica", 10)
    text_object.setTextOrigin(40, height - 40)
    
    for line in response_text.split("\n"):
        text_object.textLine(line)
    
    # Draw the text object onto the canvas
    c.drawText(text_object)
    
    # Save the canvas to the buffer
    c.showPage()
    c.save()

    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error("Failed to extract text from image")
        logger.error("Image OCR error: %s", e, exc_info=True)
        return ""
    
def run_cli():
    subject = Prompt.ask("Enter subject (GST/Audit)", default="GST")
    retrievers, _ = create_retrievers(subject)
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    prompt_template = choose_prompt_template(subject)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retrievers["syllabus"],
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    console.print("\n[bold green]CA Exam Assistant CLI is ready. Type your question below:[/bold green]")
    while True:
        query = Prompt.ask("\n[bold yellow]Question[/bold yellow]", default="What is time of supply under GST?")
        try:
            result = chain.invoke({"query": query})
            console.print("\n[bold cyan]Answer:[/bold cyan]", result["result"])
        except Exception as e:
            logger.error("CLI query error: %s", e, exc_info=True)
            console.print("[red]An error occurred while processing your question.[/red]")

def run_streamlit():
    st.set_page_config(page_title="📘 CA Exam Strategy Assistant", layout="centered")
    st.title("📘 CA Exam Strategy Assistant")
    subject = st.selectbox("Select Subject", ["GST", "Audit"])
    retrievers, _ = create_retrievers(subject)
    query_type = st.radio("Query Type", ["Conceptual", "Trend Analysis"])
    is_trend_query = query_type == "Trend Analysis"

    st.markdown("---")
    st.markdown("### 📤 Upload Image from Mobile or Desktop")
    image_file = st.file_uploader("Upload an image from camera or gallery", type=["jpg", "jpeg", "png"])
    extracted_text = ""

    if image_file:
        st.image(image_file, caption="Preview", use_column_width=True)
        if st.toggle("🔍 Show Extracted Text"):
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image_file)
                st.text_area("📝 Extracted Text", value=extracted_text, height=200)

    st.markdown("---")
    user_input = st.text_input("Ask your question or use extracted text")

    if extracted_text and not user_input:
        use_extracted = st.checkbox("Use extracted image text as question")
        if use_extracted:
            user_input = extracted_text

    mic_clicked = st.button("🎤 Speak your query")
    if mic_clicked:
        transcript = transcribe_speech()
        if transcript:
            user_input = transcript

    # if user_input:
    #     try:
    #         prompt_template = choose_prompt_template(subject, is_trend_query)
    #         llm = initialize_llm()
    #         retriever = retrievers["previous_papers"] if is_trend_query else retrievers["syllabus"]
    #         chain = RetrievalQA.from_chain_type(
    #             llm=llm,
    #             retriever=retriever,
    #             chain_type="stuff",
    #             chain_type_kwargs={"prompt": prompt_template},
    #             return_source_documents=True
    #         )
    #         with st.spinner("Thinking..."):
    #             result = chain.invoke({"query": user_input})
    #             st.markdown("### 📘 Answer")
    #             st.write(result["result"])
    #             st.markdown("---")
    #             st.markdown("#### 🧾 Sources")
    #             for i, doc in enumerate(result["source_documents"]):
    #                 st.markdown(f"**{i+1}.** `{doc.metadata.get('source', 'Unknown')}`")
    #     except Exception as e:
    #         logger.error("Streamlit query error: %s", e, exc_info=True)
    #         st.error("An error occurred while processing your query.")
    if user_input:
        try:
            prompt_template = choose_prompt_template(subject, is_trend_query)
            llm = initialize_llm()
            retriever = retrievers["previous_papers"] if is_trend_query else retrievers["syllabus"]
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True
            )
            result = chain.invoke({"query": user_input})
            response_text = result["result"]

            st.markdown("### 🧠 Response")
            st.write(response_text)

            # ✅ Create PDF and add download button
            pdf_data = generate_pdf(response_text)

            st.download_button(
                label="📥 Download Response as PDF",
                data=pdf_data,
                file_name="ca_exam_response.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error("An error occurred while generating the response.")
            logger.error("Streamlit query error: %s", e, exc_info=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        run_streamlit()