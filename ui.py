# ui.py
import sys
import os

# 🩹 Clean PyTorch module from streamlit's internal inspection
for mod in list(sys.modules):
    if mod.startswith("torch"):
        del sys.modules[mod]

# 🔒 Disable Streamlit file watcher (recommended for stable apps)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["RUNNING_STREAMLIT"] = "1"

import streamlit as st

import streamlit as st
import speech_recognition as sr
import tempfile

from retrievers import create_retrievers
from prompts import choose_prompt_template
from core import run_query  # You should have a centralized function to run RAG QA

def transcribe_audio(language='en-IN'):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = recognizer.listen(source, timeout=5)
            with st.spinner("Transcribing..."):
                text = recognizer.recognize_google(audio, language=language)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError:
            st.error("API unavailable.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    return ""

def main():
    st.set_page_config(page_title="CA Exam Strategy Assistant", layout="centered")
    st.title("📘 CA Exam Strategy Assistant")

    subject = st.selectbox("Select Subject", ["GST", "Audit"])
    language_option = st.radio("Select Input Language", ["English", "Hindi"])
    input_method = st.radio("How would you like to enter your query?", ["Text", "Speech"])

    if "query" not in st.session_state:
        st.session_state.query = ""

    if input_method == "Text":
        st.session_state.query = st.text_input("Enter your question")
    else:
        lang_code = "en-IN" if language_option == "English" else "hi-IN"
        if st.button("🎤 Speak Now"):
            transcript = transcribe_audio(language=lang_code)
            if transcript:
                st.session_state.query = transcript
                st.success(f"Transcribed: {transcript}")

    if st.session_state.query:
        retrievers, _ = create_retrievers(subject)
        prompt_template = choose_prompt_template(subject)

        # Centralized core logic (RAG response)
        with st.spinner("Thinking..."):
            response = run_query(st.session_state.query, retrievers, prompt_template)

        st.markdown("### 📌 Answer")
        st.write(response)

if __name__ == "__main__":
    main()
