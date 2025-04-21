from dotenv import load_dotenv
import os

def load_environment_variables():
    load_dotenv()
    return os.getenv("GROQ_API_KEY"), os.getenv("ASTRA_DB_APPLICATION_TOKEN"), os.getenv("ASTRA_DB_API_ENDPOINT")

import re

def clean_text(text: str) -> str:
    # Remove instructions, headers, copyright lines, etc.
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r"(instructions|note:|source:|icai|disclaimer|do not)", line, re.IGNORECASE):
            continue
        if line.lower().startswith("©"):
            continue
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def get_subject_list():
    return ["GST", "Audit"]
