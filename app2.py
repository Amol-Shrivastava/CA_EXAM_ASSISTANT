from retrievers import create_retrievers
from prompts import choose_prompt_template
from llm import initialize_llm
from utils import clean_text
from langchain.chains.combine_documents import create_stuff_documents_chain 
import traceback
from langchain.chains import LLMChain
from ui import run_streamlit_app
import sys
import streamlit as st



def run_cli():
    print("Welcome to the CA Exam Assistant!\n")

    # User chooses the subject
    subject = input("Enter the subject (GST/Audit): ").strip().upper()
    if subject not in ["GST", "AUDIT"]:
        print("❌ Invalid input. Please enter GST or Audit.")
        return

    # Initialize LLM and retrievers
    llm = initialize_llm()
    retrievers, stores = create_retrievers(subject)

    print(f"\n🔍 Now chatting with {subject} assistant. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("🧠 Ask your CA Exam Question (or type 'exit'): ").strip()
            if question.lower() == "exit":
                print("Goodbye! 👋")
                break

            # Fetch documents
            syllabus_docs = stores["syllabus"].as_retriever(search_kwargs={"k": 5}).invoke(question)
            previous_docs = stores["previous_papers"].as_retriever(search_kwargs={"k": 7}).invoke(question)

            all_docs = syllabus_docs + previous_docs

            # print("Syllabus Docs Retrieved:", syllabus_docs)
            # print("Previous Papers Docs Retrieved:", previous_docs)

            syllabus_text = "\n\n".join([doc.page_content for doc in syllabus_docs])
            previous_papers_text = "\n\n".join([doc.page_content for doc in previous_docs])

            # print("Syllabus Text Content:", syllabus_text)
            # print("Previous Papers Text Content:", previous_papers_text)

            # context = syllabus_text + "\n" + previous_papers_text  # Combine relevant documents into context

            prompt = choose_prompt_template(question, subject, syllabus_text, previous_papers_text)


            # Build prompt using intent-based template
            # prompt = choose_prompt_template(
            #     question=question,
            #     subject=subject,
            #     syllabus_text=syllabus_text,
            #     previous_papers_text=previous_papers_text
            # )

            # Create chain with prompt
            # combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

            combine_chain = create_stuff_documents_chain(llm, prompt)

            # Step 5: Invoke the chain
            result = combine_chain.invoke({
                "context": all_docs,
                "question": question
            })

            # Combine all documents into a retrieval chain
            # all_docs = syllabus_docs + previous_docs
            # result = combine_chain.invoke({"question": question, "input_documents": all_docs})

            # Print result and sources
            print("\n📌 Answer:\n", result)
            print("\n📚 Sources:\n")
            for doc in all_docs:
                print(f"- {doc.metadata.get('source', 'Unknown')}")
        
        except Exception as e:
            # Print the standard error message
            print("\n❌ An error occurred:", e)
            
            # Print detailed traceback
            print("\nDetailed Traceback:")
            traceback.print_exc()

# if __name__ == "__main__":
#     run_cli()

def main():
    # Directly call Streamlit if this is a Streamlit run
    if 'streamlit' in sys.argv[0].lower():
        run_streamlit_app()
    else:
        run_cli()

if __name__ == "__main__":
    main()