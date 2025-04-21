# core.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llm import initialize_llm

def run_query(query, retrievers, prompt_template):
    combined_docs = []
    for retriever in retrievers.values():
        docs = retriever.get_relevant_documents(query)
        combined_docs.extend(docs)

    # You can optionally deduplicate or clean docs here

    llm = initialize_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=None,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain.run(input_documents=combined_docs, question=query)
