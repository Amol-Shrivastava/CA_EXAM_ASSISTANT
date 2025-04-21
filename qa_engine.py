from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever

def get_qa_chain(subject: str, llm, retrievers: dict, prompt):
    combined_retriever = EnsembleRetriever(
        retrievers=[retrievers["syllabus"], retrievers["previous_papers"]],
        weights=[0.4, 0.6]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=combined_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain
