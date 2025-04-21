from langchain_astradb import AstraDBVectorStore
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as e:
    import traceback
    print("⚠️ Error importing HuggingFaceEmbeddings:")
    traceback.print_exc()

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={"device": "cpu"})

def create_retrievers(subject: str):
    retrievers = {}
    stores = {}

    if subject.lower() == "gst":
        # syllabus_store = AstraDBVectorStore(embedding=embeddings, collection_name="syllabus")
        syllabus_store = AstraDBVectorStore(embedding=embeddings, collection_name="chapters")
        previous_papers_store = AstraDBVectorStore(embedding=embeddings, collection_name="previous_papers") 
    elif subject.lower() == "audit":
        syllabus_store = AstraDBVectorStore(embedding=embeddings, collection_name="audit_syllabus")
        previous_papers_store = AstraDBVectorStore(embedding=embeddings, collection_name="audit_question_papers")
    else:
        raise ValueError("Subject must be 'gst' or 'audit'")

    retrievers["syllabus"] = syllabus_store.as_retriever(search_kwargs={"k": 5})
    retrievers["previous_papers"] = previous_papers_store.as_retriever(search_kwargs={"k": 7})

    stores["syllabus"] = syllabus_store
    stores["previous_papers"] = previous_papers_store

    return retrievers, stores
