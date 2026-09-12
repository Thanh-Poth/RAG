import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.65
DEFAULT_PERSIST_DIR = "vector_db/"

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
_vectorstore = None


def _get_vectorstore(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        if not os.path.isdir(persist_directory) or not os.listdir(persist_directory):
            raise RuntimeError(
                f"Base vectorielle introuvable ou vide dans '{persist_directory}'. "
                "Lance d'abord 'python -m src.ingest' pour l'alimenter."
            )
        _vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=_embeddings,
        )
    return _vectorstore


def get_relevant_context(
    query: str,
    threshold: float = DEFAULT_THRESHOLD,
    k: int = 5,
    persist_directory: str = DEFAULT_PERSIST_DIR,
):
    vectorstore = _get_vectorstore(persist_directory)
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    filtered_results = [doc for doc, score in results if score >= threshold]

    if not filtered_results:
        return None

    context = "\n\n".join(doc.page_content for doc in filtered_results)
    return context
