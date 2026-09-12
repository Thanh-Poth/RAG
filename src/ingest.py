import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def ingest_data(data_directory: str = "data/", persist_directory: str = "vector_db/"):
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(
            f"Le dossier '{data_directory}' n'existe pas. "
            "Place tes documents de cours dedans avant de lancer l'ingestion."
        )

    all_documents = []
    for pattern in ("**/*.py", "**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            data_directory,
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        all_documents.extend(loader.load())

    if not all_documents:
        print(f"Aucun document trouvé dans '{data_directory}'. Rien à ingérer.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )



    chunks = text_splitter.split_documents(all_documents)
    print(f"{len(chunks)} chunks générés à partir de {len(all_documents)} fichiers.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    print(f"Base vectorielle mise à jour et sauvegardée dans '{persist_directory}'.")


if __name__ == "__main__":
    ingest_data()
