from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def test_chunking_parameters():
    dummy_text = "Cours " * 200  # Texte fictif suffisamment long

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(dummy_text)

    assert len(chunks[0]) <= 500, "Le chunk dépasse la limite stricte de 500 caractères."
    assert len(chunks) > 1, "Le texte n'a pas été divisé correctement."


def test_embedding_model_output():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode("Modèle de prédiction de prix LSTM")

    assert len(embedding) == 384, "La dimension de l'embedding est incorrecte pour MiniLM-L6-v2."


def test_similarity_search_ranks_relevant_document_first(test_chroma_collection):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = {
        "doc_lstm": "Le modèle LSTM est utilisé pour prédire les séries temporelles de prix d'actifs.",
        "doc_cuisine": "La tartiflette est un plat savoyard à base de pommes de terre, reblochon et lardons.",
        "doc_meteo": "Le climat océanique se caractérise par des étés doux et des hivers frais.",
    }

    test_chroma_collection.add(
        ids=list(documents.keys()),
        embeddings=[model.encode(text).tolist() for text in documents.values()],
        documents=list(documents.values()),
    )

    query_embedding = model.encode("Comment fonctionne un LSTM pour prédire un prix ?").tolist()
    results = test_chroma_collection.query(query_embeddings=[query_embedding], n_results=1)

    assert results["ids"][0][0] == "doc_lstm"
