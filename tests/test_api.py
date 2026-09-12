import main


def test_ask_endpoint_valid_context(client, monkeypatch):
    monkeypatch.setattr(
        main, "get_relevant_context", lambda question, threshold=0.65: "Contexte factice sur le backtesting MQL5."
    )
    monkeypatch.setattr(
        main, "generate_answer", lambda question, context: "Réponse générée à partir du contexte."
    )

    payload = {"question": "Comment lancer le script de backtesting MQL5 ?"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Réponse générée à partir du contexte."


def test_safety_similarity_threshold(client, monkeypatch):
    monkeypatch.setattr(main, "get_relevant_context", lambda question, threshold=0.65: None)

    payload = {"question": "Quelle est la recette de la tartiflette ?"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "Aucune information pertinente" in data.get("answer", "")


def test_ask_endpoint_missing_field(client):
    response = client.post("/ask", json={"query": "champ incorrect"})
    assert response.status_code == 422


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
