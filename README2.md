# Assistant RAG — Cours ESILV

Assistant capable de répondre à des questions sur des supports de cours (ou tout
corpus de documents texte) en s'appuyant uniquement sur le contenu fourni, avec
un garde-fou anti-hallucination : si aucun document pertinent n'est trouvé,
l'assistant répond explicitement qu'il ne sait pas plutôt que d'inventer.

## Architecture

```
Question utilisateur
        │
        ▼
  FastAPI (/ask)
        │
        ▼
  Retriever (Chroma + sentence-transformers/all-MiniLM-L6-v2)
        │  filtre par seuil de similarité (0.65)
        ▼
  contexte pertinent trouvé ? ──non──► "Aucune information pertinente..."
        │ oui
        ▼
  Générateur (google/flan-t5-small, local, CPU)
        │
        ▼
  Réponse finale
```

## Stack

- **Embeddings** : `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)
- **Base vectorielle** : ChromaDB
- **Génération** : `google/flan-t5-small` via `transformers` (local, gratuit, CPU-friendly)
- **API** : FastAPI
- **Tests** : pytest, tests d'API isolés par monkeypatch + test d'intégration retrieval
- **CI** : GitHub Actions (tests à chaque push/PR)
- **Conteneurisation** : Docker

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

1. Placer les documents source (`.py`, `.md`, `.txt`) dans `data/`.
2. Générer la base vectorielle :
   ```bash
   python -m src.ingest
   ```
3. Lancer l'API :
   ```bash
   uvicorn main:app --reload
   ```
4. Interroger l'assistant :
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Comment fonctionne le backtesting MQL5 ?"}'
   ```

## Tests

```bash
pytest -v
```

## Docker

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 rag-assistant
```

## Limites connues

- Le modèle de génération (`flan-t5-small`) est volontairement petit pour tourner
  sans GPU ; la qualité de génération est donc plus limitée qu'avec un modèle
  plus gros ou une API payante (OpenAI, Anthropic...).
- Pas d'authentification sur l'endpoint `/ask` — à ajouter avant tout déploiement public.
