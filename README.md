# RAG Assistant: Grounded Question Answering over Course Material

> A Retrieval-Augmented Generation (RAG) assistant that answers questions **only from the documents you give it**, and says "I don't know" instead of hallucinating. It runs fully locally on CPU, is served through a REST API, tested, containerized, and checked by CI.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)

---

## Porject Overview

| | |
|---|---|
| **Problem** | LLMs invent answers when they lack context. For course notes or internal docs, a wrong answer is worse than no answer. |
| **Solution** | Retrieve the most relevant passages with semantic search; answer only if a passage passes a similarity threshold, otherwise refuse explicitly. |
| **Constraints I designed for** | No GPU, no paid API: every model runs locally on CPU. |
| **Skills** | NLP / information retrieval, vector databases, API design, automated testing, CI, Docker, and honest documentation of limits. |

---

## Architecture

```
User question
      |
      v
FastAPI  POST /ask
      |
      v
Retriever: ChromaDB + sentence-transformers/all-MiniLM-L6-v2 (384-dim)
      |   keep only passages above similarity threshold (0.65)
      v
Relevant context found? --no--> "No relevant information found" (anti-hallucination guard)
      | yes
      v
Generator: google/flan-t5-small (local, CPU)
      |
      v
Final answer
```

**Anti-hallucination guard**: the similarity threshold (0.65) is a hard gate. If nothing relevant is retrieved, the generator is never called and the API answers that it does not know.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Small, fast, good quality for semantic search on CPU |
| Vector store | ChromaDB | Simple, local, persistent |
| Generation | `google/flan-t5-small` via `transformers` | Free and runs without a GPU |
| API | FastAPI | Typed, auto-documented endpoint |
| Tests | pytest | API tests isolated with monkeypatch + a retrieval integration test |
| CI | GitHub Actions | Tests run on every push and pull request |
| Packaging | Docker | One-command reproducible environment |

---

## Repository layout

```
src/              Ingestion and retrieval/generation logic
data/             Source documents (.py, .md, .txt)
vector_db/        Persisted Chroma index
tests/            pytest suite
main.py           FastAPI app
Dockerfile
.github/workflows CI pipeline
```

---

## Usage

```bash
pip install -r requirements.txt

# 1. Put your documents (.py, .md, .txt) in data/
# 2. Build the vector index
python -m src.ingest

# 3. Start the API
uvicorn main:app --reload

# 4. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does backtesting work in MQL5?"}'
```

Run the tests: `pytest -v`

Run with Docker:

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 rag-assistant
```

---

## Engineering decisions

- **Refuse rather than guess.** A threshold on retrieval similarity decides whether the model may answer at all.
- **Testable by design.** API tests monkeypatch the retriever and generator, so they are fast and deterministic. A separate integration test checks real retrieval.
- **Cost-free stack.** Every component is open source and local, which makes the project reproducible by anyone.

## Known limitations

- `flan-t5-small` is intentionally small, so answer quality is below larger models or paid APIs. The retrieval layer is model-agnostic and the generator can be swapped.
- The `/ask` endpoint has **no authentication**; add it before any public deployment.

## Roadmap

- Swap in a stronger generator (local or API-based).
- Add retrieval evaluation (precision / recall on a labelled question set).
- Add authentication and rate limiting.

---

## Skills

**ML / NLP**: embeddings, semantic search, retrieval-augmented generation, hallucination mitigation
**Backend**: FastAPI, REST design
**MLOps**: pytest, GitHub Actions CI, Docker, reproducible environments

---

## Author

**Thanh Son Potherat**: engineering student focused on quantitative finance, machine learning and data science.
[GitHub](https://github.com/Thanh-Poth) | [LinkedIn](https://www.linkedin.com/in/thanh-son-potherat-b793b333a/)
