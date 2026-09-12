from fastapi import FastAPI
from pydantic import BaseModel

from src.retriever import get_relevant_context
from src.llm_generator import generate_answer

app = FastAPI(title="RAG Trading Dashboard Assistant")

NO_CONTEXT_MESSAGE = (
    "Aucune information pertinente n'a été trouvée dans les documents fournis. "
    "Je ne peux pas répondre pour éviter toute hallucination."
)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_assistant(req: QuestionRequest):
    context = get_relevant_context(req.question, threshold=0.65)

    if not context:
        return {"answer": NO_CONTEXT_MESSAGE}

    answer = generate_answer(req.question, context)

    return {
        "context_used": context,
        "answer": answer,
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}
