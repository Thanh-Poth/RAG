from functools import lru_cache
from transformers import pipeline

GENERATOR_MODEL_NAME = "google/flan-t5-small"

_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline("text2text-generation", model=GENERATOR_MODEL_NAME)
    return _generator


def generate_answer(question: str, context: str, max_new_tokens: int = 256) -> str:
    prompt = (
        "Réponds à la question uniquement à partir du contexte fourni. "
        "Si le contexte ne permet pas de répondre, dis que tu ne sais pas.\n\n"
        f"Contexte :\n{context}\n\n"
        f"Question : {question}\n"
        "Réponse :"
    )

    generator = _get_generator()
    output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return output[0]["generated_text"].strip()
