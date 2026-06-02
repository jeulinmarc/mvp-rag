"""
LLM client supporting multiple providers via OpenAI-compatible APIs.
Switch provider with the LLM_PROVIDER env variable: "ollama" (default) or "nebius".
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROVIDER_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # dummy, Ollama ignores it but the SDK requires it
        "model": "qwen2.5:7b",
    },
    "nebius": {
        "base_url": "https://api.studio.nebius.com/v1/",
        "api_key": os.getenv("NEBIUS_API_KEY", ""),
        "model": "Qwen/Qwen3-14B",
    },
}

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
TEMPERATURE = 0.2
MAX_TOKENS = 1024

_client = None


def get_client() -> OpenAI:
    """Lazy-init the OpenAI client for the configured provider."""
    global _client
    if _client is None:
        if PROVIDER not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{PROVIDER}'. Choose from: {list(PROVIDER_CONFIG)}"
            )
        cfg = PROVIDER_CONFIG[PROVIDER]
        if not cfg["api_key"]:
            raise ValueError(
                f"Provider '{PROVIDER}' requires an API key. "
                f"Check your .env file."
            )
        _client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return _client


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """
    Assemble system + user messages for a RAG prompt.
    `chunks` is the output of retrieve.retrieve().
    """
    context_blocks = []
    for c in chunks:
        header = f"[Source: {c['filename']}, page {c['page']}]"
        context_blocks.append(f"{header}\n{c['text']}")
    context = "\n\n".join(context_blocks)

    system_message = (
        "Tu es un assistant qui répond aux questions UNIQUEMENT à partir du contexte fourni. "
        "Si la réponse n'est pas dans le contexte, réponds exactement : "
        "\"Je ne trouve pas cette information dans les documents fournis.\" "
        "Cite tes sources entre crochets sous la forme [filename, page X]."
    )

    user_message = (
        f"Voici des extraits de documents :\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Question : {question}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def ask(question: str, chunks: list[dict]) -> str:
    """Send a RAG prompt to the configured LLM, return the answer text."""
    client = get_client()
    model = PROVIDER_CONFIG[PROVIDER]["model"]
    messages = build_messages(question, chunks)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Mini-test sans Qdrant: on simule des chunks à la main
    fake_chunks = [
        {
            "filename": "test_doc.txt",
            "page": 1,
            "text": "Le projet Eigenmind a été lancé en 2024 par Merlin Intelligence. "
                    "Il combine RAG classique et analyse spectrale de graphes.",
        },
        {
            "filename": "test_doc.txt",
            "page": 2,
            "text": "L'analyse spectrale repose sur le calcul du Laplacien du graphe "
                    "de similarité entre chunks.",
        },
    ]

    print(f"Provider actif : {PROVIDER}")
    print(f"Modèle         : {PROVIDER_CONFIG[PROVIDER]['model']}\n")

    question = "Quand Eigenmind a-t-il été lancé et par qui ?"
    print(f"Question : {question}\n")
    print("Réponse  :")
    print(ask(question, fake_chunks))