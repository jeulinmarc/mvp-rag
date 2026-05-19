"""
Retrieval: take a natural-language question, return the top-k most similar chunks.
"""
from qdrant_client import QdrantClient

from embed_text import embed
from store_chunks import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

TOP_K = 5


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def retrieve(
    question: str,
    k: int = TOP_K,
    collection: str = COLLECTION_NAME,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Embed the question and return the top-k most similar chunks.
    Each result is a dict: {score, filename, page, chunk_index, text}.
    """
    client = get_client()
    query_vector = embed(question).tolist()

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=k,
        score_threshold=score_threshold,
        with_payload=True,
    ).points

    return [
        {
            "score": point.score,
            "filename": point.payload["filename"],
            "page": point.payload["page"],
            "chunk_index": point.payload["chunk_index"],
            "text": point.payload["text"],
        }
        for point in results
    ]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQuestion : {question}\n")

    hits = retrieve(question, k=TOP_K)

    if not hits:
        print("Aucun résultat. La collection est peut-être vide.")
        sys.exit(0)

    for i, hit in enumerate(hits, start=1):
        print(f"--- Résultat #{i} (score={hit['score']:.4f}) ---")
        print(f"Source : {hit['filename']}, page {hit['page']}, chunk {hit['chunk_index']}")
        preview = hit["text"][:300].replace("\n", " ")
        print(f"Texte  : {preview}...\n")