"""
Qdrant ingestion: take chunks, embed them in batch, upsert into a collection.
Idempotent: re-running the same ingestion overwrites the same points.
"""
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from embed_text import embed, EMBEDDING_DIM

COLLECTION_NAME = "documents"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 64


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient, name: str = COLLECTION_NAME) -> None:
    """Create the collection if it does not exist."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Collection '{name}' créée (dim={EMBEDDING_DIM}, distance=COSINE).")


def chunk_id(chunk: dict) -> int:
    """Deterministic 64-bit int id from (filename, page, chunk_index)."""
    key = f"{chunk['filename']}::{chunk['page']}::{chunk['chunk_index']}"
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    # take the first 8 bytes as a signed int64, force positive
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1


def upsert_chunks(chunks: list[dict], collection: str = COLLECTION_NAME) -> int:
    """Embed all chunks and upsert them in batches. Returns number of points written."""
    if not chunks:
        return 0

    client = get_client()
    ensure_collection(client, collection)

    texts = [c["text"] for c in chunks]
    vectors = embed(texts)  # shape (N, 384), already L2-normalized

    assert vectors.shape[1] == EMBEDDING_DIM, (
        f"Embedding dim mismatch: got {vectors.shape[1]}, expected {EMBEDDING_DIM}"
    )

    points = [
        PointStruct(
            id=chunk_id(chunk),
            vector=vectors[i].tolist(),
            payload={
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        client.upsert(collection_name=collection, points=batch)

    return len(points)


if __name__ == "__main__":
    import sys
    from load_pdf import load_and_chunk

    if len(sys.argv) < 2:
        print("Usage: python store_chunks.py <path/to/file.pdf>")
        sys.exit(1)

    chunks = load_and_chunk(sys.argv[1])
    print(f"Chunks à ingérer : {len(chunks)}")

    n = upsert_chunks(chunks)
    print(f"Points upserted   : {n}")

    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    print(f"Total dans '{COLLECTION_NAME}' : {info.points_count} points")