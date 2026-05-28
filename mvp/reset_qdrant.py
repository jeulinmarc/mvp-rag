"""Drop the Qdrant collection. Run when you want a clean slate."""
from qdrant_client import QdrantClient
from store_chunks import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' supprimée.")
else:
    print(f"Collection '{COLLECTION_NAME}' n'existait pas.")