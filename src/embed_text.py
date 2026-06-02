"""
Sentence embedding utility.
Loads sentence-transformers/all-MiniLM-L6-V2 lazily,
embeds one string or a batch, return L2-normallized 384-d vectors.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the model. First call downloads ~90MB, subsenquents calls reuse."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model

def embed(text: str | list[str]) -> np.ndarray:
    """
    Embed a single string or a list of strings.
    Returns:
        - shape (384,) for a single string
        - shape (N, 384) for a list of N strings
    Vectors are L2-normalized: cosine similarity = dot product.
    """
    model = get_model()
    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return vectors[0] if is_single else vectors
    
if __name__ == "__main__":
    v1 = embed("Le félin dort sur le canapé")
    v2 = embed("Un félin se repose sur un sofa")
    v3 = embed("La bourse a chuté de 5% aujourd'hui")

    sim_close = float(np.dot(v1, v2))
    sim_far = float(np.dot(v1, v3))

    print(f"Dimension: {v1.shape}")
    print(f"Norme de v1: {np.linalg.norm(v1):.4f}")
    print(f"Similarité (chat / félin):  {sim_close:.4f}")
    print(f"Similarité (chat / bourse): {sim_far:.4f}")