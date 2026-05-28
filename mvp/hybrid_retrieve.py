"""
Hybrid retrieval: combine Qdrant dense top-k with graph signals.

Strategy:
  1. Fetch a wide dense top-K (K_DENSE_FETCH).
  2. Add Singular / Hinge / Theta nodes whose cosine to the question
     exceeds RELEVANCE_THRESHOLD, applying a type-specific boost.
  3. Rerank and return the top K_FINAL.

All special-node vectors are cached in RAM at first build.
"""
import numpy as np
from dataclasses import dataclass, field
from qdrant_client import QdrantClient

from embed_text import embed
from store_chunks import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

K_DENSE_FETCH = 15
K_FINAL = 5
RELEVANCE_THRESHOLD = 0.30

BOOST_SINGULAR = 0.10
BOOST_HINGE = 0.07
BOOST_THETA = 0.05


@dataclass
class HybridResult:
    """A retrieved chunk with its hybrid score and node type tags."""
    node_id: int
    score: float                 # final hybrid score
    base_cosine: float           # raw cosine to the question
    types: list[str] = field(default_factory=list)  # e.g. ['dense', 'singular']
    payload: dict = field(default_factory=dict)


class GraphAwareCache:
    """
    Caches vectors and payloads for Singular / Hinge / Theta nodes.
    Built once per session (or after corpus changes).
    """
    def __init__(self):
        self.singular_ids: set[int] = set()
        self.hinge_ids: set[int] = set()
        self.theta_ids: set[int] = set()
        self.vectors: dict[int, np.ndarray] = {}
        self.payloads: dict[int, dict] = {}

    def build(self, collection: str = COLLECTION_NAME) -> None:
        """Compute graph, decomposition, special nodes, and cache them in RAM."""
        from build_graph import build_graph_from_qdrant, fetch_all_points
        from spectral import compute_spectral_decomposition
        from singular import singular_node_ids
        from hinge import hinge_node_ids
        from theta import theta_node_ids

        print("  → Loading all points from Qdrant...")
        all_payloads, all_vectors = fetch_all_points(collection)

        print("  → Building similarity graph...")
        G = build_graph_from_qdrant(collection)

        print("  → Computing spectral decomposition...")
        decomp = compute_spectral_decomposition(G)

        print("  → Identifying Singular nodes...")
        self.singular_ids = singular_node_ids(G, decomp, top_fraction=0.15)

        print("  → Identifying Hinge nodes...")
        self.hinge_ids = hinge_node_ids(G, decomp, top_fraction=0.15)

        print("  → Identifying Theta nodes...")
        excluded = self.singular_ids | self.hinge_ids
        self.theta_ids = theta_node_ids(G, decomp, excluded=excluded)

        # Cache vectors and payloads for all special nodes
        special_ids = self.singular_ids | self.hinge_ids | self.theta_ids
        for payload, vec in zip(all_payloads, all_vectors):
            nid = payload["id"]
            if nid in special_ids:
                self.vectors[nid] = vec
                self.payloads[nid] = payload

        total = len(self.singular_ids) + len(self.hinge_ids) + len(self.theta_ids)
        print(
            f"  Cache built: {len(self.singular_ids)} Singular, "
            f"{len(self.hinge_ids)} Hinge, {len(self.theta_ids)} Theta "
            f"({total} unique vectors cached)"
        )

    def node_types(self, node_id: int) -> list[str]:
        """Return the types tagged for a given node."""
        types = []
        if node_id in self.singular_ids:
            types.append("singular")
        if node_id in self.hinge_ids:
            types.append("hinge")
        if node_id in self.theta_ids:
            types.append("theta")
        return types

    def boost_for_node(self, node_id: int) -> float:
        """Composite boost: sum of boosts for all types tagged."""
        boost = 0.0
        if node_id in self.singular_ids:
            boost += BOOST_SINGULAR
        if node_id in self.hinge_ids:
            boost += BOOST_HINGE
        if node_id in self.theta_ids:
            boost += BOOST_THETA
        return boost


def hybrid_retrieve(
    question: str,
    cache: GraphAwareCache,
    k_final: int = K_FINAL,
    k_dense_fetch: int = K_DENSE_FETCH,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    collection: str = COLLECTION_NAME,
) -> list[HybridResult]:
    """Perform hybrid retrieval using a prebuilt cache."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    q_vec = embed(question)

    # --- 1) Dense top-K ---
    dense_response = client.query_points(
        collection_name=collection,
        query=q_vec.tolist(),
        limit=k_dense_fetch,
        with_payload=True,
    ).points

    candidates: dict[int, HybridResult] = {}
    for p in dense_response:
        nid = p.id
        types = ["dense"] + cache.node_types(nid)
        boost = cache.boost_for_node(nid)
        candidates[nid] = HybridResult(
            node_id=nid,
            score=p.score + boost,
            base_cosine=p.score,
            types=types,
            payload=p.payload,
        )

    # --- 2) Add cached special nodes if relevant ---
    for nid, vec in cache.vectors.items():
        if nid in candidates:
            continue  # already counted via dense
        cos = float(np.dot(q_vec, vec))
        if cos < relevance_threshold:
            continue
        types = cache.node_types(nid)
        boost = cache.boost_for_node(nid)
        candidates[nid] = HybridResult(
            node_id=nid,
            score=cos + boost,
            base_cosine=cos,
            types=types,
            payload=cache.payloads[nid],
        )

    # --- 3) Sort and trim ---
    ranked = sorted(candidates.values(), key=lambda r: -r.score)
    return ranked[:k_final]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python hybrid_retrieve.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"→ Building graph-aware cache (one-time per session)...")
    cache = GraphAwareCache()
    cache.build()

    print(f"\n→ Question : {question}\n")
    results = hybrid_retrieve(question, cache)

    if not results:
        print("Aucun résultat.")
        sys.exit(0)

    print(f"Top {len(results)} chunks (hybrid scoring) :")
    print("─" * 100)
    for i, r in enumerate(results, start=1):
        types_str = ", ".join(r.types)
        text_preview = r.payload.get("text", "")[:120].replace("\n", " ")
        print(
            f"#{i} | score={r.score:.3f} (cos={r.base_cosine:.3f}) | types: [{types_str}]"
        )
        print(
            f"    {r.payload.get('filename', '?')} p{r.payload.get('page', '?')}"
            f" chunk#{r.payload.get('chunk_index', '?')}"
        )
        print(f"    {text_preview}...")
        print()
