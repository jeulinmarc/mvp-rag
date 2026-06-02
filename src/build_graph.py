"""
Build a similarity graph over all chunks in the Qdrant collection.
Each node = one chunk, each edge weight = cosine similarity.
We use a symmetric k-NN graph (union strategy).
"""
import numpy as np
import networkx as nx
from qdrant_client import QdrantClient

from store_chunks import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

K_NEIGHBORS = 10        # number of nearest neighbors per node
MIN_EDGE_WEIGHT = 0.0   # discard edges weaker than this (0 = keep all k-NN edges)


def fetch_all_points(collection: str = COLLECTION_NAME) -> tuple[list[dict], np.ndarray]:
    """
    Retrieve all points (vectors + payloads) from the Qdrant collection.
    Returns (payloads, vectors_matrix).
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    payloads = []
    vectors = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            payloads.append({"id": p.id, **p.payload})
            vectors.append(p.vector)
        if offset is None:
            break

    return payloads, np.array(vectors, dtype=np.float32)


def build_knn_graph(
    vectors: np.ndarray,
    payloads: list[dict],
    k: int = K_NEIGHBORS,
    min_weight: float = MIN_EDGE_WEIGHT,
) -> nx.Graph:
    """
    Build a symmetric k-NN graph from L2-normalized embeddings.
    Edge weight = cosine similarity (which equals dot product on normalized vectors).
    """
    n = vectors.shape[0]
    if n == 0:
        return nx.Graph()

    G = nx.Graph()
    for i, payload in enumerate(payloads):
        G.add_node(i, **payload)

    # Full similarity matrix: O(n²) — fine up to a few thousand chunks
    sim = vectors @ vectors.T  # shape (n, n)
    np.fill_diagonal(sim, -np.inf)  # exclude self-loops from top-k

    # Top-k neighbors per node (argsort descending)
    k_actual = min(k, n - 1)
    top_k_indices = np.argpartition(-sim, kth=k_actual - 1, axis=1)[:, :k_actual]

    for i in range(n):
        for j in top_k_indices[i]:
            weight = float(sim[i, j])
            if weight < min_weight:
                continue
            # add_edge is symmetric in nx.Graph — union strategy automatic
            if G.has_edge(i, j):
                # keep the max if asymmetry produced different "views"
                G[i][j]["weight"] = max(G[i][j]["weight"], weight)
            else:
                G.add_edge(i, j, weight=weight)

    return G


def build_graph_from_qdrant(
    collection: str = COLLECTION_NAME,
    k: int = K_NEIGHBORS,
) -> nx.Graph:
    """Convenience wrapper: fetch from Qdrant + build graph."""
    payloads, vectors = fetch_all_points(collection)
    return build_knn_graph(vectors, payloads, k=k)


def graph_stats(G: nx.Graph) -> dict:
    """Compute summary statistics for inspection."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {"nodes": 0, "edges": 0}

    degrees = [d for _, d in G.degree()]
    weights = [G[i][j]["weight"] for i, j in G.edges()]
    components = list(nx.connected_components(G))

    return {
        "nodes": n,
        "edges": m,
        "avg_degree": float(np.mean(degrees)),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "avg_edge_weight": float(np.mean(weights)) if weights else 0.0,
        "min_edge_weight": float(np.min(weights)) if weights else 0.0,
        "max_edge_weight": float(np.max(weights)) if weights else 0.0,
        "n_components": len(components),
        "largest_component_size": max(len(c) for c in components),
    }


if __name__ == "__main__":
    print("→ Récupération des points depuis Qdrant...")
    payloads, vectors = fetch_all_points()
    print(f"  {len(payloads)} chunks récupérés (dim={vectors.shape[1] if len(vectors) else 0})")

    if len(payloads) < 2:
        print("Pas assez de chunks pour construire un graphe. Ingère un PDF d'abord.")
        exit(1)

    print(f"\n→ Construction du graphe k-NN (k={K_NEIGHBORS})...")
    G = build_knn_graph(vectors, payloads, k=K_NEIGHBORS)

    stats = graph_stats(G)
    print("\nStatistiques du graphe :")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:25s} : {value:.4f}")
        else:
            print(f"  {key:25s} : {value}")

    # Show a few example edges to sanity-check
    print("\nExemple d'arêtes (échantillon trié par poids décroissant) :")
    sorted_edges = sorted(G.edges(data=True), key=lambda e: -e[2]["weight"])
    for i, (u, v, data) in enumerate(sorted_edges[:5]):
        payload_u = G.nodes[u]
        payload_v = G.nodes[v]
        print(
            f"  {data['weight']:.3f} | "
            f"{payload_u['filename']} p{payload_u['page']} ↔ "
            f"{payload_v['filename']} p{payload_v['page']}"
        )