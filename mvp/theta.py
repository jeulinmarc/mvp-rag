"""
Identification of Theta nodes — representatives of sub-clusters,
extracted via intermediate spectral modes (v_3 to v_K).

Strategy:
  1. Optionally exclude Singular and Hinge nodes (passed as `excluded` set).
  2. For each intermediate eigenvector v_k (k=3..K_THETA), pick the most
     positive and most negative projections (extremes of that sub-cluster mode).
  3. Optionally deduplicate by semantic similarity of the original embeddings.
"""
import numpy as np
import networkx as nx
from dataclasses import dataclass

from spectral import SpectralDecomposition

K_THETA = 8                     # use modes v_3 to v_{K_THETA+2}
NODES_PER_MODE = 2              # number of extremes per mode (each side)
MIN_TEXT_LENGTH = 150
DEDUP_THRESHOLD = 0.85          # cosine threshold above which we deduplicate


@dataclass
class ThetaNode:
    """A chunk identified as Theta, with the dominant mode and projection."""
    node_id: int
    dominant_mode: int          # index k of the eigenvector
    projection: float           # signed projection v_k[i]
    abs_projection: float       # |v_k[i]|, used for ranking
    payload: dict


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumes L2-normalized inputs)."""
    return float(np.dot(u, v))


def identify_theta_nodes(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    k_theta: int = K_THETA,
    nodes_per_mode: int = NODES_PER_MODE,
    min_text_length: int = MIN_TEXT_LENGTH,
    excluded: set[int] | None = None,
    dedup_threshold: float = DEDUP_THRESHOLD,
    vectors: dict[int, np.ndarray] | None = None,
) -> list[ThetaNode]:
    """
    Extract representatives of sub-clusters via intermediate spectral modes.

    Parameters:
        excluded: node ids to skip (typically Singular + Hinge).
        vectors:  dict {node_id: original 384-d embedding} for dedup.
                  If None, dedup is skipped.
    """
    excluded = excluded or set()
    node_index = decomp.node_index
    id_to_idx = {u: i for i, u in enumerate(node_index)}
    n = len(node_index)

    k_max = min(k_theta + 2, decomp.eigenvectors.shape[1])
    if k_max <= 2:
        return []

    candidates: list[ThetaNode] = []
    selected_ids: set[int] = set()

    for k in range(2, k_max):  # 0-indexed → v_3 is index 2
        v_k = decomp.eigenvectors[:, k]

        # Indices sorted by v_k ascending; take extremes from both sides
        sorted_indices = np.argsort(v_k)
        pos_extremes = sorted_indices[-nodes_per_mode:][::-1]  # most positive
        neg_extremes = sorted_indices[:nodes_per_mode]         # most negative

        for idx in list(pos_extremes) + list(neg_extremes):
            node_id = node_index[idx]
            if node_id in excluded or node_id in selected_ids:
                continue
            payload = G.nodes[node_id]
            text = payload.get("text", "")
            if len(text) < min_text_length:
                continue

            candidates.append(ThetaNode(
                node_id=node_id,
                dominant_mode=k + 1,  # 1-indexed for human readability
                projection=float(v_k[idx]),
                abs_projection=float(abs(v_k[idx])),
                payload=payload,
            ))
            selected_ids.add(node_id)

    # Sort by absolute projection magnitude (most representative first)
    candidates.sort(key=lambda t: -t.abs_projection)

    # Optional semantic dedup
    if vectors is not None and dedup_threshold < 1.0:
        deduplicated: list[ThetaNode] = []
        kept_vectors: list[np.ndarray] = []
        for theta in candidates:
            vec = vectors.get(theta.node_id)
            if vec is None:
                deduplicated.append(theta)
                continue
            is_dup = any(
                _cosine_similarity(vec, kv) > dedup_threshold
                for kv in kept_vectors
            )
            if not is_dup:
                deduplicated.append(theta)
                kept_vectors.append(vec)
        candidates = deduplicated

    return candidates


def theta_node_ids(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    excluded: set[int] | None = None,
) -> set[int]:
    """Convenience: return only the node ids of Theta nodes."""
    return {t.node_id for t in identify_theta_nodes(G, decomp, excluded=excluded)}


if __name__ == "__main__":
    from build_graph import build_graph_from_qdrant, fetch_all_points, graph_stats
    from spectral import compute_spectral_decomposition
    from singular import singular_node_ids
    from hinge import hinge_node_ids

    print("→ Reconstruction du graphe et récupération des embeddings...")
    payloads, vectors_matrix = fetch_all_points()
    G = build_graph_from_qdrant()
    stats = graph_stats(G)
    print(f"  {stats['nodes']} nœuds, {stats['edges']} arêtes")

    if stats["nodes"] < 10:
        print("Corpus trop petit pour identifier des Theta nodes.")
        exit(1)

    print("\n→ Décomposition spectrale...")
    decomp = compute_spectral_decomposition(G)

    # Build {node_id: vector} dict for dedup (node_id == index in payloads order? NO!)
    # node_id was set as Qdrant point id; we need to match payload['id']
    id_to_vector: dict[int, np.ndarray] = {}
    for payload, vec in zip(payloads, vectors_matrix):
        id_to_vector[payload["id"]] = vec

    print("\n→ Identification des Singular et Hinge (pour exclusion)...")
    singulars = singular_node_ids(G, decomp, top_fraction=0.15)
    hinges = hinge_node_ids(G, decomp, top_fraction=0.15)
    excluded = singulars | hinges
    print(f"  {len(singulars)} Singular + {len(hinges)} Hinge → {len(excluded)} nœuds exclus")

    print(f"\n→ Identification des Theta nodes (modes v_3 à v_{K_THETA + 2})...")
    thetas = identify_theta_nodes(
        G, decomp,
        excluded=excluded,
        vectors=id_to_vector,
    )
    print(f"  {len(thetas)} Theta nodes identifiés (après dédup à cos > {DEDUP_THRESHOLD})")

    print("\nTous les Theta nodes (triés par projection max) :")
    print("─" * 100)
    for i, t in enumerate(thetas, start=1):
        text_preview = t.payload.get("text", "")[:120].replace("\n", " ")
        sign = "+" if t.projection > 0 else "−"
        print(
            f"#{i:2d} | mode v_{t.dominant_mode} ({sign}) "
            f"|projection|={t.abs_projection:.3f}"
        )
        print(
            f"     {t.payload.get('filename', '?')} p{t.payload.get('page', '?')}"
            f" chunk#{t.payload.get('chunk_index', '?')}"
        )
        print(f"     {text_preview}...")
        print()
