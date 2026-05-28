"""
Identification of Singular nodes — atypical, non-redundant chunks.
Combines three criteria:
  1. Low weighted degree (low overall similarity to the rest)
  2. Low mean similarity to nearest neighbors
  3. High projection on high-frequency spectral modes
"""
import numpy as np
import networkx as nx
from dataclasses import dataclass

from spectral import SpectralDecomposition

K_HIGH_FREQ = 10           # number of highest-frequency eigenvectors to use
MIN_TEXT_LENGTH = 100      # filter out very short chunks (likely noise)

WEIGHT_SPECTRAL = 0.5
WEIGHT_DEGREE = 0.3
WEIGHT_NEIGHBOR_SIM = 0.2


@dataclass
class SingularNode:
    """A chunk identified as Singular, with its scores for inspection."""
    node_id: int
    score: float                # composite score (higher = more Singular)
    score_spectral: float
    score_low_degree: float
    score_low_neighbor_sim: float
    payload: dict


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max scaling to [0, 1]. If constant, return zeros."""
    span = x.max() - x.min()
    if span < 1e-12:
        return np.zeros_like(x)
    return (x - x.min()) / span


def compute_singular_scores(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    k_high_freq: int = K_HIGH_FREQ,
) -> np.ndarray:
    """
    Compute a composite singularity score per node.
    Returns an array of shape (n,) aligned with decomp.node_index.
    """
    n = len(decomp.node_index)

    # --- Criterion 1: low weighted degree ---
    degrees = np.array([
        sum(G[u][v]["weight"] for v in G.neighbors(u))
        for u in decomp.node_index
    ])
    # invert so that LOW degree → HIGH score
    score_low_degree = _normalize(-degrees)

    # --- Criterion 2: low mean similarity to neighbors ---
    neighbor_sims = []
    for u in decomp.node_index:
        neighbors = list(G.neighbors(u))
        if neighbors:
            mean_sim = np.mean([G[u][v]["weight"] for v in neighbors])
        else:
            mean_sim = 0.0
        neighbor_sims.append(mean_sim)
    neighbor_sims = np.array(neighbor_sims)
    score_low_neighbor_sim = _normalize(-neighbor_sims)

    # --- Criterion 3: high-frequency spectral projection ---
    # Last k_high_freq eigenvectors (highest eigenvalues)
    k_actual = min(k_high_freq, decomp.eigenvectors.shape[1])
    high_freq_vectors = decomp.eigenvectors[:, -k_actual:]  # shape (n, k)
    # Sum of squared projections per node
    spectral_atypism = np.sum(high_freq_vectors ** 2, axis=1)
    score_spectral = _normalize(spectral_atypism)

    # --- Composite score ---
    composite = (
        WEIGHT_SPECTRAL * score_spectral
        + WEIGHT_DEGREE * score_low_degree
        + WEIGHT_NEIGHBOR_SIM * score_low_neighbor_sim
    )

    return composite, score_spectral, score_low_degree, score_low_neighbor_sim


def identify_singular_nodes(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    top_fraction: float = 0.15,
    min_text_length: int = MIN_TEXT_LENGTH,
    k_high_freq: int = K_HIGH_FREQ,
) -> list[SingularNode]:
    """
    Return the top `top_fraction` of nodes ranked by singularity score,
    excluding chunks that are too short (likely noise).
    """
    composite, s_spec, s_deg, s_nei = compute_singular_scores(G, decomp, k_high_freq)
    n = len(composite)

    candidates = []
    for idx, node_id in enumerate(decomp.node_index):
        payload = G.nodes[node_id]
        text = payload.get("text", "")
        if len(text) < min_text_length:
            continue
        candidates.append(SingularNode(
            node_id=node_id,
            score=float(composite[idx]),
            score_spectral=float(s_spec[idx]),
            score_low_degree=float(s_deg[idx]),
            score_low_neighbor_sim=float(s_nei[idx]),
            payload=payload,
        ))

    candidates.sort(key=lambda s: -s.score)
    n_top = max(1, int(top_fraction * len(candidates)))
    return candidates[:n_top]


def singular_node_ids(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    top_fraction: float = 0.15,
) -> set[int]:
    """Convenience: return only the node ids of Singular nodes (used by retrieval)."""
    return {s.node_id for s in identify_singular_nodes(G, decomp, top_fraction)}


if __name__ == "__main__":
    from build_graph import build_graph_from_qdrant, graph_stats
    from spectral import compute_spectral_decomposition

    print("→ Reconstruction du graphe...")
    G = build_graph_from_qdrant()
    stats = graph_stats(G)
    print(f"  {stats['nodes']} nœuds, {stats['edges']} arêtes")

    if stats["nodes"] < 5:
        print("Corpus trop petit pour identifier des Singular nodes.")
        exit(1)

    print("\n→ Décomposition spectrale...")
    decomp = compute_spectral_decomposition(G)
    print(f"  λ_2 = {decomp.algebraic_connectivity:.4f}")

    print("\n→ Identification des Singular nodes (top 15%)...")
    singulars = identify_singular_nodes(G, decomp, top_fraction=0.15)
    print(f"  {len(singulars)} Singular nodes identifiés")

    print("\nTop 10 nœuds les plus Singular :")
    print("─" * 100)
    for i, s in enumerate(singulars[:10], start=1):
        text_preview = s.payload.get("text", "")[:120].replace("\n", " ")
        print(
            f"#{i:2d} | score={s.score:.3f} "
            f"(spec={s.score_spectral:.2f} deg={s.score_low_degree:.2f} nei={s.score_low_neighbor_sim:.2f})"
        )
        print(
            f"     {s.payload.get('filename', '?')} p{s.payload.get('page', '?')}"
            f" chunk#{s.payload.get('chunk_index', '?')}"
        )
        print(f"     {text_preview}...")
        print()

    # Sanity check: compare with the LEAST Singular (most central)
    composite, _, _, _ = compute_singular_scores(G, decomp)
    least_singular_idx = np.argsort(composite)[:3]
    print("─" * 100)
    print("Pour comparaison, les 3 chunks les MOINS Singular (chunks 'mainstream') :")
    for idx in least_singular_idx:
        node_id = decomp.node_index[idx]
        payload = G.nodes[node_id]
        text_preview = payload.get("text", "")[:120].replace("\n", " ")
        print(f"     score={composite[idx]:.3f} | p{payload.get('page', '?')} | {text_preview}...")