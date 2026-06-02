"""
Identification of Hinge nodes — pivot chunks that bridge multiple clusters.
Combines three criteria:
  1. Betweenness centrality (canonical measure of bridging)
  2. Proximity of v_2 to zero (Fiedler-frontier nodes)
  3. Spectral diversity of neighbors (multi-cluster membership)
Articulation points get a max score automatically.
"""
import numpy as np
import networkx as nx
from dataclasses import dataclass

from spectral import SpectralDecomposition

K_SPECTRAL_EMBED = 6        # number of eigenvectors for neighbor diversity
MIN_TEXT_LENGTH = 150       # filter out too-short chunks

WEIGHT_BETWEENNESS = 0.5
WEIGHT_FIEDLER = 0.3
WEIGHT_DIVERSITY = 0.2


@dataclass
class HingeNode:
    """A chunk identified as Hinge, with sub-scores for inspection."""
    node_id: int
    score: float
    score_betweenness: float
    score_fiedler: float
    score_diversity: float
    is_articulation: bool
    payload: dict


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max scaling to [0, 1]. If constant, return zeros."""
    span = x.max() - x.min()
    if span < 1e-12:
        return np.zeros_like(x)
    return (x - x.min()) / span


def compute_hinge_scores(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    k_spectral: int = K_SPECTRAL_EMBED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, set]:
    """
    Compute composite hinge score per node, aligned with decomp.node_index.
    Returns (composite, s_between, s_fiedler, s_diversity, articulation_set).
    """
    node_index = decomp.node_index
    n = len(node_index)

    # --- Criterion 1: betweenness centrality ---
    # Returns dict {node_id: betweenness}
    btw_dict = nx.betweenness_centrality(G, weight="weight", normalized=True)
    betweenness = np.array([btw_dict[u] for u in node_index])
    score_between = _normalize(betweenness)

    # --- Criterion 2: Fiedler frontier (small |v_2|) ---
    fiedler = decomp.fiedler_vector()
    score_fiedler = _normalize(-np.abs(fiedler))  # negate so small |v_2| → high score

    # --- Criterion 3: spectral diversity of neighbors ---
    k_actual = min(k_spectral, decomp.eigenvectors.shape[1] - 1)
    spectral_embed = decomp.eigenvectors[:, 1:1 + k_actual]  # skip v_1
    id_to_idx = {u: i for i, u in enumerate(node_index)}

    diversity = np.zeros(n)
    for idx, u in enumerate(node_index):
        neighbors = list(G.neighbors(u))
        if len(neighbors) < 2:
            diversity[idx] = 0.0
            continue
        neighbor_embeds = np.array([
            spectral_embed[id_to_idx[v]] for v in neighbors
        ])
        # Mean std across dimensions = spread of neighbors in spectral space
        diversity[idx] = float(np.mean(np.std(neighbor_embeds, axis=0)))
    score_diversity = _normalize(diversity)

    # --- Articulation points (bypass for any node) ---
    articulation = set(nx.articulation_points(G))

    # --- Composite ---
    composite = (
        WEIGHT_BETWEENNESS * score_between
        + WEIGHT_FIEDLER * score_fiedler
        + WEIGHT_DIVERSITY * score_diversity
    )
    # Force max score for articulation points
    for u in articulation:
        composite[id_to_idx[u]] = 1.0

    return composite, score_between, score_fiedler, score_diversity, articulation


def identify_hinge_nodes(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    top_fraction: float = 0.15,
    min_text_length: int = MIN_TEXT_LENGTH,
    k_spectral: int = K_SPECTRAL_EMBED,
) -> list[HingeNode]:
    """
    Return the top `top_fraction` of nodes ranked by hinge score,
    excluding chunks that are too short.
    """
    composite, s_btw, s_fie, s_div, articulation = compute_hinge_scores(
        G, decomp, k_spectral
    )
    id_to_idx = {u: i for i, u in enumerate(decomp.node_index)}

    candidates = []
    for idx, node_id in enumerate(decomp.node_index):
        payload = G.nodes[node_id]
        text = payload.get("text", "")
        if len(text) < min_text_length:
            continue
        candidates.append(HingeNode(
            node_id=node_id,
            score=float(composite[idx]),
            score_betweenness=float(s_btw[idx]),
            score_fiedler=float(s_fie[idx]),
            score_diversity=float(s_div[idx]),
            is_articulation=(node_id in articulation),
            payload=payload,
        ))

    candidates.sort(key=lambda h: -h.score)
    n_top = max(1, int(top_fraction * len(candidates)))
    return candidates[:n_top]


def hinge_node_ids(
    G: nx.Graph,
    decomp: SpectralDecomposition,
    top_fraction: float = 0.15,
) -> set[int]:
    """Convenience: return only the node ids of Hinge nodes."""
    return {h.node_id for h in identify_hinge_nodes(G, decomp, top_fraction)}


if __name__ == "__main__":
    from build_graph import build_graph_from_qdrant, graph_stats
    from spectral import compute_spectral_decomposition

    print("→ Reconstruction du graphe...")
    G = build_graph_from_qdrant()
    stats = graph_stats(G)
    print(f"  {stats['nodes']} nœuds, {stats['edges']} arêtes")

    if stats["nodes"] < 5:
        print("Corpus trop petit pour identifier des Hinge nodes.")
        exit(1)

    print("\n→ Décomposition spectrale...")
    decomp = compute_spectral_decomposition(G)

    print("\n→ Identification des Hinge nodes (top 15%)...")
    hinges = identify_hinge_nodes(G, decomp, top_fraction=0.15)
    n_articulation = sum(1 for h in hinges if h.is_articulation)
    print(f"  {len(hinges)} Hinge nodes identifiés ({n_articulation} articulation points)")

    print("\nTop 10 nœuds les plus Hinge :")
    print("─" * 100)
    for i, h in enumerate(hinges[:10], start=1):
        text_preview = h.payload.get("text", "")[:120].replace("\n", " ")
        marker = " [ARTICULATION]" if h.is_articulation else ""
        print(
            f"#{i:2d} | score={h.score:.3f} "
            f"(btw={h.score_betweenness:.2f} fie={h.score_fiedler:.2f} div={h.score_diversity:.2f}){marker}"
        )
        print(
            f"     {h.payload.get('filename', '?')} p{h.payload.get('page', '?')}"
            f" chunk#{h.payload.get('chunk_index', '?')}"
        )
        print(f"     {text_preview}...")
        print()

    # Comparison: least-Hinge nodes (core of clusters)
    composite, _, _, _, _ = compute_hinge_scores(G, decomp)
    least_hinge_idx = np.argsort(composite)[:3]
    print("─" * 100)
    print("Pour comparaison, les 3 chunks les MOINS Hinge (au cœur de leur cluster) :")
    for idx in least_hinge_idx:
        node_id = decomp.node_index[idx]
        payload = G.nodes[node_id]
        text_preview = payload.get("text", "")[:120].replace("\n", " ")
        print(f"     score={composite[idx]:.3f} | p{payload.get('page', '?')} | {text_preview}...")
