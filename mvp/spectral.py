"""
Spectral decomposition of the similarity graph.
Computes the normalized Laplacian and its eigendecomposition.
"""
import numpy as np
import networkx as nx
from dataclasses import dataclass


@dataclass
class SpectralDecomposition:
    """Container for the spectral analysis results."""
    eigenvalues: np.ndarray       # shape (n,), sorted ascending
    eigenvectors: np.ndarray      # shape (n, n), eigenvectors[:, i] is the i-th
    node_index: list              # mapping graph node id → row in eigenvectors
    is_connected: bool
    algebraic_connectivity: float  # λ_2

    def fiedler_vector(self) -> np.ndarray:
        """Return v_2, the Fiedler vector (best bipartite cut)."""
        return self.eigenvectors[:, 1]

    def spectral_embedding(self, k: int = 10) -> np.ndarray:
        """
        Return the spectral embedding: each node mapped to its (v_2, ..., v_k) coords.
        We skip v_1 (constant, no info).
        Shape: (n_nodes, k-1).
        """
        return self.eigenvectors[:, 1:k]


def compute_spectral_decomposition(
    G: nx.Graph,
    normalized: bool = True,
) -> SpectralDecomposition:
    """
    Compute the eigendecomposition of the graph Laplacian.
    Uses np.linalg.eigh (exploits symmetry, O(n³) complexity).
    """
    n = G.number_of_nodes()
    if n < 2:
        raise ValueError(f"Cannot decompose a graph with {n} nodes.")

    # Ensure consistent node ordering — sort by node id
    node_index = sorted(G.nodes())

    # Build the Laplacian (sparse → dense numpy array)
    if normalized:
        L_sparse = nx.normalized_laplacian_matrix(G, nodelist=node_index)
    else:
        L_sparse = nx.laplacian_matrix(G, nodelist=node_index)

    L = L_sparse.toarray().astype(np.float64)
    # Ensure perfect symmetry (numerical noise from sparse representation)
    L = (L + L.T) / 2.0

    # Eigendecomposition — eigh returns sorted ascending
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Conventional sign normalization: force first nonzero component positive
    for i in range(eigenvectors.shape[1]):
        first_nonzero = np.argmax(np.abs(eigenvectors[:, i]) > 1e-9)
        if eigenvectors[first_nonzero, i] < 0:
            eigenvectors[:, i] *= -1

    is_connected = nx.is_connected(G)
    algebraic_connectivity = float(eigenvalues[1]) if n >= 2 else 0.0

    return SpectralDecomposition(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        node_index=node_index,
        is_connected=is_connected,
        algebraic_connectivity=algebraic_connectivity,
    )


def fiedler_bipartition(decomp: SpectralDecomposition) -> tuple[list, list]:
    """
    Use the Fiedler vector to split nodes into two groups (positive vs negative).
    Returns (cluster_pos, cluster_neg) as lists of node ids.
    """
    fiedler = decomp.fiedler_vector()
    cluster_pos = [decomp.node_index[i] for i, v in enumerate(fiedler) if v > 0]
    cluster_neg = [decomp.node_index[i] for i, v in enumerate(fiedler) if v <= 0]
    return cluster_pos, cluster_neg


def spectral_summary(decomp: SpectralDecomposition, top_k_eigenvalues: int = 10) -> dict:
    """Produce a human-readable summary of the decomposition."""
    n = len(decomp.eigenvalues)
    return {
        "n_nodes": n,
        "is_connected": decomp.is_connected,
        "lambda_1 (≈ 0)": float(decomp.eigenvalues[0]),
        "lambda_2 (Fiedler / algebraic connectivity)": decomp.algebraic_connectivity,
        f"first_{top_k_eigenvalues}_eigenvalues": [
            float(x) for x in decomp.eigenvalues[:top_k_eigenvalues]
        ],
        "spectral_gap (λ_2 / λ_n)": float(
            decomp.eigenvalues[1] / decomp.eigenvalues[-1]
        ) if decomp.eigenvalues[-1] > 1e-9 else 0.0,
    }


if __name__ == "__main__":
    from build_graph import build_graph_from_qdrant, graph_stats

    print("→ Reconstruction du graphe depuis Qdrant...")
    G = build_graph_from_qdrant()
    stats = graph_stats(G)
    print(f"  {stats['nodes']} nœuds, {stats['edges']} arêtes, {stats['n_components']} composante(s)")

    if stats["nodes"] < 2:
        print("Pas assez de nœuds pour l'analyse spectrale.")
        exit(1)

    print("\n→ Décomposition spectrale (Laplacien normalisé)...")
    decomp = compute_spectral_decomposition(G, normalized=True)

    print("\nRésumé spectral :")
    summary = spectral_summary(decomp, top_k_eigenvalues=10)
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for i, v in enumerate(value):
                print(f"      λ_{i+1} = {v:.4f}")
        elif isinstance(value, float):
            print(f"  {key:50s}: {value:.4f}")
        else:
            print(f"  {key:50s}: {value}")

    print("\n→ Bipartition de Fiedler (coupe binaire naturelle) :")
    cluster_pos, cluster_neg = fiedler_bipartition(decomp)
    print(f"  Cluster A (v_2 > 0) : {len(cluster_pos)} nœuds")
    print(f"  Cluster B (v_2 ≤ 0) : {len(cluster_neg)} nœuds")

    # Aperçu textuel des 3 chunks les plus centraux de chaque cluster
    fiedler = decomp.fiedler_vector()
    print("\n  Chunks les plus 'caractéristiques' du Cluster A (v_2 le plus positif) :")
    top_pos = np.argsort(-fiedler)[:3]
    for idx in top_pos:
        node_id = decomp.node_index[idx]
        payload = G.nodes[node_id]
        preview = payload.get("text", "")[:80].replace("\n", " ")
        print(f"    v_2={fiedler[idx]:+.4f} | p{payload.get('page', '?')} | {preview}...")

    print("\n  Chunks les plus 'caractéristiques' du Cluster B (v_2 le plus négatif) :")
    top_neg = np.argsort(fiedler)[:3]
    for idx in top_neg:
        node_id = decomp.node_index[idx]
        payload = G.nodes[node_id]
        preview = payload.get("text", "")[:80].replace("\n", " ")
        print(f"    v_2={fiedler[idx]:+.4f} | p{payload.get('page', '?')} | {preview}...")