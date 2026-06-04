"""
Graph Explorer page — interactive 2D visualization of the semantic graph.
Colors nodes by type (Singular / Hinge / Theta / mainstream).
Edge transparency proportional to similarity.
"""
import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from hybrid_retrieve import GraphAwareCache
from store_chunks import COLLECTION_NAME
from collection_ui import collection_selector

st.title("🕸️ Graph Explorer")
st.caption(
    "Visualisation du graphe sémantique. Les nœuds colorés sont les "
    "Singular (orange), Hinge (rouge) et Theta (bleu). Survole un nœud pour voir son contenu."
)

# Collection visualisée (le changement invalide le cache graphe).
active_collection = collection_selector()

# Palette
COLOR_MAINSTREAM = "#9CA3AF"   # gray-400
COLOR_SINGULAR = "#F59E0B"     # amber-500
COLOR_HINGE = "#DC2626"        # red-600
COLOR_THETA = "#2563EB"        # blue-600

# Priority for color when a node is multi-tagged (rare due to exclusion cascade)
TYPE_PRIORITY = ["hinge", "singular", "theta", "mainstream"]
TYPE_COLOR = {
    "mainstream": COLOR_MAINSTREAM,
    "singular": COLOR_SINGULAR,
    "hinge": COLOR_HINGE,
    "theta": COLOR_THETA,
}


# ---------------------------------------------------------------------------
# Ensure cache and graph are available
# ---------------------------------------------------------------------------

def ensure_graph_cache(collection: str) -> GraphAwareCache | None:
    """Build cache once per session, after each ingest invalidates it."""
    if st.session_state.get("graph_cache") is not None:
        return st.session_state.graph_cache

    from streamlit_app import get_qdrant_client
    client = get_qdrant_client()
    if not client.collection_exists(collection):
        st.warning("Aucune collection Qdrant trouvée. Va sur la page Ingest pour indexer un PDF.")
        return None
    info = client.get_collection(collection)
    if info.points_count < 5:
        st.warning(f"Seulement {info.points_count} chunks indexés. Trop peu pour un graphe pertinent.")
        return None

    with st.spinner("Construction du graphe sémantique…"):
        cache = GraphAwareCache()
        cache.build(collection=collection)
        st.session_state.graph_cache = cache
    return cache


cache = ensure_graph_cache(active_collection)
if cache is None:
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar params
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.markdown("**Visualisation**")
    layout_choice = st.selectbox(
        "Layout",
        options=["Kamada-Kawai", "Spring (Fruchterman-Reingold)", "Spectral"],
        index=0,
        help="Algorithme de placement des nœuds.",
    )
    min_edge_weight = st.slider(
        "Seuil min. d'arête",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="N'affiche que les arêtes au-dessus de ce poids.",
    )
    show_labels = st.toggle("Afficher labels (chunk #)", value=False)
    node_size = st.slider("Taille des nœuds", 5, 30, 12)


# ---------------------------------------------------------------------------
# Recompute graph from Qdrant (we need it again — cache only stored special-node vectors)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_graph_and_layout(_cache_signature: int, layout_name: str, collection: str):
    """
    Rebuild the graph from Qdrant + compute the layout.
    `_cache_signature` is bumped whenever ingestion changes the corpus —
    here we pass len(cache.vectors) as a proxy proxy for "corpus changed".
    """
    from build_graph import build_graph_from_qdrant
    G = build_graph_from_qdrant(collection=collection)

    if layout_name == "Kamada-Kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_name == "Spring (Fruchterman-Reingold)":
        pos = nx.spring_layout(G, iterations=80, seed=42)
    else:  # Spectral
        pos = nx.spectral_layout(G)
    return G, pos


# Use len(special vectors cached) as a rough signature
sig = len(cache.vectors) + sum(1 for _ in cache.singular_ids) * 100
G, pos = get_graph_and_layout(sig, layout_choice, active_collection)


# ---------------------------------------------------------------------------
# Build Plotly traces
# ---------------------------------------------------------------------------

def node_type_for(node_id: int) -> str:
    """Pick a single type per node according to priority."""
    types = []
    if node_id in cache.hinge_ids:
        types.append("hinge")
    if node_id in cache.singular_ids:
        types.append("singular")
    if node_id in cache.theta_ids:
        types.append("theta")
    if not types:
        return "mainstream"
    for t in TYPE_PRIORITY:
        if t in types:
            return t
    return "mainstream"


# Edges
edge_x, edge_y, edge_widths, edge_alphas = [], [], [], []
for u, v, data in G.edges(data=True):
    w = data.get("weight", 0.0)
    if w < min_edge_weight:
        continue
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color="rgba(120,120,120,0.35)"),
    hoverinfo="none",
    mode="lines",
    showlegend=False,
)


# Nodes by type for legend
node_traces = []
for type_name in ["mainstream", "theta", "singular", "hinge"]:
    xs, ys, hovers, labels = [], [], [], []
    for node_id in G.nodes():
        if node_type_for(node_id) != type_name:
            continue
        x, y = pos[node_id]
        xs.append(x)
        ys.append(y)
        payload = G.nodes[node_id]
        text_preview = (payload.get("text", "") or "")[:200].replace("\n", " ")
        hover = (
            f"<b>{payload.get('filename', '?')}</b> p{payload.get('page', '?')}<br>"
            f"chunk #{payload.get('chunk_index', '?')}<br>"
            f"type: <b>{type_name}</b><br>"
            f"degree: {G.degree(node_id)}<br>"
            f"<i>{text_preview}…</i>"
        )
        hovers.append(hover)
        labels.append(str(payload.get("chunk_index", "")))

    if not xs:
        continue
    node_traces.append(go.Scatter(
        x=xs, y=ys,
        mode="markers+text" if show_labels else "markers",
        text=labels if show_labels else None,
        textfont=dict(size=8, color="#444"),
        textposition="top center",
        hoverinfo="text",
        hovertext=hovers,
        marker=dict(
            size=node_size,
            color=TYPE_COLOR[type_name],
            line=dict(width=1, color="white"),
        ),
        name=type_name.capitalize() + f" ({len(xs)})",
        showlegend=True,
    ))


fig = go.Figure(data=[edge_trace] + node_traces)
fig.update_layout(
    height=750,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Stats below the graph
# ---------------------------------------------------------------------------

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nœuds total", G.number_of_nodes())
c2.metric("Arêtes (affichées)", sum(1 for _, _, d in G.edges(data=True) if d["weight"] >= min_edge_weight))
c3.metric("Singular", len(cache.singular_ids), help="Chunks atypiques / originaux")
c4.metric("Hinge", len(cache.hinge_ids), help="Chunks pivots inter-clusters")
c5.metric("Theta", len(cache.theta_ids), help="Représentants de sous-clusters")


with st.expander("ℹ️ Lecture du graphe"):
    st.markdown("""
- **Clusters visibles** : groupes denses de nœuds → thématiques cohérentes du document.
- **Hinge (rouge)** placés entre deux clusters → passerelles narratives.
- **Singular (orange)** isolés en marge → contenus originaux non redondants (remerciements, références, encadrés…).
- **Theta (bleu)** au cœur de plusieurs zones → représentants de sous-thèmes.
- **Arêtes plus visibles** = chunks plus similaires sémantiquement.
- **Boule indistincte** = corpus trop homogène ou k trop élevé.
- **Composantes séparées** = sujets sans lien dans le corpus, ou k trop bas.
    """)
