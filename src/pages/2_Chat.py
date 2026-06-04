"""
Chat page — ask a natural-language question over the indexed corpus.
Uses the graph-aware hybrid retrieval, displays sources and types.
"""
import time
import streamlit as st

from hybrid_retrieve import GraphAwareCache, hybrid_retrieve, K_FINAL
from ask_llm import build_messages, get_client, PROVIDER, PROVIDER_CONFIG, TEMPERATURE, MAX_TOKENS
from store_chunks import COLLECTION_NAME
from collection_ui import collection_selector

st.title("💬 Chat")
st.caption("Pose une question en langage naturel. Le retrieval combine top-k cosinus et signaux graphe.")

# Collection interrogée (le changement invalide le cache graphe).
active_collection = collection_selector()


# ---------------------------------------------------------------------------
# Build / restore the graph-aware cache
# ---------------------------------------------------------------------------

def ensure_graph_cache(collection: str) -> GraphAwareCache | None:
    """Build the GraphAwareCache once per session, after each ingest invalidates."""
    if st.session_state.get("graph_cache") is not None:
        return st.session_state.graph_cache

    # Check that the collection has enough chunks for a meaningful graph
    from streamlit_app import get_qdrant_client
    client = get_qdrant_client()
    if not client.collection_exists(collection):
        st.warning("Aucune collection Qdrant trouvée. Va sur la page Ingest pour indexer un PDF.")
        return None
    info = client.get_collection(collection)
    if info.points_count < 5:
        st.warning(
            f"Seulement {info.points_count} chunks indexés — pas assez pour construire un graphe pertinent. "
            "Ingère plus de contenu."
        )
        return None

    with st.spinner("Construction du graphe sémantique et de la décomposition spectrale…"):
        cache = GraphAwareCache()
        cache.build(collection=collection)
        st.session_state.graph_cache = cache
    return cache


cache = ensure_graph_cache(active_collection)


# ---------------------------------------------------------------------------
# Sidebar — retrieval and LLM parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.markdown("**Paramètres de la requête**")
    top_k = st.slider("Top-k final", min_value=3, max_value=15, value=K_FINAL)
    temperature = st.slider("Temperature LLM", 0.0, 1.5, TEMPERATURE, 0.1)
    use_hybrid = st.toggle("Retrieval hybride (graphe)", value=True,
                           help="Désactive pour comparer à un RAG dense pur.")
    show_chunks = st.toggle("Afficher les chunks récupérés", value=True)


# ---------------------------------------------------------------------------
# Chat history initialization
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role", "content", "sources"}


def render_history():
    """Display all past messages in the conversation."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        types = f" [{', '.join(src.get('types', []))}]" if src.get("types") else ""
                        st.markdown(
                            f"- **{src['filename']}** p{src['page']} "
                            f"(score={src['score']:.3f}{types}) "
                            f"`chunk #{src.get('chunk_index', '?')}`"
                        )
                        st.caption(src["text"][:200] + "…")


render_history()


# ---------------------------------------------------------------------------
# Input — ask a question
# ---------------------------------------------------------------------------

if cache is not None:
    question = st.chat_input("Ta question…")
    if question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve
        with st.chat_message("assistant"):
            with st.status("Recherche…", expanded=show_chunks) as status:
                t0 = time.time()
                if use_hybrid:
                    results = hybrid_retrieve(question, cache, k_final=top_k, collection=active_collection)
                else:
                    # Fallback to pure dense retrieve for comparison
                    from retrieve import retrieve as dense_retrieve
                    raw = dense_retrieve(question, k=top_k, collection=active_collection)
                    # Adapt to HybridResult-ish structure for display uniformity
                    results = []
                    for r in raw:
                        from hybrid_retrieve import HybridResult
                        results.append(HybridResult(
                            node_id=-1,
                            score=r["score"],
                            base_cosine=r["score"],
                            types=["dense"],
                            payload={
                                "filename": r["filename"],
                                "page": r["page"],
                                "chunk_index": r["chunk_index"],
                                "text": r["text"],
                            },
                        ))
                t_retrieve = time.time() - t0

                if show_chunks:
                    st.markdown(f"**{len(results)} chunks récupérés en {t_retrieve*1000:.0f}ms**")
                    for i, r in enumerate(results, start=1):
                        types_str = ", ".join(r.types) if r.types else "dense"
                        st.markdown(
                            f"`#{i}` score={r.score:.3f} (cos={r.base_cosine:.3f}) "
                            f"· **{r.payload['filename']}** p{r.payload['page']} "
                            f"· [{types_str}]"
                        )

                status.update(label=f"Retrieval terminé en {t_retrieve*1000:.0f}ms", state="complete")

            # LLM streaming
            chunks_for_llm = [
                {
                    "filename": r.payload["filename"],
                    "page": r.payload["page"],
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "text": r.payload["text"],
                    "score": r.score,
                }
                for r in results
            ]
            messages = build_messages(question, chunks_for_llm)

            client = get_client()
            model = PROVIDER_CONFIG[PROVIDER]["model"]

            placeholder = st.empty()
            accumulated = ""
            t0 = time.time()
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        accumulated += delta
                        placeholder.markdown(accumulated + "▌")
                placeholder.markdown(accumulated)
            except Exception as e:
                placeholder.error(f"Erreur LLM : {e}")
                accumulated = f"[erreur LLM : {e}]"
            t_llm = time.time() - t0

            # Footer
            st.caption(
                f"Provider : {PROVIDER} ({model}) · "
                f"retrieve {t_retrieve*1000:.0f}ms · LLM {t_llm:.1f}s"
            )

            # Add to history
            sources = [
                {
                    "filename": r.payload["filename"],
                    "page": r.payload["page"],
                    "chunk_index": r.payload.get("chunk_index", "?"),
                    "score": r.score,
                    "types": r.types,
                    "text": r.payload["text"],
                }
                for r in results
            ]
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": accumulated,
                "sources": sources,
            })


# ---------------------------------------------------------------------------
# Sidebar — clear history
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
