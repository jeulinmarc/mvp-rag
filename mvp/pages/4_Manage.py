"""
Manage page — list indexed files, delete by filename, or purge the whole collection.
Always invalidates the graph cache after any modification.
"""
import streamlit as st
import pandas as pd
from qdrant_client.models import Filter, FieldCondition, MatchValue

from store_chunks import COLLECTION_NAME

st.title("🗂️ Manage")
st.caption("Liste les documents indexés, supprime un fichier spécifique, ou vide toute la collection.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client():
    from streamlit_app import get_qdrant_client
    return get_qdrant_client()


def list_indexed_files() -> dict:
    """
    Scroll the collection and aggregate by filename.
    Returns: {filename: {"chunks": int, "pages": set[int]}}
    """
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {}

    files = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            fn = p.payload.get("filename", "unknown")
            if fn not in files:
                files[fn] = {"chunks": 0, "pages": set()}
            files[fn]["chunks"] += 1
            page = p.payload.get("page")
            if page is not None:
                files[fn]["pages"].add(page)
        if offset is None:
            break
    return files


def delete_file(filename: str) -> int:
    """Delete all chunks with the given filename. Returns approximate count deleted."""
    client = get_client()
    # Count first (approximate)
    files = list_indexed_files()
    n = files.get(filename, {}).get("chunks", 0)
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[
            FieldCondition(key="filename", match=MatchValue(value=filename))
        ]),
    )
    return n


def purge_collection() -> None:
    """Drop the entire collection. It will be recreated on next ingestion."""
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)


def invalidate_caches():
    """Clear graph cache and last ingest info."""
    st.session_state.graph_cache = None
    st.session_state.pop("last_ingest", None)


# ---------------------------------------------------------------------------
# List of indexed files
# ---------------------------------------------------------------------------

files = list_indexed_files()

if not files:
    st.info("Aucun fichier indexé pour l'instant. Va sur la page Ingest pour démarrer.")
    st.page_link("pages/1_Ingest.py", label="→ Ingest", icon="📥")
    st.stop()


# Build a DataFrame for display
rows = [
    {
        "Fichier": fn,
        "Chunks": data["chunks"],
        "Pages couvertes": len(data["pages"]),
        "Plage de pages": (
            f"{min(data['pages'])} – {max(data['pages'])}"
            if data["pages"] else "—"
        ),
    }
    for fn, data in sorted(files.items())
]
df = pd.DataFrame(rows)

st.subheader(f"Documents indexés ({len(files)})")
st.dataframe(df, use_container_width=True, hide_index=True)

total_chunks = sum(d["chunks"] for d in files.values())
st.metric("Total chunks dans Qdrant", total_chunks)


# ---------------------------------------------------------------------------
# Selective delete
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Supprimer un fichier")

selected = st.selectbox(
    "Fichier à supprimer",
    options=sorted(files.keys()),
    index=0,
    help="Tous les chunks de ce fichier seront supprimés.",
)

if st.button("🗑️ Supprimer ce fichier", type="primary"):
    st.session_state.pending_delete = selected

if st.session_state.get("pending_delete"):
    target = st.session_state.pending_delete
    st.warning(f"⚠️ Confirmer la suppression de **{target}** ? Cette action est irréversible.")
    c1, c2 = st.columns(2)
    if c1.button("✅ Oui, supprimer définitivement", type="primary", use_container_width=True):
        n = delete_file(target)
        invalidate_caches()
        st.session_state.pending_delete = None
        st.success(f"Supprimé : {n} chunks de **{target}**")
        st.rerun()
    if c2.button("❌ Annuler", use_container_width=True):
        st.session_state.pending_delete = None
        st.rerun()


# ---------------------------------------------------------------------------
# Purge everything
# ---------------------------------------------------------------------------

st.divider()
with st.expander("☠️ Zone dangereuse — purge complète"):
    st.markdown(
        "Supprime **toute la collection Qdrant**. Tous les fichiers indexés disparaissent. "
        "Le graphe spectral et tous les caches sont également invalidés."
    )

    if st.button("Tout effacer", type="secondary"):
        st.session_state.pending_purge = True

    if st.session_state.get("pending_purge"):
        st.error(
            "⚠️ **Cette action est irréversible.** "
            f"Tu vas perdre {total_chunks} chunks de {len(files)} fichier(s)."
        )
        c1, c2 = st.columns(2)
        if c1.button("✅ Oui, tout supprimer", type="primary", use_container_width=True):
            purge_collection()
            invalidate_caches()
            st.session_state.pending_purge = False
            st.session_state.chat_history = []
            st.success("Collection vidée. Tu peux maintenant repartir de zéro.")
            st.rerun()
        if c2.button("❌ Non, annuler", use_container_width=True):
            st.session_state.pending_purge = False
            st.rerun()
