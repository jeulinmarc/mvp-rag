"""
Shared UI helper: choose which Qdrant collection a page operates on.

The collection name used to be a single hard-coded constant (`documents`).
Each page now exposes a selector so the user can pick an existing collection
(or, on the Ingest page, create a new one). The choice is stored in
`st.session_state["active_collection"]` so it persists while navigating between
pages. Changing the collection invalidates the in-memory graph cache so the
Chat / Graph pages rebuild for the newly selected data.
"""
from __future__ import annotations

import streamlit as st

from store_chunks import COLLECTION_NAME

SESSION_KEY = "active_collection"
_NEW_COLLECTION_LABEL = "➕ Nouvelle collection…"


def list_collections() -> list[str]:
    """Return the names of all existing Qdrant collections (sorted)."""
    from streamlit_app import get_qdrant_client

    try:
        client = get_qdrant_client()
        return sorted(c.name for c in client.get_collections().collections)
    except Exception:
        return []


def get_active_collection() -> str:
    """Currently selected collection (defaults to the legacy `documents`)."""
    return st.session_state.get(SESSION_KEY, COLLECTION_NAME)


def _set_active(name: str) -> None:
    """Persist the active collection; invalidate graph cache when it changes."""
    previous = st.session_state.get(SESSION_KEY)
    if previous is not None and previous != name:
        st.session_state.graph_cache = None
    st.session_state[SESSION_KEY] = name


def collection_selector(*, allow_create: bool = False, sidebar: bool = True) -> str:
    """Render a collection picker and return the active collection name.

    Parameters
    ----------
    allow_create:
        If True, adds a "new collection" entry with a free-text field (used on
        the Ingest page so a fresh collection can be created on the fly).
    sidebar:
        Render in the sidebar (default) or inline in the page body.
    """
    container = st.sidebar if sidebar else st
    existing = list_collections()
    current = get_active_collection()

    options = list(existing)
    if allow_create:
        options.append(_NEW_COLLECTION_LABEL)

    with container:
        st.markdown("**Collection cible**")

        if not options:
            st.info("Aucune collection. Va sur la page Ingest pour en créer une.")
            _set_active(current)
            return current

        default_index = options.index(current) if current in options else 0
        choice = st.selectbox(
            "Collection cible",
            options=options,
            index=default_index,
            key="_collection_select",
            label_visibility="collapsed",
        )

        if allow_create and choice == _NEW_COLLECTION_LABEL:
            typed = st.text_input(
                "Nom de la nouvelle collection",
                value="",
                placeholder="ex: cb_corpus",
                key="_collection_new_name",
            )
            name = typed.strip() or current
        else:
            name = choice

    _set_active(name)
    return name
