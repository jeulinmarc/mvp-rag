"""
Eigenmind MVP — Streamlit multipage app.
Entry point with home, setup checks, and shared session state init.

Run with:  streamlit run streamlit_app.py
"""
import streamlit as st
import requests
from qdrant_client import QdrantClient

from store_chunks import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

st.set_page_config(
    page_title="Eigenmind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Shared cached resources (one instance for the whole server)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    """Persistent Qdrant client, shared across all pages and sessions."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_qdrant() -> tuple[bool, str]:
    try:
        client = get_qdrant_client()
        info = client.get_collections()
        return True, f"{len(info.collections)} collection(s)"
    except Exception as e:
        return False, str(e)


def check_ollama() -> tuple[bool, str]:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        models = r.json().get("models", [])
        return True, f"{len(models)} modèle(s)"
    except Exception as e:
        return False, str(e)


def collection_stats() -> dict:
    try:
        client = get_qdrant_client()
        if not client.collection_exists(COLLECTION_NAME):
            return {"exists": False, "count": 0}
        info = client.get_collection(COLLECTION_NAME)
        return {"exists": True, "count": info.points_count}
    except Exception:
        return {"exists": False, "count": 0}


# ---------------------------------------------------------------------------
# Sidebar — global session info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 Eigenmind")
    st.caption("RAG augmenté par analyse spectrale")
    st.divider()

    st.markdown("**État des services**")
    qdrant_ok, qdrant_info = check_qdrant()
    ollama_ok, ollama_info = check_ollama()

    if qdrant_ok:
        st.success(f"Qdrant · {qdrant_info}", icon="✅")
    else:
        st.error(f"Qdrant DOWN · {qdrant_info[:50]}", icon="❌")

    if ollama_ok:
        st.success(f"Ollama · {ollama_info}", icon="✅")
    else:
        st.error(f"Ollama DOWN · {ollama_info[:50]}", icon="❌")

    st.divider()

    stats = collection_stats()
    if stats["exists"]:
        st.metric("Chunks indexés", stats["count"])
    else:
        st.metric("Chunks indexés", 0)
        st.caption("Collection vide ou non créée.")


# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------

st.title("🧠 Eigenmind")
st.markdown(
    "Reimplémentation d'un système de **RAG augmenté par analyse spectrale de graphes**. "
    "Utilise le menu de gauche pour naviguer entre les pages."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Ingestion")
    st.markdown(
        "Uploade un PDF et il est découpé, embeddé, et stocké dans Qdrant. "
        "Le graphe spectral et les Singular/Hinge/Theta nodes seront recalculés "
        "automatiquement à la première requête sur la page Chat."
    )
    st.page_link("pages/1_Ingest.py", label="Aller à Ingest", icon="📥")

with col2:
    st.markdown("### 💬 Chat")
    st.markdown(
        "Pose une question en langage naturel sur le corpus indexé. "
        "Le retrieval hybride (dense + signaux graphe) sélectionne les chunks "
        "les plus pertinents, et un LLM local génère une réponse citée."
    )
    st.page_link("pages/2_Chat.py", label="Aller à Chat", icon="💬")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🕸️ Graph Explorer")
    st.markdown(
        "Visualise le graphe sémantique du corpus. Les Singular nodes (atypiques), "
        "Hinge nodes (pivots) et Theta nodes (sous-clusters) sont colorés différemment "
        "pour révéler la structure narrative du document."
    )
    st.page_link("pages/3_Graph_Explorer.py", label="Aller à Graph Explorer", icon="🕸️")

with col4:
    st.markdown("### 🗂️ Manage")
    st.markdown(
        "Liste les documents ingérés, leur date d'ajout et leur nombre de chunks. "
        "Permet la suppression sélective par fichier ou la purge complète de la collection."
    )
    st.page_link("pages/4_Manage.py", label="Aller à Manage", icon="🗂️")

col5, _ = st.columns(2)

with col5:
    st.markdown("### 📚 Theory")
    st.markdown(
        "Lis la théorie de chaque étape à côté du code qui l'implémente. "
        "Sélectionne un chapitre et le `.md` de `theory/` s'affiche en parallèle "
        "du fichier Python correspondant."
    )
    st.page_link("pages/5_Theory.py", label="Aller à Theory", icon="📚")

st.divider()

with st.expander("ℹ️ Comment ça marche sous le capot"):
    st.markdown("""
**Phase d'ingestion** (offline, sur upload d'un PDF) :
1. Le PDF est extrait page par page (`pypdf`).
2. Découpé en chunks de ~500 caractères avec overlap (`RecursiveCharacterTextSplitter`).
3. Chaque chunk est embeddé en 384-d (`sentence-transformers all-MiniLM-L6-v2`).
4. Stocké dans Qdrant avec un ID déterministe (hash SHA-1).

**Phase d'analyse** (offline, à la première requête après changement) :
5. Construction du graphe k-NN sur les embeddings (`NetworkX`).
6. Calcul du Laplacien normalisé et de sa décomposition propre (`numpy.linalg.eigh`).
7. Extraction des Singular nodes (modes haute fréquence), Hinge nodes (betweenness + frontière Fiedler), Theta nodes (modes intermédiaires).

**Phase de requête** (online, à chaque question) :
8. Embed de la question, dense top-K via Qdrant.
9. Ajout des Singular/Hinge/Theta pertinents (cos > seuil) avec un boost de score.
10. Le top-k final est passé en contexte au LLM (Ollama Qwen 2.5 7B par défaut).
11. Réponse streamée avec citations `[filename, page X]`.
    """)
