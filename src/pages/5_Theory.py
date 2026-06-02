"""
Theory page — side-by-side view of a theory chapter (Markdown) and its
associated source code, to make the link between explanation and implementation
explicit.

Layout :
- Sidebar : chapter selector only
- Main : two panes — left = rendered Markdown, right = code (with file paths)

The chapter → (markdown, [code files]) mapping is declared explicitly below
so it stays in sync with the agreed convention "1 .py + 1 theory/.md per step".
"""
from pathlib import Path
import streamlit as st

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so `streamlit run` works from anywhere
# ---------------------------------------------------------------------------

MVP_DIR = Path(__file__).resolve().parent.parent      # .../eigenmind/mvp
ROOT_DIR = MVP_DIR.parent                              # .../eigenmind
THEORY_DIR = ROOT_DIR / "theory"
PAGES_DIR = MVP_DIR / "pages"


# ---------------------------------------------------------------------------
# Chapter mapping
#   key   = displayed title in the selector
#   md    = theory file (relative to THEORY_DIR)
#   code  = list of code files (relative to MVP_DIR) — empty list = intro only
# ---------------------------------------------------------------------------

CHAPTERS: list[dict] = [
    {
        "title": "0. Roadmap",
        "md": "00_roadmap.md",
        "code": [],
    },
    # Phase 1 — RAG end-to-end
    {
        "title": "1.1 — Qdrant (vector DB)",
        "md": "01-1_qdrant_vector_db.md",
        "code": ["docker-compose.yml"],
    },
    {
        "title": "1.2 — Sentence embeddings",
        "md": "01-2_embeddings.md",
        "code": ["embed_text.py"],
    },
    {
        "title": "1.3 — Chunking PDF",
        "md": "01-3_chunking.md",
        "code": ["load_pdf.py"],
    },
    {
        "title": "1.4 — Stockage Qdrant",
        "md": "01-4_qdrant_storage.md",
        "code": ["store_chunks.py"],
    },
    {
        "title": "1.5 — Retrieval dense",
        "md": "01-5_retrieval.md",
        "code": ["retrieve.py"],
    },
    {
        "title": "1.6 — Prompting LLM",
        "md": "01-6_llm_prompting.md",
        "code": ["ask_llm.py"],
    },
    {
        "title": "1.7 — Assemblage pipeline",
        "md": "01-7_assemblage_pipeline.md",
        "code": ["mini_rag.py"],
    },
    {
        "title": "1.8 — Ingestion avancée (mémo officiel)",
        "md": "01-8_ingestion_avancee.md",
        "code": ["load_pdf.py", "embed_text.py", "store_chunks.py"],
    },
    # Phase 2 — Spectral graph layer
    {
        "title": "2.1 — Graphe de similarité",
        "md": "02-1_similarity_graph.md",
        "code": ["build_graph.py"],
    },
    {
        "title": "2.2 — Analyse spectrale",
        "md": "02-2_spectral_analysis.md",
        "code": ["spectral.py"],
    },
    {
        "title": "2.3 — Singular nodes",
        "md": "02-3_singular_nodes.md",
        "code": ["singular.py"],
    },
    {
        "title": "2.4 — Hinge nodes",
        "md": "02-4_hinge_nodes.md",
        "code": ["hinge.py"],
    },
    {
        "title": "2.5 — Theta nodes",
        "md": "02-5_theta_nodes.md",
        "code": ["theta.py"],
    },
    {
        "title": "2.6 — Retrieval hybride",
        "md": "02-6_hybrid_retrieval.md",
        "code": ["hybrid_retrieve.py"],
    },
    {
        "title": "2.7 — Épistémologie & validation (mémo officiel)",
        "md": "02-7_epistemologie_et_validation.md",
        "code": [],
    },
    # Phase 3 — Streamlit UI
    {
        "title": "3.1 — Streamlit fundamentals",
        "md": "03-1_streamlit_fundamentals.md",
        "code": ["streamlit_app.py"],
    },
    {
        "title": "3.2 — Multipage & state",
        "md": "03-2_multipage_state.md",
        "code": ["pages/1_Ingest.py", "pages/2_Chat.py"],
    },
    {
        "title": "3.3 — Visualisation du graphe",
        "md": "03-3_graph_visualization.md",
        "code": ["pages/3_Graph_Explorer.py"],
    },
    {
        "title": "3.4 — Manage",
        "md": "03-4_manage.md",
        "code": ["pages/4_Manage.py"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map common extensions → Pygments language hint used by st.code
EXT_TO_LANG = {
    ".py": "python",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".md": "markdown",
    ".json": "json",
}


def read_text(path: Path) -> str:
    """Safe read — returns an error placeholder if the file is missing."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"_(fichier introuvable : `{path}`)_"
    except Exception as e:  # pragma: no cover
        return f"_(erreur de lecture : {e})_"


def code_language(path: Path) -> str:
    return EXT_TO_LANG.get(path.suffix.lower(), "text")


# ---------------------------------------------------------------------------
# Sidebar — selector + options
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Theory · Eigenmind", page_icon="📚", layout="wide")

st.sidebar.markdown("### 📚 Theory")
st.sidebar.caption("Lis la théorie à côté de son implémentation.")

titles = [c["title"] for c in CHAPTERS]
selected_title = st.sidebar.selectbox("Chapitre", titles, index=0)
chapter = next(c for c in CHAPTERS if c["title"] == selected_title)


# ---------------------------------------------------------------------------
# Main — title and panels
# ---------------------------------------------------------------------------

st.title("📚 Théorie & Code")
st.caption(
    "Sélectionne un chapitre dans la barre de gauche. La théorie et le code "
    "correspondant s'affichent côte à côte pour que tu puisses faire le lien."
)
st.divider()


def render_theory_pane(md_rel: str) -> None:
    md_path = THEORY_DIR / md_rel
    st.caption(f"📄 `theory/{md_rel}`")
    st.markdown(read_text(md_path))


def render_code_pane(code_paths: list[str]) -> None:
    if not code_paths:
        st.info("Pas de code associé à ce chapitre (intro / roadmap).")
        return

    # If several files, show one tab per file
    if len(code_paths) > 1:
        tabs = st.tabs([Path(p).name for p in code_paths])
        for tab, rel in zip(tabs, code_paths):
            with tab:
                _render_single_code(rel)
    else:
        _render_single_code(code_paths[0])


def _render_single_code(rel: str) -> None:
    # Code files live in src/ ; infra files (docker-compose…) live at the repo root.
    path = MVP_DIR / rel
    if not path.exists():
        path = ROOT_DIR / rel
    try:
        shown = path.relative_to(ROOT_DIR)
    except ValueError:
        shown = rel
    st.caption(f"💻 `{shown}`")
    st.code(read_text(path), language=code_language(path), line_numbers=True)


# Fixed layout : theory left, code right, paths always shown
left, right = st.columns(2, gap="large")
with left:
    render_theory_pane(chapter["md"])
with right:
    render_code_pane(chapter["code"])
