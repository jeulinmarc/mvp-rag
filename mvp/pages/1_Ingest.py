"""
Ingest page — upload a PDF, chunk it, embed it, store in Qdrant.
Invalidates the graph cache so the next Chat session rebuilds it.
"""
import time
from pathlib import Path
import tempfile
import streamlit as st

from load_pdf import load_and_chunk
from store_chunks import upsert_chunks, COLLECTION_NAME

st.title("📥 Ingest")
st.caption(
    "Uploade un PDF pour l'indexer. L'ingestion est idempotente : "
    "réingérer le même fichier ne crée pas de doublons."
)


# ---------------------------------------------------------------------------
# Upload widget
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choisis un PDF",
    type=["pdf"],
    help="Le fichier sera stocké temporairement pendant l'ingestion puis supprimé.",
)


def _save_to_temp(uploaded) -> Path:
    """Write uploaded file content to a temporary path on disk."""
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Ingestion flow
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"Fichier reçu : **{uploaded_file.name}** "
                   f"({uploaded_file.size / 1024:.1f} KB)")
    with col2:
        ingest_clicked = st.button("🚀 Ingérer", type="primary", use_container_width=True)

    if ingest_clicked:
        progress = st.progress(0.0, text="Préparation...")

        try:
            # Step 1 — save to temp file
            progress.progress(0.1, text="Sauvegarde temporaire du PDF...")
            tmp_path = _save_to_temp(uploaded_file)
            # Override the filename in chunks to keep the original name
            original_name = uploaded_file.name

            # Step 2 — load and chunk
            progress.progress(0.3, text="Extraction du texte et chunking...")
            t0 = time.time()
            chunks = load_and_chunk(tmp_path)
            for c in chunks:
                c["filename"] = original_name  # override tmp name
            t_chunk = time.time() - t0

            if not chunks:
                st.error(
                    "Aucun texte extrait du PDF. Probablement un PDF scanné. "
                    "L'OCR sera ajouté en phase 5."
                )
                st.stop()

            # Step 3 — embed + upsert
            progress.progress(0.6, text=f"Embedding et upsert de {len(chunks)} chunks...")
            t0 = time.time()
            n = upsert_chunks(chunks)
            t_upsert = time.time() - t0

            # Step 4 — invalidate graph cache
            progress.progress(0.9, text="Invalidation du cache graphe...")
            st.session_state.graph_cache = None
            st.session_state.ingestion_done = True
            st.session_state.last_ingest = {
                "filename": original_name,
                "chunks": len(chunks),
                "points_upserted": n,
                "time_chunking": t_chunk,
                "time_upsert": t_upsert,
            }

            # Cleanup temp file
            tmp_path.unlink(missing_ok=True)

            progress.progress(1.0, text="Terminé !")
            time.sleep(0.5)
            progress.empty()

            st.success(f"✅ Ingestion réussie : {n} chunks indexés.")
            st.balloons()

        except Exception as e:
            progress.empty()
            st.error(f"Erreur pendant l'ingestion : {e}")
            st.exception(e)


# ---------------------------------------------------------------------------
# Show last ingestion summary
# ---------------------------------------------------------------------------

if "last_ingest" in st.session_state and st.session_state.last_ingest:
    info = st.session_state.last_ingest
    st.divider()
    st.subheader("Dernière ingestion")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fichier", info["filename"][:25] + ("…" if len(info["filename"]) > 25 else ""))
    c2.metric("Chunks", info["chunks"])
    c3.metric("Chunking", f"{info['time_chunking']:.1f}s")
    c4.metric("Embed+Upsert", f"{info['time_upsert']:.1f}s")

    st.page_link("pages/2_Chat.py", label="→ Aller poser une question", icon="💬")


# ---------------------------------------------------------------------------
# Tips
# ---------------------------------------------------------------------------

with st.expander("ℹ️ À savoir"):
    st.markdown("""
- **Idempotence** : ré-ingérer le même PDF n'ajoute pas de doublons.
  Les `chunk_id` sont des hash déterministes de (filename, page, chunk_index).
- **Invalidation du graphe** : à chaque ingestion, le cache du graphe spectral est
  effacé. Il sera reconstruit à la prochaine requête sur la page Chat.
- **Limites actuelles** :
  - PDF scannés non supportés (OCR en phase 5).
  - Pas de DOCX/XLSX/PPTX (extension à venir en phase 4).
  - Pas de limite de taille mais les très gros PDF (>1000 pages) peuvent ralentir.
    """)
