# Eigenmind — Roadmap de recodage

Ordre de travail : MVP fonctionnel d'abord (Phases 1-3 dans `src/`), puis refactor vers la structure du repo officiel (Phases 4-5 dans `final/`).

Pour chaque étape : un fichier de code + un document théorique dans `theory/`.

**État actuel** : Phases 1, 2 et 3 terminées (CLI aligné sur le retrieval hybride avec mode `compare` ; 5 pages Streamlit validées end-to-end). MVP fonctionnellement complet. Phase 4 à démarrer.

---

## Phase 1 — MVP RAG end-to-end (dossier `src/`) ✅

But : un script CLI qui prend un PDF et répond à une question. Pas d'UI, pas de graphe.

- [x] **1.1** Setup environnement — `docker-compose.yml`, `.env`, `requirements.txt`, venv
  *Théorie : vector DB, modèle client-serveur Qdrant*
- [x] **1.2** Embeddings — `embed_text.py`
  *Théorie : embedding sémantique, 384-d, similarité cosinus, `all-MiniLM-L6-v2`*
- [x] **1.3** Chargement PDF + chunking — `load_pdf.py`
  *Théorie : stratégies de chunking, overlap, trade-off taille/précision*
- [x] **1.4** Stockage Qdrant — `store_chunks.py`
  *Théorie : collection, point, payload, vector, distance Cosine, HNSW, IDs déterministes SHA-1*
- [x] **1.5** Retrieval — `retrieve.py`
  *Théorie : k-NN approximé, score threshold, MMR*
- [x] **1.6** Appel LLM — `ask_llm.py`
  *Théorie : structure d'un prompt RAG, température, max tokens, Ollama API OpenAI-compatible*
- [x] **1.7** Assemblage — `mini_rag.py`
  *MVP CLI fonctionnel end-to-end. Sous-commandes `ingest` / `query` / `ask` ; option `--mode dense|hybrid|compare` (voir 2.6).*

## Phase 2 — Couche graphe spectrale (dossier `src/`) ✅

But : enrichir le retrieval avec l'analyse spectrale du graphe sémantique.

- [x] **2.1** Graphe de similarité — `build_graph.py`
  *Théorie (mémo) : `W = ΦΦᵀ` seuillé à `τ=0.65`, diagonale nulle, sous-graphe local par BFS, deux régimes (top-k vs exploration). Code MVP : k-NN global (k=10) + union — à aligner.*
- [x] **2.2** Analyse spectrale — `spectral.py`
  *Théorie (mémo) : Laplacien normalisé `L_sym`, `np.linalg.eigh`, vecteur de Fiedler, énergie de Dirichlet, **inégalité de Cheeger**.*
- [x] **2.3** Nœuds Singular — `singular.py`
  *Théorie (mémo) : **pôles thématiques** = extrema des modes **basses** fréquences (antipodes spectraux). Code MVP : atypisme hautes fréquences — à aligner.*
- [x] **2.4** Nœuds Hinge — `hinge.py`
  *Théorie (mémo) : **champ géodésique** sur `ℓ=−log W` (Dijkstra, source périphérique, `H(i)=B(i)(1−|x|)`). Code MVP : betweenness + Fiedler — à aligner.*
- [x] **2.5** Nœuds Theta — `theta.py`
  *Théorie (mémo) : **relaxation SDP du nombre de Lovász-θ** (dual Lemaréchal–Oustry, `FS(i)=‖y_i‖²`, farthest-point). Code MVP : modes propres intermédiaires — à aligner.*
- [x] **2.6** Retrieval hybride — `hybrid_retrieve.py`
  *Théorie (mémo) : agrégation `selection_tags` → tagging à 3 labels (Singular/Hinge/Theta), deux régimes top-k vs BFS, « chunks nommés pas vecteurs latents ». Code MVP : fusion par boosts (cos > 0.30 ; +0.10/+0.07/+0.05) sur `GraphAwareCache` — à aligner.*
  *Exposé dans le CLI (`mini_rag.py --mode hybrid`) et dans la page Chat. Le mode `compare` lance dense + hybride en parallèle et affiche le Δ retrieval + les deux réponses, pour mesurer l'impact de la couche spectrale.*

## Phase 3 — Interface Streamlit (dossier `src/`) ✅

But : transformer le CLI en app web utilisable.

- [x] **3.1** App monopage / entrée — `streamlit_app.py`
  *Théorie : `st.session_state`, file uploader, `@st.cache_resource`, layout wide*
- [x] **3.2** Découpage multipage — `pages/1_Ingest.py`, `pages/2_Chat.py`
  *Théorie : convention multipage Streamlit, partage d'état entre pages, streaming LLM*
- [x] **3.3** Graph Explorer — `pages/3_Graph_Explorer.py`
  *Théorie : layouts de graphe (spring, kamada-kawai), coloration par type de nœud, Plotly*
- [x] **3.4** Manage — `pages/4_Manage.py`
  *Théorie : filtres Qdrant par payload, soft vs hard delete, agrégation par filename*
- [x] **3.5** Theory page (théorie ↔ code côte à côte) — `pages/5_Theory.py`
  *Page méta : sélecteur de chapitre, rendu Markdown du `theory/*.md` à côté du fichier code correspondant (onglets si plusieurs). Pas de `.md` théorique dédié (page utilitaire).*

## Phase 4 — Refactor vers la structure du repo (dossier `final/`) ⬜

But : ranger le code MVP dans l'arborescence du package `eigenmind/`.

- [ ] **4.1** Packaging — `pyproject.toml`, `eigenmind/__init__.py`, `pip install -e .`
  *Théorie : install éditable, entry points, extras optionnels*
- [ ] **4.2** Config — `eigenmind/config.py`
  *Théorie : 12-factor app, fallback `.env` → `st.secrets`*
- [ ] **4.3** Core — `core/embeddings.py`, `core/document_loaders.py`, `core/chunking.py`, `core/llm.py`
  *Théorie : architecture en couches, single responsibility*
- [ ] **4.4** VectorDB — `vectordb/qdrant_client.py`, `vectordb/ingestion.py`, `vectordb/retrieval.py`
  *Théorie : pattern repository, namespacing multi-user*
- [ ] **4.5** Graph — `graph/build_graph.py`, `graph/spectral.py`, `graph/singular.py`, `graph/hinge.py`, `graph/theta.py`, `graph/exploration.py`
  *Théorie : orchestration de subgraphes thématiques*
- [ ] **4.6** Pipelines — `pipelines/ingest.py`, `pipelines/rag.py`
  *Théorie : pattern pipeline*
- [ ] **4.7** UI — `eigenmind/ui/`
  *Théorie : séparation logique métier / UI*
- [ ] **4.8** Scripts & tests — `scripts/ingest_recursive.py`, `tests/unit/`
  *Théorie : tests purs numpy, déterminisme, fixtures*

## Phase 5 — Features avancées

> **Décision (juin 2026)** : seul l'**OCR** est ajouté au MVP (vrai besoin, auto-contenu). Le reste
> (connecteurs Drive/SharePoint, multi-user, Smart Resume, perf) **n'est pas fait dans `mvp-rag`** :
> ces features existent déjà dans le repo de production `merlin-intelligence/eigenmind`.

- [x] **5.1** OCR — `src/load_pdf.py` (`_ocr_page`, param `ocr=auto|always|never`)
  *Théorie : canal bruité, seuil caractères/page **par page** (PDF mixtes), `pdf2image`→`pytesseract`. Cf. `01-8`.*
- [ ] ~~**5.2** Google Drive connector~~ — non fait dans le MVP (repo de prod)
- [ ] ~~**5.3** SharePoint connector~~ — non fait dans le MVP (repo de prod)
- [ ] ~~**5.4** Multi-user auth~~ — non fait dans le MVP (repo de prod)
- [ ] ~~**5.5** Smart Resume~~ — non fait dans le MVP (repo de prod)
- [ ] ~~**5.6** Polish performance~~ — non fait dans le MVP (repo de prod)

---

## Index des documents théoriques

> **Document de référence.** Le mémo technique officiel de Merlin Intelligence est archivé dans
> `theory/260522_Eigenmind_Cognitive_Maps.pdf`. **En cas de divergence, c'est le PDF qui fait
> foi** : les docs Phase 2 ont été corrigés en conséquence (les définitions de Singular, Hinge et
> Theta du MVP divergeaient ; voir les notes « ⚠️ Note d'implémentation MVP » en fin de chaque doc).

**Phase 1 — RAG**
- `01-1_qdrant_vector_db.md`
- `01-2_embeddings.md`
- `01-3_chunking.md`
- `01-4_qdrant_storage.md`
- `01-5_retrieval.md`
- `01-6_llm_prompting.md`
- `01-7_assemblage_pipeline.md`
- `01-8_ingestion_avancee.md` — compléments mémo officiel (OCR canal bruité, ChunkNorris, HNSW)

**Phase 2 — Graphe spectral** (corrigés d'après le mémo officiel)
- `02-1_similarity_graph.md` — W=ΦΦᵀ seuillé (τ), sous-graphe BFS, deux régimes
- `02-2_spectral_analysis.md` — Laplacien, Fiedler, **Cheeger**
- `02-3_singular_nodes.md` — **pôles thématiques** (antipodes basses fréquences)
- `02-4_hinge_nodes.md` — **champ géodésique** log-similarité
- `02-5_theta_nodes.md` — **relaxation SDP de Lovász-θ**
- `02-6_hybrid_retrieval.md` — agrégation `selection_tags` (3 labels)
- `02-7_epistemologie_et_validation.md` — 3 niveaux de claim, failure modes, validation, refs

**Phase 3 — Streamlit**
- `03-1_streamlit_fundamentals.md`
- `03-2_multipage_state.md`
- `03-3_graph_visualization.md`
- `03-4_manage.md`

**Phase 4-5** — à écrire au fil des étapes.
