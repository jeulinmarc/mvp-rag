# Eigenmind — Roadmap de recodage

Ordre de travail : MVP fonctionnel d'abord (Phases 1-3 dans `mvp/`), puis refactor vers la structure du repo officiel (Phases 4-5 dans `final/`).

Pour chaque étape : un fichier de code + un document théorique dans `theory/`.

---

## Phase 1 — MVP RAG end-to-end (dossier `mvp/`)

But : un script CLI qui prend un PDF et répond à une question. Pas d'UI, pas de graphe.

- [x] **1.1** Setup environnement — `docker-compose.yml`, `.env`, `requirements.txt`, venv  
  *Théorie : vector DB, modèle client-serveur Qdrant*
- [x] **1.2** Embeddings — `embed_text.py`  
  *Théorie : embedding sémantique, 384-d, similarité cosinus, `all-MiniLM-L6-v2`*
- [x] **1.3** Chargement PDF + chunking — `load_pdf.py`  
  *Théorie : stratégies de chunking, overlap, trade-off taille/précision*
- [x] **1.4** Stockage Qdrant — `store_chunks.py`  
  *Théorie : collection, point, payload, vector, distance Cosine, HNSW*
- [ ] **1.5** Retrieval — `retrieve.py`  
  *Théorie : k-NN approximé, score threshold, MMR*
- [ ] **1.6** Appel LLM — `ask_llm.py`  
  *Théorie : structure d'un prompt RAG, température, max tokens*
- [ ] **1.7** Assemblage — `mini_rag.py`  
  *MVP CLI fonctionnel end-to-end*

## Phase 2 — Couche graphe spectrale (dossier `mvp/`)

But : enrichir le retrieval avec l'analyse spectrale du graphe sémantique.

- [ ] **2.1** Graphe de similarité — `build_graph.py`  
  *Théorie : matrice de similarité, k-NN graph, graphe pondéré symétrique*
- [ ] **2.2** Analyse spectrale — `spectral.py`  
  *Théorie : Laplacien L = D - A, eigendécomposition, connectivité algébrique, Fiedler vector*
- [ ] **2.3** Nœuds Singular — `singular.py`  
  *Théorie : centralité, originalité sémantique, info non-redondante*
- [ ] **2.4** Nœuds Hinge — `hinge.py`  
  *Théorie : betweenness centrality, articulation points*
- [ ] **2.5** Nœuds Theta — `theta.py`  
  *Théorie : modes propres intermédiaires du Laplacien*
- [ ] **2.6** Retrieval hybride — `hybrid_retrieve.py`  
  *Théorie : combiner similarité dense et signaux structurels du graphe*

## Phase 3 — Interface Streamlit (dossier `mvp/`)

But : transformer le CLI en app web utilisable.

- [ ] **3.1** App monopage — `streamlit_app.py`  
  *Théorie : `st.session_state`, file uploader, `@st.cache_resource`*
- [ ] **3.2** Découpage multipage — `pages/1_Ingest.py`, `pages/2_Chat.py`  
  *Théorie : convention multipage Streamlit, partage d'état*
- [ ] **3.3** Graph Explorer — `pages/3_Graph_Explorer.py`  
  *Théorie : layouts de graphe (spring, kamada-kawai), coloration des nœuds spéciaux*
- [ ] **3.4** Manage — `pages/4_Manage.py`  
  *Théorie : filtres Qdrant par payload, soft vs hard delete*

## Phase 4 — Refactor vers la structure du repo (dossier `final/`)

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

## Phase 5 — Connecteurs et features avancées (dossier `final/`)

- [ ] **5.1** OCR — extension de `core/document_loaders.py`  
  *Théorie : pytesseract, quand basculer sur OCR*
- [ ] **5.2** Google Drive connector — `connectors/google_drive.py`  
  *Théorie : OAuth 2.0 user flow vs service account, scopes*
- [ ] **5.3** SharePoint connector — `connectors/sharepoint.py`  
  *Théorie : Microsoft Graph API, Azure AD app registration*
- [ ] **5.4** Multi-user auth — `ui/auth.py`, `user_data/users.json`  
  *Théorie : password hashing (bcrypt/argon2), isolation par user*
- [ ] **5.5** Smart Resume — extension de `pipelines/ingest.py`  
  *Théorie : idempotence pipeline, filtres Qdrant sur payload*
- [ ] **5.6** Polish performance — lazy loading, GC, CPU-only torch  
  *Théorie : profil mémoire Transformer, lazy init*

---

## Index des documents théoriques

- `01-1_qdrant_vector_db.md`
- `01-2_embeddings.md`
- `01-3_chunking.md`
- `01-4_qdrant_storage.md`
- `01-5_retrieval.md`
- … (à étendre au fil des étapes)