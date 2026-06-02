# Eigenmind — Reimplementation from scratch

Reimplémentation pédagogique du projet [merlin-intelligence/eigenmind](https://github.com/merlin-intelligence/eigenmind) : un système de RAG (Retrieval-Augmented Generation) augmenté par une couche d'analyse spectrale de graphes sémantiques.

L'objectif de ce repo n'est pas de fournir un produit utilisable, mais de **comprendre** comment fonctionne un RAG moderne en le recodant entièrement, brique par brique. Chaque étape s'accompagne d'un document théorique dans `theory/`.

## Fonctionnalités

- **Ingestion de PDF** : extraction (OCR optionnel pour les scans / images), chunking, embeddings, stockage dans Qdrant.
- **RAG** : question en langage naturel → passages pertinents → réponse citée par un LLM local.
- **Couche graphe spectrale** : le corpus est analysé comme un graphe sémantique pour faire ressortir pôles thématiques, connecteurs et thèmes-frontière, et enrichir le retrieval.
- **Interface web Streamlit** : ingestion, chat, explorateur de graphe, gestion de la collection, lecture théorie ↔ code.
- **CLI** : `ingest` / `query` / `ask`, en retrieval dense, hybride, ou comparaison des deux.

Chaque brique est documentée dans `theory/`.

## Architecture

```
mvp-rag/
├── README.md                   # Ce fichier (à la racine)
├── docker-compose.yml          # Qdrant en local (infra → racine)
├── requirements.txt            # Dépendances Python (→ racine)
├── .env.example                # Gabarit de config (cp .env.example .env)
├── qdrant_storage/             # Données Qdrant (gitignored, volume Docker)
├── venv/                       # Environnement Python (gitignored)
├── src/                        # Code source uniquement
│   │   # pipeline RAG (ingestion → retrieval → LLM)
│   ├── embed_text.py           # Sentence embeddings (MiniLM)
│   ├── load_pdf.py             # Extraction PDF + chunking
│   ├── store_chunks.py         # Ingestion Qdrant
│   ├── retrieve.py             # Recherche dense top-k
│   ├── ask_llm.py              # Appel LLM (Ollama / Nebius / Groq)
│   ├── mini_rag.py             # Orchestrateur CLI (modes dense/hybrid/compare)
│   ├── reset_qdrant.py         # Drop de la collection (clean slate)
│   │
│   │   # couche graphe spectrale
│   ├── build_graph.py          # Graphe de similarité k-NN
│   ├── spectral.py             # Laplacien normalisé + décomposition propre
│   ├── singular.py             # Nœuds Singular (modes haute fréquence)
│   ├── hinge.py                # Nœuds Hinge (betweenness + Fiedler)
│   ├── theta.py                # Nœuds Theta (sous-clusters thématiques)
│   ├── hybrid_retrieve.py      # Fusion dense + signaux graphe (GraphAwareCache)
│   │
│   │   # interface Streamlit multipage
│   ├── streamlit_app.py        # Entrée / accueil
│   └── pages/
│       ├── 1_Ingest.py         # Upload + ingestion PDF
│       ├── 2_Chat.py           # Q/R (toggle hybride vs dense)
│       ├── 3_Graph_Explorer.py # Visualisation du graphe (Plotly)
│       ├── 4_Manage.py         # Gestion de la collection
│       └── 5_Theory.py         # Théorie ↔ code côte à côte
└── theory/                     # Documents théoriques (01-* à 03-*) + mémo officiel (PDF)
```

## Stack technique

- **Python 3.10+** (testé sur 3.13)
- **Qdrant** (vector database, en Docker)
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-d, CPU-friendly)
- **pypdf** + **LangChain text splitters** (extraction et chunking)
- **Ollama** (LLM local, par défaut `qwen2.5:7b`) ou n'importe quelle API OpenAI-compatible (Nebius, Groq, OpenRouter…)

Tout est gratuit, tout tourne en local. Aucune carte bancaire requise.

---

## Quickstart

> Installation détaillée (Docker, Ollama, Python pas-à-pas) et configuration des
> fournisseurs LLM : voir **[docs/INSTALL.md](docs/INSTALL.md)**.

Prérequis : Docker, Python 3.10+, Ollama. Ensuite, depuis la racine du repo :

```bash
git clone https://github.com/jeulinmarc/mvp-rag.git
cd mvp-rag

python3 -m venv venv && source venv/bin/activate   # venv à la racine
pip install -r requirements.txt
cp .env.example .env

docker compose up -d            # Qdrant (depuis la racine)
ollama pull qwen2.5:7b          # modèle LLM local (~4.7 Go)

cd src && streamlit run streamlit_app.py           # http://localhost:8501
```

## Utilisation

### Vérifier que tout tourne

Avant chaque session, vérifie que les deux services sont up :

```bash
curl -s http://localhost:6333/ > /dev/null && echo "Qdrant OK" || echo "Qdrant DOWN"
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama OK" || echo "Ollama DOWN"
```

Si Qdrant est down : `docker compose up -d` depuis la racine.
Si Ollama est down : `brew services start ollama` (ou `ollama serve` en foreground).

### Ingérer un PDF

```bash
python mini_rag.py ingest path/to/document.pdf
```

Sortie typique :

```
→ Ingestion de document.pdf
  87 chunks extraits
  87 points upserted dans 'documents'
  Total collection : 87 points
  Temps : 4.2s
```

L'ingestion est **idempotente** : ré-ingérer le même PDF n'ajoute pas de doublons, il écrase les points existants à l'identique.

### Poser une question

```bash
python mini_rag.py query "Ta question en langage naturel"
```

Sortie typique :

```
→ Question : What is the attention mechanism?
  Provider : ollama (qwen2.5:7b)

  5 chunks récupérés en 18ms :
    #1  score=0.652  document.pdf p3 — An attention function can be...
    #2  score=0.598  document.pdf p4 — Self-attention, sometimes...
    ...

────────────────────────────────────────────────────────────
Réponse :

L'attention mechanism est une fonction qui mappe une query et un ensemble de
paires key-value à un output [document.pdf, page 3]. La self-attention est
une variante qui relie différentes positions d'une même séquence
[document.pdf, page 4].

────────────────────────────────────────────────────────────
Temps : retrieve 18ms · LLM 6.2s
```

### Tout-en-un

```bash
python mini_rag.py ask path/to/document.pdf "Ta question"
```

Ingère puis interroge en une seule commande.

### Modes de retrieval (couche graphe)

Le `query` (et le `ask`) acceptent `--mode` :

```bash
python mini_rag.py query "..." --mode dense      # cosinus pur (RAG classique)
python mini_rag.py query "..." --mode hybrid     # + graphe spectral (défaut)
python mini_rag.py query "..." --mode compare    # les deux côte à côte
```

Le mode `compare` lance le retrieval dense **et** hybride, affiche le **Δ retrieval**
(quels chunks le graphe ajoute, via Singular/Hinge/Theta) puis génère **deux réponses**
— c'est la façon la plus directe de voir l'impact de la couche eigenmind sur le résultat.

> ⚠️ `hybrid` et `compare` construisent le graphe + la décomposition spectrale à chaque
> appel CLI (one-shot). C'est plus lent que `dense` ; dans Streamlit, ce cache est
> construit une seule fois par session.

### Options

```bash
python mini_rag.py query "..." -k 10           # top-10 chunks au lieu de top-5
python mini_rag.py query "..." --no-sources    # masquer la liste des chunks
python mini_rag.py --help                       # aide générale
python mini_rag.py query --help                 # aide d'une sous-commande
```

### Interface web (Streamlit)

Plutôt que le CLI, on peut lancer l'app multipage (ingestion, chat, explorateur de
graphe, gestion, théorie) :

```bash
streamlit run streamlit_app.py
```

Puis ouvrir [http://localhost:8501](http://localhost:8501). La page **Chat** propose un
toggle « Retrieval hybride (graphe) » pour comparer en direct hybride vs dense pur.

---

## Apprentissage et théorie

Chaque étape de l'implémentation est documentée en profondeur dans `theory/` :


| Fichier                       | Sujet                                                  |
| ----------------------------- | ------------------------------------------------------ |
| `00_roadmap.md`               | Plan général du projet, vue d'ensemble                 |
| `01-1_qdrant_vector_db.md`    | Vector DBs, HNSW, quantization, multi-tenancy          |
| `01-2_embeddings.md`          | Transformers, contrastive loss, MTEB, modèles français |
| `01-3_chunking.md`            | Stratégies de chunking, overlap, PDF tricky cases      |
| `01-4_qdrant_storage.md`      | Payload indexing, snapshots, write-ahead log           |
| `01-5_retrieval.md`           | MMR, rerankers, hybrid search BM25, HyDE, évaluation   |
| `01-6_llm_prompting.md`       | Pipeline d'inférence, température, prompt engineering  |
| `01-7_assemblage_pipeline.md` | Patterns d'orchestration, CLI design                   |
| `01-8_ingestion_avancee.md`   | Compléments mémo officiel : OCR, ChunkNorris, HNSW     |
| `02-1_similarity_graph.md`    | W=ΦΦᵀ seuillé (τ), sous-graphe BFS, deux régimes       |
| `02-2_spectral_analysis.md`   | Laplacien normalisé, vecteur de Fiedler, **Cheeger**   |
| `02-3_singular_nodes.md`      | Pôles thématiques (antipodes basses fréquences)        |
| `02-4_hinge_nodes.md`         | Connecteurs via champ géodésique log-similarité        |
| `02-5_theta_nodes.md`         | Thèmes-frontière : relaxation SDP de Lovász-θ          |
| `02-6_hybrid_retrieval.md`    | Agrégation `selection_tags` (3 labels)                 |
| `02-7_epistemologie_et_validation.md` | Niveaux de claim, failure modes, validation    |
| `260522_…_Cognitive_Maps.pdf` | **Mémo officiel Merlin Intelligence (fait foi)**       |
| `03-1_streamlit_fundamentals.md` | session_state, cache_resource, file uploader        |
| `03-2_multipage_state.md`     | Convention multipage, partage d'état, streaming LLM    |
| `03-3_graph_visualization.md` | Layouts de graphe, coloration par type, Plotly         |
| `03-4_manage.md`              | Filtres Qdrant par payload, soft vs hard delete        |


À lire en parallèle du code, dans l'ordre numérique.

---

## Troubleshooting

### `Connection refused: localhost:6333`

Qdrant n'est pas démarré. Lance :

```bash
docker compose up -d
```

### `Connection refused: localhost:11434`

Ollama n'est pas démarré. Lance :

```bash
brew services start ollama
```

### `model 'qwen2.5:7b' not found`

Le modèle n'a pas été téléchargé. Lance :

```bash
ollama pull qwen2.5:7b
```

> **Réglages du LLM** (réponse coupée, langue de réponse, « triche » hors contexte,
> latence…) ne sont pas des pannes mais des paramètres (`MAX_TOKENS`, `TEMPERATURE`,
> system prompt, choix du modèle) — voir `theory/01-6_llm_prompting.md`.

### `huggingface_hub.errors.RepositoryNotFoundError`

Typo dans le nom du modèle dans `embed_text.py`. Vérifie `sentence-transformers/all-MiniLM-L6-v2` (avec un **s** à `transformers`).

---

## Arrêter les services

Quand tu termines une session :

```bash
# Arrêter Qdrant (les données restent grâce au volume)
docker compose down

# Arrêter Ollama
brew services stop ollama

# Désactiver le venv Python
deactivate
```

Pour tout nettoyer (efface les données Qdrant) :

```bash
docker compose down -v
rm -rf qdrant_storage/
```

---

## Crédits et licence

Projet pédagogique. Repo original : [merlin-intelligence/eigenmind](https://github.com/merlin-intelligence/eigenmind).
