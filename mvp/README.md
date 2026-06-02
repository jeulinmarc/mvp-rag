# Eigenmind — Reimplementation from scratch

Reimplémentation pédagogique du projet [merlin-intelligence/eigenmind](https://github.com/merlin-intelligence/eigenmind) : un système de RAG (Retrieval-Augmented Generation) augmenté par une couche d'analyse spectrale de graphes sémantiques.

L'objectif de ce repo n'est pas de fournir un produit utilisable, mais de **comprendre** comment fonctionne un RAG moderne en le recodant entièrement, brique par brique. Chaque étape s'accompagne d'un document théorique dans `theory/`.

## État du projet

- ✅ **Phase 1** — MVP RAG end-to-end (CLI avec PDF + question → réponse citée)
- ✅ **Phase 2** — Couche graphe spectrale (Singular / Hinge / Theta nodes), branchée dans le CLI et l'UI
- ✅ **Phase 3** — Interface Streamlit multipage (5 pages validées end-to-end)
- ⬜ Phase 4 — Refactor vers la structure du repo officiel (package `eigenmind/`)
- ⬜ Phase 5 — Connecteurs (Google Drive, SharePoint), OCR, multi-user

Détail dans `theory/00_roadmap.md`.

## Architecture (état actuel — Phases 1 à 3)

```
eigenmind/
├── mvp/                        # MVP fonctionnel à plat
│   ├── docker-compose.yml      # Qdrant en local
│   ├── requirements.txt        # Dépendances Python
│   │
│   │   # Phase 1 — pipeline RAG
│   ├── embed_text.py           # Sentence embeddings (MiniLM)
│   ├── load_pdf.py             # Extraction PDF + chunking
│   ├── store_chunks.py         # Ingestion Qdrant
│   ├── retrieve.py             # Recherche dense top-k
│   ├── ask_llm.py              # Appel LLM (Ollama / Nebius / Groq)
│   ├── mini_rag.py             # Orchestrateur CLI (modes dense/hybrid/compare)
│   ├── reset_qdrant.py         # Drop de la collection (clean slate)
│   │
│   │   # Phase 2 — couche graphe spectrale
│   ├── build_graph.py          # Graphe de similarité k-NN
│   ├── spectral.py             # Laplacien normalisé + décomposition propre
│   ├── singular.py             # Nœuds Singular (modes haute fréquence)
│   ├── hinge.py                # Nœuds Hinge (betweenness + Fiedler)
│   ├── theta.py                # Nœuds Theta (sous-clusters thématiques)
│   ├── hybrid_retrieve.py      # Fusion dense + signaux graphe (GraphAwareCache)
│   │
│   │   # Phase 3 — interface Streamlit multipage
│   ├── streamlit_app.py        # Entrée / accueil
│   └── pages/
│       ├── 1_Ingest.py         # Upload + ingestion PDF
│       ├── 2_Chat.py           # Q/R (toggle hybride vs dense)
│       ├── 3_Graph_Explorer.py # Visualisation du graphe (Plotly)
│       ├── 4_Manage.py         # Gestion de la collection
│       └── 5_Theory.py         # Théorie ↔ code côte à côte
├── theory/                     # Documents théoriques (01-* à 03-*)
└── final/                      # Refactor à venir (Phase 4)
```

## Stack technique

- **Python 3.10+** (testé sur 3.13)
- **Qdrant** (vector database, en Docker)
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-d, CPU-friendly)
- **pypdf** + **LangChain text splitters** (extraction et chunking)
- **Ollama** (LLM local, par défaut `qwen2.5:7b`) ou n'importe quelle API OpenAI-compatible (Nebius, Groq, OpenRouter…)

Tout est gratuit, tout tourne en local. Aucune carte bancaire requise.

---

## Installation

### Prérequis

- **macOS** (testé sur M3 16 Go) ou Linux. Windows possible via WSL2 mais non testé.
- **8 Go RAM minimum**, 16 Go recommandés (pour faire tourner un LLM local de 7B paramètres confortablement).
- **5 Go d'espace disque** (Docker image Qdrant + modèle d'embedding + modèle LLM Ollama).

### 1. Cloner le repo

```bash
git clone https://github.com/jeulinmarc/eigenmind.git
cd eigenmind/mvp
```

### 2. Installer Docker Desktop

Docker fait tourner Qdrant dans un conteneur isolé. Sans Docker, il faudrait installer Qdrant en natif (compliqué).

**macOS** : télécharge Docker Desktop sur [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/). Choisis la version qui correspond à ton chip :

- **Apple Silicon** (M1/M2/M3/M4)
- **Intel**

Installe le `.dmg`, lance Docker Desktop une fois pour qu'il finalise le setup.

Alternatives plus légères si Docker Desktop te paraît lourd : [OrbStack](https://orbstack.dev), Colima, Rancher Desktop.

**Linux** : suis le guide officiel [docs.docker.com/engine/install](https://docs.docker.com/engine/install/).

**Vérifie** :

```bash
docker --version
docker compose version
```

Les deux commandes doivent répondre avec un numéro de version.

### 3. Installer Python 3.10+

**macOS** (via Homebrew, recommandé) :

```bash
brew install python@3.12
```

**Vérifie** :

```bash
python3 --version
```

### 4. Installer Ollama (LLM local)

Ollama fait tourner des LLMs open-source en local. Aucun compte, aucune clé API.

**macOS** :

```bash
brew install ollama
```

**Linux** :

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Lance Ollama en service système** (démarre automatiquement, survit au reboot) :

```bash
brew services start ollama          # macOS
# ou
sudo systemctl enable --now ollama   # Linux
```

**Télécharge le modèle par défaut** (~4.7 Go, prends ~5 min selon ta connexion) :

```bash
ollama pull qwen2.5:7b
```

**Vérifie** :

```bash
curl http://localhost:11434/api/tags
```

Doit retourner un JSON listant les modèles installés.

### 5. Setup Python et dépendances

Depuis `mvp/` :

```bash
# Créer le venv
python3 -m venv venv

# Activer le venv (à refaire à chaque session)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

Tu sauras que le venv est actif quand ton prompt commence par `(venv)`.

### 6. Configurer les variables d'environnement

Crée un fichier `.env` à la racine de `mvp/` :

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
LLM_PROVIDER=ollama
```

Le fichier `.env` est dans le `.gitignore`, il ne sera jamais commité.

### 7. Lancer Qdrant

Depuis `mvp/`, avec Docker Desktop actif :

```bash
docker compose up -d
```

Le `-d` (detached) lance le conteneur en arrière-plan. Vérifie :

```bash
curl http://localhost:6333/
```

Doit répondre par un petit JSON contenant `"title":"qdrant"`. Tu peux aussi ouvrir [http://localhost:6333/dashboard](http://localhost:6333/dashboard) dans ton navigateur — c'est l'UI web de Qdrant.

---

## Utilisation

### Vérifier que tout tourne

Avant chaque session, vérifie que les deux services sont up :

```bash
curl -s http://localhost:6333/ > /dev/null && echo "Qdrant OK" || echo "Qdrant DOWN"
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama OK" || echo "Ollama DOWN"
```

Si Qdrant est down : `docker compose up -d` depuis `mvp/`.
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

## Changer de LLM

Le code supporte n'importe quelle API OpenAI-compatible. Édite `.env` :

### Ollama (défaut, gratuit, local)

```
LLM_PROVIDER=ollama
```

Modèles supportés out-of-the-box : `qwen2.5:7b`, `qwen2.5:3b`, `llama3.1:8b`, etc. Change `PROVIDER_CONFIG["ollama"]["model"]` dans `ask_llm.py`.

### Nebius (cloud, payant après crédits gratuits)

```
LLM_PROVIDER=nebius
NEBIUS_API_KEY=ton_token
```

Compte sur [studio.nebius.com](https://studio.nebius.com) (carte bancaire requise depuis fin 2025).

### Groq (cloud, gratuit, sans CB)

Ajoute Groq dans `PROVIDER_CONFIG` de `ask_llm.py` :

```python
"groq": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": os.getenv("GROQ_API_KEY", ""),
    "model": "llama-3.3-70b-versatile",
},
```

Compte sur [console.groq.com](https://console.groq.com), puis dans `.env` :

```
LLM_PROVIDER=groq
GROQ_API_KEY=ton_token
```

---

## Apprentissage et théorie

Chaque étape de l'implémentation est documentée en profondeur dans `theory/` :


| Fichier                       | Sujet                                                  |
| ----------------------------- | ------------------------------------------------------ |
| `00_roadmap.md`               | Plan général du projet, toutes phases                  |
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

### Réponse LLM coupée en milieu de phrase

`MAX_TOKENS` trop bas dans `ask_llm.py`. Passe à `2048` ou `4096`.

### Le LLM répond en anglais alors que la question est en français

Le system prompt n'est pas assez ferme. Ajoute "Réponds toujours en français." dans `build_messages()` de `ask_llm.py`.

### Le LLM "triche" avec ses connaissances générales au lieu du contexte

Baisse `TEMPERATURE` à `0.1` ou `0` dans `ask_llm.py`. Renforce l'instruction de refus dans le system prompt.

### Latence LLM trop élevée (>15s)

Normal sur CPU avec Qwen 7B. Options :

- Passer à `qwen2.5:3b` (plus rapide, qualité moindre).
- Activer MPS sur Apple Silicon : `device="mps"` dans `embed_text.py`.
- Utiliser un provider distant rapide (Groq).

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