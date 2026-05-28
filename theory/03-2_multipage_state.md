# 3.2 — Architecture multipage et partage d'état

## La convention `pages/`

Streamlit suit une convention de filesystem pour le multipage : tout fichier `.py` placé dans un dossier `pages/` à côté du script principal devient une page accessible via la sidebar.

```
mvp/
├── streamlit_app.py        # entry point ("Home"), lancé par `streamlit run`
└── pages/
    ├── 1_Ingest.py
    ├── 2_Chat.py
    ├── 3_Graph_Explorer.py
    └── 4_Manage.py
```

Le préfixe numérique impose l'ordre dans la sidebar. Le caractère `_` devient un espace dans le label affiché. L'extension `.py` est invisible.

L'entry point (`streamlit_app.py` ici, ou n'importe quel nom passé à `streamlit run`) est la page d'accueil. Les autres sont accessibles via la sidebar auto-générée.

## Le routing implicite

Streamlit gère les URLs proprement : `localhost:8501/Ingest`, `localhost:8501/Chat`, etc. Bookmarkable, partageable.

Pour naviguer programmatiquement (par ex. après upload d'un PDF, basculer vers Chat) :

```python
st.page_link("pages/2_Chat.py", label="Aller poser une question", icon="💬")
```

`st.page_link` génère un lien cliquable. Il n'existe pas (encore) de redirection automatique programmatique sans clic utilisateur.

## Partage d'état : la mémoire commune

`st.session_state` est **partagé entre toutes les pages d'une même session**. C'est ce qui rend l'architecture viable.

Exemple typique pour Eigenmind :

```python
# Page Ingest
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

if st.button("Lancer l'ingestion"):
    ingest_pdf(uploaded_file)
    st.session_state.ingestion_done = True
    st.session_state.graph_cache = None  # invalider le cache graphe

# Page Chat
if st.session_state.get("graph_cache") is None:
    with st.spinner("Construction du graphe..."):
        cache = GraphAwareCache()
        cache.build()
        st.session_state.graph_cache = cache

results = hybrid_retrieve(question, st.session_state.graph_cache)
```

La logique est claire : Ingest invalide le cache, Chat le reconstruit si absent.

## Le cycle d'invalidation

C'est le point le plus subtil d'une app multipage avec données persistantes.

Trois caches/états à gérer :

| Élément | Vit où | Invalidé par |
|---|---|---|
| Embedder, client Qdrant | `@st.cache_resource` | Jamais (objets stables) |
| GraphAwareCache (Singular/Hinge/Theta) | `st.session_state.graph_cache` | Toute ingestion/suppression |
| Chat history | `st.session_state.chat_history` | Reset manuel utilisateur |

L'erreur classique : oublier d'invalider le `graph_cache` après suppression d'un fichier. Le retrieval continue de référencer des Singular nodes qui n'existent plus dans Qdrant, et tu te retrouves avec des `None` partout.

Pattern défensif : après **toute** modification de la collection Qdrant (ingest, delete), faire `st.session_state.graph_cache = None`. Le cache sera rebuilt à la prochaine requête de la page Chat.

## La distinction `cache_resource` vs `session_state`

Confusion fréquente. Récapitulatif :

**`@st.cache_resource`** :
- Une instance pour **tout le serveur**, partagée par toutes les sessions.
- Idéal pour : modèles ML, connexions DB, clients API.
- Avantage : économie de RAM massive (un modèle de 200 Mo n'est chargé qu'une fois).
- Limite : ne convient pas pour de la donnée propre à un utilisateur.

**`st.session_state`** :
- Un dict isolé **par session utilisateur** (en pratique, par onglet de navigateur).
- Idéal pour : état de l'utilisateur, historique, choix.
- Inconvénient : RAM × nombre d'utilisateurs.

Pour Eigenmind monolithe local : on n'a qu'un utilisateur à la fois, donc la distinction importe peu. Mais on garde le bon pattern pour la phase 4-5.

## Erreurs courantes du multipage

**Imports relatifs cassés.** Si tu importes `from ..mvp import xyz` dans une page, Streamlit ne sait pas gérer (chaque page est exécutée comme un script autonome). Solution : tous les modules helper (`embed_text.py`, `retrieve.py`, etc.) sont à plat dans `mvp/`, accessibles via imports absolus depuis n'importe quelle page tant qu'on lance Streamlit depuis `mvp/`.

**`st.set_page_config` appelé plusieurs fois.** Cette fonction doit être appelée **une seule fois par session**, dans l'entry point. Si tu la mets aussi dans une page, crash. On la met uniquement dans `streamlit_app.py`.

**Variables redéfinies à chaque rerun.** Une fonction définie dans une page est redéfinie à chaque rerun. Pas grave (Python est rapide). Mais une connexion DB ré-établie à chaque rerun, oui — d'où `@st.cache_resource`.

## Coordination Ingest → Chat

Le flow utilisateur typique :

1. **Page Ingest** : drag-and-drop d'un PDF, clic sur "Ingérer".
2. Le script appelle `ingest_pdf()`, met à jour Qdrant, set `st.session_state.ingestion_done = True`, invalide `st.session_state.graph_cache`.
3. Streamlit affiche un message de succès + un lien `st.page_link("pages/2_Chat.py")`.
4. L'utilisateur clique le lien.
5. **Page Chat** : au premier accès, détecte `graph_cache is None`, lance la reconstruction (sous spinner).
6. Une fois prête, l'utilisateur peut poser sa question.

Tout repose sur la cohérence de `st.session_state.graph_cache` entre les pages.

## Sidebar partagée

`st.sidebar` est partagé entre toutes les pages (sauf si une page la masque explicitement). Bonne pratique pour Eigenmind : mettre dans la sidebar les éléments globaux — état des services (Qdrant, Ollama), stats de la collection (nombre de chunks, taille du graphe), choix du LLM.

Chaque page peut quand même ajouter sa propre section de sidebar pour ses paramètres spécifiques.

## Performance de navigation

Le multipage de Streamlit re-execute la page complète à chaque navigation entre pages. Conséquence : si une page charge des données lourdes, la navigation y semble lente.

Mitigation : tout ce qui est cher doit être derrière `@st.cache_resource` ou `st.session_state`. Le rendu UI lui-même est rapide.

## Lien avec phase 4

En phase 4, on refactor vers un package `eigenmind/`. Les pages Streamlit déménagent dans `eigenmind/ui/pages/` et leur code est massivement allégé : juste l'UI, en appelant les fonctions de `pipelines/`. C'est la séparation **logique métier / UI** qu'on n'a pas encore en MVP.

Mais le pattern multipage et le rôle de `session_state` ne changent pas. Ce qu'on apprend ici reste valide.
