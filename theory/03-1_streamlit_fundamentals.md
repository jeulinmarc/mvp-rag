# 3.1 — Streamlit : fondamentaux et architecture multipage

## Pourquoi Streamlit

Streamlit transforme un script Python en application web interactive. Pas de HTML, pas de JavaScript, pas de routing à écrire — tu écris du Python de haut en bas et chaque widget (bouton, slider, input) devient un élément d'UI.

Trois forces qui le rendent dominant dans l'écosystème ML/data :

**1. Vitesse de prototypage.** Une app utile en 50 lignes. Aucune autre stack n'égale ça pour la phase d'exploration.

**2. Le pattern "rerun".** Chaque interaction (clic, modification de widget) re-exécute **tout le script de haut en bas**. C'est très inhabituel mais ça simplifie énormément le mental model : pas d'événements, pas de state machine, juste un script qui tourne en boucle. On verra les implications.

**3. Écosystème mature.** Plein de composants tiers, hébergement gratuit via Streamlit Cloud, intégration native avec pandas/plotly/altair.

Limites à connaître : pas adapté pour des apps multi-utilisateurs en prod (pas de gestion d'auth native, perf modeste), pas idéal pour des UX très customisées (on est limité par les composants). Pour Eigenmind à l'échelle d'un dev unique ou d'une démo, c'est parfait.

## Le modèle d'exécution : rerun-on-interaction

Concept central à intégrer pour ne pas se cogner dessus pendant des heures.

Quand un utilisateur clique un bouton ou modifie un slider, Streamlit **re-exécute tout le fichier `app.py` depuis le début**. Ce n'est pas une mise à jour partielle de l'UI — c'est une nouvelle exécution complète du script.

Conséquences :

**Tout ce qui est lent est exécuté à chaque rerun**, sauf si tu le caches explicitement. Charger un modèle de 200 Mo à chaque clic ? Désastre. D'où le besoin critique de `@st.cache_resource` et `@st.cache_data`.

**Les variables Python ordinaires sont perdues entre les reruns.** Si tu fais `count = count + 1` à chaque clic d'un bouton, count ne s'incrémente jamais — il est réinitialisé à chaque exécution. D'où le besoin de `st.session_state`.

**Pas de notion d'event handler.** Pas de "onClick callback" au sens React. Tu écris la logique de manière séquentielle : "si le bouton a été cliqué, fais X". Streamlit te fournit cette info via `if st.button("..."):`.

## `st.session_state` : la mémoire entre reruns

`st.session_state` est un dictionnaire persistant **par session utilisateur** (pas par app globale — chaque onglet de navigateur a sa propre session).

```python
import streamlit as st

# Initialisation safe avec setdefault
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Incrémenter"):
    st.session_state.counter += 1

st.write(f"Compteur : {st.session_state.counter}")
```

À chaque rerun, `st.session_state.counter` garde sa valeur. C'est la primitive fondamentale pour tout ce qui doit persister : un graphe construit, des résultats de retrieval, l'historique d'un chat, le PDF actuellement ingéré.

Pour Eigenmind on stockera typiquement :

- `st.session_state.graph_cache` — le `GraphAwareCache` construit une fois et réutilisé
- `st.session_state.chat_history` — l'historique des questions/réponses
- `st.session_state.last_results` — les chunks retournés au dernier retrieval

## Le caching : `@st.cache_resource` vs `@st.cache_data`

Streamlit fournit deux décorateurs pour éviter de recalculer.

**`@st.cache_resource`** : pour les objets **partagés entre sessions**, typiquement des connexions, modèles, clients. Une seule instance pour tout le serveur. Idéal pour :

```python
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)
```

Le modèle est chargé une fois, partagé par tous les utilisateurs. Économise mémoire et temps.

**`@st.cache_data`** : pour les **données** (sérialisables : DataFrame, dict, list...). Re-exécutée si les arguments changent.

```python
@st.cache_data
def load_pdf_chunks(pdf_path: str) -> list[dict]:
    return load_and_chunk(pdf_path)
```

Tant que `pdf_path` ne change pas, la fonction n'est plus exécutée — son résultat est restitué depuis le cache. Si tu fournis un nouveau path, recalcul.

Règle de pouce : pour Eigenmind, `cache_resource` pour le `GraphAwareCache`, l'embedder, le client Qdrant. `cache_data` pour les fonctions de traitement (chunking, extraction).

## Layout : columns, containers, expanders

Streamlit déroule les éléments verticalement par défaut. Trois primitives pour structurer :

**`st.columns(n)`** crée n colonnes côte à côte :

```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Chunks", 87)
with col2:
    st.metric("Singular", 13)
with col3:
    st.metric("Hinge", 12)
```

**`st.container()`** crée un bloc nommé — utile pour rafraîchir un élément spécifique sans tout rerun (avec `st.empty()` pour vider et redessiner).

**`st.expander("Détails")`** crée une section repliable. Idéal pour cacher des détails techniques (sources d'une réponse, paramètres avancés) sans alourdir l'UI.

**`st.sidebar`** : barre latérale persistante pour les paramètres globaux. On y mettra le sélecteur de modèle LLM, le top-k, etc.

## Widgets et leurs valeurs

Chaque widget retourne sa valeur actuelle :

```python
question = st.text_input("Votre question")
k = st.slider("Top-k", min_value=1, max_value=20, value=5)
temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.1)
provider = st.selectbox("Provider", ["ollama", "groq"])

if st.button("Envoyer"):
    response = ask_llm(question, k=k, temperature=temperature, provider=provider)
    st.write(response)
```

Quand l'utilisateur modifie le slider, le script rerun avec la nouvelle valeur de `k`, mais le bouton "Envoyer" n'est pas "cliqué" automatiquement. La condition `if st.button(...)` n'est `True` que sur le rerun déclenché par le clic du bouton.

**Subtilité fréquente** : si tu cliques le bouton, le script rerun avec `st.button(...) == True`, fait le travail, et affiche le résultat. Au rerun suivant (déclenché par autre chose, par ex modifier le slider), le bouton n'est plus "cliqué" et `st.button(...) == False` — le résultat disparaît. Solution : stocker le résultat dans `session_state`.

## Architecture multipage

Streamlit 1.10+ supporte nativement le multipage via un dossier `pages/` à côté du fichier principal :

```
src/
├── streamlit_app.py        # page d'accueil (entry point)
└── pages/
    ├── 1_Ingest.py
    ├── 2_Chat.py
    ├── 3_Graph_Explorer.py
    └── 4_Manage.py
```

Le préfixe `N_` numérote l'ordre dans la sidebar. Le `_` devient un espace dans le label affiché. Streamlit génère automatiquement la navigation latérale.

**Lancement** : `streamlit run streamlit_app.py` depuis le dossier `src/`. Streamlit détecte le dossier `pages/` et construit la nav.

**Partage d'état entre pages** : `st.session_state` est partagé automatiquement entre toutes les pages de la même session. Si tu construis le cache dans `streamlit_app.py`, il est dispo dans `pages/2_Chat.py`. C'est ce qui rend l'architecture utilisable.

## Streaming des réponses LLM

Pour les LLM lents (Ollama CPU à 10s par réponse), afficher la réponse token par token transforme l'UX. Pattern Streamlit :

```python
def stream_response(question, chunks):
    response_placeholder = st.empty()
    accumulated = ""
    for token in llm_stream(question, chunks):
        accumulated += token
        response_placeholder.markdown(accumulated)
```

`st.empty()` crée un emplacement réutilisable, qu'on met à jour à chaque token. L'utilisateur voit le texte apparaître progressivement.

L'API OpenAI-compatible (Ollama incluse) supporte `stream=True` qui retourne un générateur de chunks. C'est ce qu'on utilisera en page Chat.

## Performance : ce qui pèse en RAM

Streamlit garde **chaque session en RAM**. Si chaque session conserve un `GraphAwareCache` de 50 Mo et que tu as 10 utilisateurs simultanés, tu pèses 500 Mo rien que pour les caches.

Mitigations :

- `@st.cache_resource` pour les choses partageables (embedder, client Qdrant) — UNE instance pour tous.
- Garder `st.session_state` léger : pas de duplication d'objets gros.
- Pour de la prod multi-user (phase 5), envisager FastAPI + frontend séparé. Streamlit reste pour le dev.

## Le piège du rerun pendant le streaming

Si le script rerun pendant qu'une réponse LLM est en cours de streaming (parce que l'utilisateur a touché un slider), le streaming est interrompu. Solution : bloquer les widgets pendant le streaming, ou utiliser `st.status` avec un état "running" qui désactive temporairement les interactions.

Pour le MVP Streamlit on garde simple : pas de protection, on accepte que l'utilisateur ne touche pas l'UI pendant qu'il génère une réponse.
