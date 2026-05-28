# 3.3 — Visualisation interactive du graphe sémantique

## Pourquoi visualiser le graphe

Tous les concepts de la phase 2 (clusters Fiedler, Singular, Hinge, Theta) sont abstraits. Une visualisation transforme ces objets mathématiques en quelque chose de **directement intuitif** pour l'utilisateur :

- Voir les clusters comme des amas distincts dans l'espace.
- Voir les Hinges comme des nœuds qui font le pont entre amas.
- Voir les Singulars comme des points isolés en marge.
- Voir le poids des liens via l'épaisseur des arêtes.

C'est aussi un excellent outil de **debug** : si ton graphe ressemble à une boule indistincte, c'est que ton corpus est trop homogène. Si tu vois des composantes séparées, c'est que ton k est trop bas. La visualisation diagnostique d'un coup d'œil.

## Le problème du layout

Un graphe est une structure abstraite : nœuds + arêtes. Il n'a pas de position dans l'espace. Pour le dessiner, il faut un **algorithme de layout** qui assigne des coordonnées 2D à chaque nœud.

Quatre familles principales :

### 1. Spring (force-directed)

Modèle physique : les arêtes sont des ressorts, les nœuds des charges électriques qui se repoussent. On simule jusqu'à l'équilibre.

Avantages : résultats esthétiques, clusters naturellement séparés, pas de paramètres à régler.
Inconvénients : non-déterministe (chaque run donne un résultat différent), lent pour gros graphes (O(n²) par itération).

Algorithmes : Fruchterman-Reingold (le défaut NetworkX), Kamada-Kawai (énergie minimale), ForceAtlas2 (Gephi).

### 2. Spectral

Utilise les vecteurs propres du Laplacien (déjà calculés en 2.2 !) comme coordonnées.

```
position(i) = (v_2[i], v_3[i])
```

Avantages : déterministe, mathématiquement fondé, mappe directement la structure spectrale en 2D.
Inconvénients : peut donner des layouts "écrasés" si les premiers vecteurs propres ne discriminent pas bien.

Pour Eigenmind, c'est élégant — la visualisation **utilise exactement les mêmes maths** que l'analyse. On l'expose comme option.

### 3. Circular / hierarchical

Place les nœuds en cercle ou en arbre. Adapté aux graphes orientés ou aux hiérarchies, pas à un graphe k-NN.

### 4. UMAP / t-SNE sur embeddings

On ignore le graphe et on projette directement les embeddings 384-d en 2D via une réduction de dimension. Avantages : préserve la similarité sémantique, lisible. Inconvénients : ne montre pas les arêtes du graphe.

Approche hybride pertinente : positions UMAP + dessin des arêtes par-dessus. Eigenmind ne l'implémente pas en MVP mais c'est un upgrade naturel.

## Choix par défaut Eigenmind

On utilise **Kamada-Kawai** (un force-directed à énergie minimale). Pourquoi :

- Bonne séparation des clusters naturellement.
- Pas de paramètres à régler.
- Suffisamment rapide pour <500 nœuds.
- Disponible directement dans NetworkX : `nx.kamada_kawai_layout(G)`.

Pour des graphes plus gros, on basculera sur `nx.spring_layout(G, iterations=50)` (Fruchterman-Reingold).

## La coloration : encoder les types

Chaque nœud a une "catégorie" :
- **mainstream** (gris) : le chunk standard, masse du document.
- **Singular** (orange) : atypique, original.
- **Hinge** (rouge) : pivot inter-cluster.
- **Theta** (bleu) : représentant de sous-cluster.

Cette palette n'est pas arbitraire :
- Le gris pour la masse, peu saillant, ne distrait pas l'œil.
- L'orange et le rouge (chaudes) pour les "remarquables" — l'œil va dessus.
- Le bleu pour les Theta, distinct mais plus calme.

Si un nœud est dans plusieurs catégories (rare avec notre exclusion en cascade), on choisit la priorité **Hinge > Singular > Theta** (Hinge est le plus rare, donc le plus informatif).

## La taille : encoder l'importance

Deux conventions possibles :

- **Taille proportionnelle au degré** (combien de voisins) — montre les "hubs".
- **Taille proportionnelle au score spécifique** (Singular score, Hinge score) — montre l'intensité du rôle.

Eigenmind utilise la taille proportionnelle au **degré pondéré**, pour ne pas surcharger l'utilisateur avec plusieurs significations de la taille.

## Les arêtes

On dessine les arêtes avec **transparence proportionnelle au poids**. Une arête à 0.9 est opaque, une à 0.4 est translucide. Permet de voir les liens forts d'un coup d'œil sans masquer la structure par un fouillis d'arêtes faibles.

Pour les très gros graphes (>500 nœuds), on **n'affiche que les arêtes de poids > seuil** (par ex 0.6). Au-delà de quelques milliers d'arêtes, le rendu devient illisible et lent.

## Plotly vs Pyvis

Deux libs Python populaires pour les graphes interactifs.

**Plotly** :
- Stack standard Python data viz.
- Bonne intégration Streamlit (`st.plotly_chart`).
- Statique par défaut, interactif via hover/zoom/pan.
- Pas d'animation, pas de drag-and-drop.

**Pyvis** (basé sur vis.js) :
- Spécialisé graphes.
- Drag-and-drop, animation des forces, physique en temps réel.
- Mais intégration Streamlit pénible (génère du HTML, qu'il faut embed dans une iframe).
- Customisation des couleurs/tailles plus délicate.

Eigenmind utilise **Plotly** :
- Setup en 30 lignes vs 80 pour pyvis.
- Suffisant pour l'inspection visuelle (zoom + hover sur les chunks).
- Performant jusqu'à ~1000 nœuds.

Pour un Graph Explorer vraiment dynamique (drag, repositionnement manuel), pyvis serait meilleur. À considérer pour phase 4+.

## L'interaction au hover

Quand l'utilisateur survole un nœud, on affiche :
- Le filename + numéro de page.
- Un extrait du texte (premiers 200 caractères).
- Le type (Singular / Hinge / Theta / mainstream).
- Le degré, score singular/hinge si pertinent.

Plotly gère ça via `hovertext` ou `customdata` + `hovertemplate`.

## Sub-graph exploration (avancé)

Phase 4+, idée naturelle d'extension : permettre à l'utilisateur de cliquer sur un nœud et n'afficher que son **voisinage** (lui + ses k voisins + leurs voisins). Réduit la complexité visuelle pour explorer en profondeur.

NetworkX supporte ça avec `nx.ego_graph(G, node, radius=2)`.

## Layout caching

Le layout est coûteux à calculer (Kamada-Kawai sur 500 nœuds peut prendre 5-10s). On le cache **dans `st.session_state`** :

```python
if "graph_layout" not in st.session_state:
    st.session_state.graph_layout = nx.kamada_kawai_layout(G)
positions = st.session_state.graph_layout
```

Le layout est invalidé en même temps que `graph_cache` (à chaque ingestion).

## Le piège : layouts non-comparables entre runs

Les layouts force-directed dépendent d'une initialisation aléatoire. Deux exécutions donnent des positions différentes. Si l'utilisateur revient sur la page après une autre activité, le graphe est repositionné — déconcertant.

Mitigations :
- Fixer le `seed` du random state.
- Cacher le layout calculé.

On fait les deux dans le MVP.
