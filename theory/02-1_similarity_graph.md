# 2.1 — Construction du graphe de similarité

> **Référence faisant foi.** Ce chapitre suit le mémo officiel Merlin Intelligence
> (`theory/260522_Eigenmind_Cognitive_Maps.pdf`, §3.1 et §4.1). Là où notre MVP avait pris
> une route différente, c'est **la définition du mémo qui prévaut**. Une note d'implémentation
> en fin de doc signale ce que `build_graph.py` fait encore autrement.

## Pourquoi un graphe au-dessus des embeddings ?

Avec uniquement des embeddings et Qdrant, on voit le corpus comme un **nuage de points** dans
un espace 384-d. On peut trouver les voisins d'un point, mais pas la **structure globale** :
quels groupes de chunks parlent du même sujet, lesquels font le pont entre groupes, lesquels
sont isolés.

Le graphe explicite cette structure : chaque chunk est un nœud, deux chunks sémantiquement
proches sont reliés, le poids de l'arête est leur similarité cosinus. Sur ce graphe on applique
toute la machinerie de la théorie des graphes : Laplacien, géodésiques, relaxations SDP.

## Deux régimes de similarité (le point que notre MVP avait raté)

Le mémo distingue **deux usages** de la similarité, qui ne sont pas redondants (§3.2.2) :

- **Retrieval top-k** (`store.similarity_search`, utilisé dans `pipelines/rag.py`) : un seul
  appel, `k` résultats, **pas de structure de graphe**. C'est le RAG dense classique. Il optimise
  la *pertinence au prompt*.
- **Exploration BFS** (`exploration.py`) : une **séquence** de lookups top-k où chaque requête
  est un *ID de point déjà stocké* plutôt qu'un nouvel embedding, accumulant un **sous-graphe de
  travail**. Il optimise la *couverture du voisinage du prompt*.

> Question d'analyste : *« Est-ce que je veux ce que le prompt demande, ou à quoi ressemble le
> monde autour du prompt ? »*

**Les trois stratégies de la phase 2 (Singular, Hinge, Theta) opèrent sur le second régime** :
elles analysent le sous-graphe BFS, pas la collection entière.

## Étape 0 — le sous-graphe par BFS sur l'index ANN (`exploration.py`)

Le « graphe » parcouru est l'**index approximate-nearest-neighbour (ANN) de Qdrant** : ses arêtes
relient chaque vecteur stocké à ses `k` plus proches voisins en espace d'embedding. Le BFS suit
ces liens ANN vers l'extérieur depuis le prompt.

1. **Seed.** Encoder le prompt `q ↦ φ(q) ∈ ℝ^d`. Récupérer les `K_0` vecteurs les plus proches
   (`K_0 = NEIGHBORS_TO_FETCH` dans `config.py`) ; initialiser la file BFS `Q` et l'ensemble
   visité `V`.
2. **Expansion BFS.** Tant que `|V| < N_max` (`N_max = MAX_CHUNKS_FOR_CONTEXT`) et `Q ≠ ∅` :
   défiler `v`, récupérer ses `K_0` plus proches voisins hors `V`, enfiler les nouveaux IDs.
3. **Récupération groupée.** Fetch tous les vecteurs `{φ_i}_{i∈V}` et payloads en un seul appel.
4. **Matrice de similarité** (étape 3.1 ci-dessous).

> **Caveat ANN (§4.1.1).** Le sous-graphe induit porte trois limites structurelles :
> *approximation* (HNSW ne renvoie pas les voisins exacts), *asymétrie* (le graphe ANN est
> orienté ; `u` peut être dans les `k` voisins de `v` sans réciproque — la matrice cosinus
> `W = ΦΦᵀ` restaure ensuite la symétrie), et *dépendance à l'encodeur* (la connectivité
> sémantique ne vaut que ce que vaut l'embedding).

## Étape 1 — la matrice de similarité cosinus (`singular.py`, §3.1)

### Matrice d'embedding et matrice de Gram

On empile les `n` embeddings du sous-graphe dans `Φ ∈ ℝ^{n×d}` (ligne `i` = `φ_iᵀ`). Chaque `φ_i`
est (approximativement) ℓ₂-unitaire. La matrice de Gram brute est :

```
W_raw = Φ Φᵀ ∈ ℝ^{n×n} ,    W_raw[i,j] = φ_iᵀ φ_j ≈ cos∠(φ_i, φ_j)
```

Symétrique, semi-définie positive, de rang ≤ `d`. Ses valeurs propres non nulles sont exactement
les **valeurs singulières au carré** de `Φ` (d'où le nom du module `singular.py`).

### Sparsification par **seuil** (et non par k-NN)

C'est ici que notre MVP divergeait. Le mémo **ne fait pas de k-NN** : il applique un **seuil de
similarité unique** `τ = SIMILARITY_THRESHOLD` (défaut **0.65**) :

```
W[i,j] = W_raw[i,j]   si W_raw[i,j] ≥ τ  et  i ≠ j
         0            sinon
```

La diagonale est forcée à zéro : `W_ii = 0`.

### Pourquoi zéro sur la diagonale ?

Les méthodes spectrales requièrent une **adjacence**, pas un noyau. Avec `W_ii = ‖φ_i‖² = 1` sur
la diagonale, le Laplacien `L = D − W` serait biaisé par les boucles (*self-loops*), et l'espace
nul du Laplacien normalisé n'encoderait plus proprement les composantes connexes. On enlève donc
la diagonale pour traiter `W` comme l'adjacence d'un graphe pondéré sans boucle.

### Ce que ça produit

Une matrice symétrique `W ∈ [0,1]^{n×n}` avec `W_ii = 0`, dont la densité d'arêtes est pilotée
par `τ`. **`W` est l'unique objet mathématique** consommé par les trois stratégies de §4 :
spectrale (Singular), géodésique (Hinge) et theta (Frontier).

## `τ` : le bouton de calibration (et la « ligne éditoriale »)

Le scalaire `τ` gouverne à lui seul (§3.1 *Mathematical Grounding*) :

1. l'ensemble d'arêtes du graphe induit par le BFS ;
2. la **connectivité** de `W` ;
3. le graphe d'interdépendance `H = 1[W ≥ τ]` qui alimente la relaxation theta (cf. `02-5`).

- `τ` **bas** → graphe dense, spectre plus lisse, liaisons thématiques permissives (bon pour
  l'exploration, dangereux pour l'inférence).
- `τ` **haut** → graphe fragmenté, multiplicité de `λ = 0` qui gonfle, notion de parenté stricte
  (bon pour l'analyse contrastive, mais risque de fragmentation).

Interprétation sémantique (§3.1) : chaque entrée `W_ij` dit *« les chunks i et j parlent de la
même chose, avec confiance W_ij »*. Le seuil `τ` est la **ligne éditoriale** du système :
*« au-dessus de quel niveau de similarité j'accepte que deux passages parlent du même sujet ? »*

## NetworkX : la lib graphe en Python

NetworkX manipule des graphes en Python. À l'échelle d'un sous-graphe analyste (quelques
centaines de nœuds), elle suffit largement.

```python
import networkx as nx
G = nx.from_numpy_array(W)              # graphe pondéré non-orienté depuis W
A = nx.to_numpy_array(G)                # retour matrice si besoin
```

## Quand reconstruire ?

Le sous-graphe est **local au prompt** : il est reconstruit à chaque requête d'exploration via le
BFS. Il n'y a donc pas de « graphe global du corpus » à maintenir — c'est une différence
importante avec une approche k-NN globale.

## ⚠️ Note d'implémentation MVP

`build_graph.py` (notre code) construit pour l'instant un **k-NN global (k=10) sur tous les
chunks de la collection**, symétrisé par union, et le met en cache (`GraphAwareCache`). C'est une
simplification pédagogique qui **diverge du mémo** sur deux points :

| | Notre `build_graph.py` | Mémo officiel |
|---|---|---|
| Périmètre | corpus entier | sous-graphe **local** par BFS (`exploration.py`) |
| Sparsification | k-NN (k=10) + union | **seuil** `τ = 0.65` sur `W = ΦΦᵀ` |

Les deux produisent un `W` symétrique exploitable, mais le mémo fait foi : à l'adoption du repo
complet (phase 4), on passera au sous-graphe BFS + seuillage. La suite de la phase 2 (`02-2` à
`02-6`) décrit les définitions **du mémo**.
