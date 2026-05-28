# 2.1 — Construction d'un graphe de similarité

## Pourquoi un graphe au-dessus des embeddings ?

Avec uniquement des embeddings et Qdrant, tu vois le corpus comme un **nuage de points** dans un espace 384-d. Tu peux trouver les voisins d'un point donné, mais tu ne vois pas la **structure globale** : quels groupes de chunks parlent du même sujet, quels chunks sont des ponts entre groupes, quels chunks sont isolés.

Un graphe explicite cette structure. Tu transformes le nuage de points en un **réseau** où :
- chaque chunk est un nœud,
- deux chunks sémantiquement proches sont reliés par une arête,
- le poids de l'arête est leur similarité cosinus.

Sur ce graphe tu peux appliquer toute la machinerie de la théorie des graphes : centralité, clustering, décomposition spectrale, parcours, communautés. C'est radicalement plus expressif qu'un nuage de points.

## Le compromis : graphe dense vs graphe sparse

Si tu connectes **tous les chunks à tous les autres**, ton graphe est complet : `n` nœuds, `n²` arêtes. Pour 1000 chunks, ça fait 1 million d'arêtes. La plupart sont des liens entre chunks très différents (similarité ~0.05), donc du bruit pur. Inutile en pratique.

Pour avoir un graphe **utile**, il faut **sparsifier** : ne garder que les arêtes significatives. Trois stratégies classiques :

### 1. Threshold global (le plus simple)

On ne garde que les arêtes avec `similarité > seuil`. Par exemple seuil = 0.5.

```
si cos(chunk_i, chunk_j) > 0.5 :
    ajouter une arête (i, j, poids = cos)
```

Avantages : simple, le poids reflète vraiment la similarité.
Inconvénients : choix arbitraire du seuil. Trop bas → bruit. Trop haut → graphe disconnecté. Et certains chunks "moyennement" similaires à tout le monde peuvent finir isolés sans aucune arête.

### 2. k-NN graph (notre choix)

Pour chaque chunk, on garde les **k voisins les plus similaires**, point. Pas de seuil global. Chaque nœud a forcément k arêtes sortantes.

```
pour chaque chunk_i :
    voisins = top-k chunks j != i par similarité
    pour chaque j dans voisins :
        ajouter une arête (i, j, poids = cos(i, j))
```

Avantages : tout le monde est connecté, garantie de structure. Reflète la **structure locale** du corpus.
Inconvénients : k-NN est **asymétrique** par construction (i peut avoir j comme voisin sans que l'inverse soit vrai). On symétrise ensuite.

### 3. Mutual k-NN graph

Variante stricte de k-NN : on garde une arête (i, j) uniquement si **les deux** sont voisins l'un de l'autre. Plus stringent, donne un graphe plus sparse et "épuré".

Utile pour faire ressortir les liens vraiment forts. Inconvénient : certains chunks peuvent se retrouver sans aucune arête.

## Choix de k

Combien de voisins par chunk ?

- **k trop petit** (2-3) : graphe fragmenté, plusieurs composantes connexes, perte d'info structurelle.
- **k trop grand** (50+) : graphe dense, le bruit revient.
- **Sweet spot** : k entre 5 et 15 pour la plupart des corpus.

Règle empirique : `k ≈ sqrt(n)` où `n` est le nombre de chunks. Pour 100 chunks, k=10. Pour 10 000 chunks, k=100. Mais en pratique on plafonne à ~15 même pour les très gros corpus.

Eigenmind utilise typiquement k=10. C'est ce qu'on met par défaut.

## Symétrie : graphe orienté ou non ?

Mathématiquement, la similarité cosinus est **symétrique** : `cos(a, b) = cos(b, a)`. Mais le k-NN est asymétrique : si j est dans le top-k de i, l'inverse n'est pas garanti.

Deux façons de symétriser :

**Union (OR)** — on garde une arête (i, j) si i est voisin de j **ou** j est voisin de i. Graphe plus dense, plus de connectivité.

**Intersection (AND)** — on garde l'arête uniquement si les deux sont voisins mutuellement (= mutual k-NN). Graphe plus sparse, liens plus stricts.

On utilise **Union** par défaut — on veut un graphe bien connecté pour la suite (Laplacien stable).

## Self-loops : à exclure

Un chunk est trivialement le voisin le plus proche de lui-même (`cos(i, i) = 1.0`). On l'exclut systématiquement : pas d'arête (i, i). Les self-loops polluent l'analyse spectrale (font monter artificiellement les degrés).

En pratique, quand on cherche les top-k voisins, on demande **k+1** et on exclut le chunk lui-même.

## Calcul efficace : matrice de similarité

Plutôt que de boucler en Python sur toutes les paires (lent), on calcule la **matrice de similarité** d'un coup avec numpy. Vu que nos embeddings sont L2-normalisés, c'est juste un produit matriciel :

```python
E = np.array(embeddings)         # shape (n, 384)
S = E @ E.T                      # shape (n, n), similarité cosinus
```

Pour 1000 chunks de dim 384 : 1000 × 1000 × 384 = ~400 millions de flops. Sur un Mac M3, ça prend ~50ms. Pour 10 000 chunks, ça monte à 4ms × 10 000 = ~5s, et la matrice fait 100M × 8 octets = 800 MB en RAM. Au-delà, on doit batcher ou utiliser des libs spécialisées.

Pour Eigenmind à l'échelle d'un utilisateur (quelques centaines à milliers de chunks), aucun souci.

## NetworkX : la lib graphe en Python

NetworkX est la lib standard pour manipuler des graphes en Python. Riche, lisible, mais pas la plus rapide — pour des graphes massifs (millions d'arêtes) on basculerait sur `graph-tool` ou `igraph`. À notre échelle, NetworkX est parfait.

Vocabulaire NetworkX :

- `nx.Graph()` — graphe non-orienté.
- `nx.DiGraph()` — graphe orienté.
- `G.add_node(id, **attrs)` — ajouter un nœud avec attributs libres.
- `G.add_edge(i, j, weight=w)` — ajouter une arête pondérée.
- `G.nodes[i]` — attributs d'un nœud.
- `G[i][j]['weight']` — poids d'une arête.
- `nx.adjacency_matrix(G)` — matrice d'adjacence sparse (pour la suite spectrale).

## Liens avec le retrieval

Le graphe n'est **pas** une alternative au retrieval Qdrant. C'est une **structure analytique** par-dessus.

Quand un utilisateur pose une question :
1. Qdrant fait le retrieval top-k vectoriel (rapide, étape obligatoire).
2. Le graphe enrichit ce top-k avec des chunks **Singular** (originaux), **Hinge** (pivots) ou via exploration locale (voisinage d'un chunk pertinent).

C'est ce qu'on fera en 2.6. Pour l'instant, on construit juste le graphe.

## Quand reconstruire le graphe ?

Le graphe dépend de tous les embeddings de la collection. Si tu ajoutes 1 nouveau chunk, ses k voisins peuvent changer **et** il peut devenir le voisin d'autres chunks (modifiant leur voisinage).

Trois stratégies :

**Recompute total** — à chaque ajout important, on reconstruit tout. Lent mais simple. Pour Eigenmind en MVP : on accepte ça, on recompute après chaque ingestion.

**Incremental update** — n'ajouter que les nouvelles arêtes liées au nouveau chunk, garder le reste intact. Plus complexe, parfois imprécis (un chunk existant peut maintenant avoir un meilleur voisin).

**Lazy / on-demand** — ne calculer le graphe que quand on en a besoin (à la requête), pas à l'ingestion. Trade-off latence vs fraîcheur.

On part sur **recompute total**, suffisant pour la phase 2.

## Caching et persistence

Le graphe doit être recalculé après chaque modif du corpus mais pas à chaque requête. Stratégies de cache :

- Sauver le graphe sur disque via `nx.write_gpickle(G, "graph.pkl")` puis `G = nx.read_gpickle("graph.pkl")`.
- Recompute uniquement si la dernière modif Qdrant est plus récente que le snapshot du graphe.

En MVP on garde simple : on recompute à chaque session du script. C'est rapide.