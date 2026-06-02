# 2.2 — Analyse spectrale d'un graphe

## L'intuition fondamentale

Un graphe peut être vu comme une **matrice**. Et toute matrice symétrique a une **décomposition propre** : on peut l'écrire comme un mélange de modes fondamentaux, chacun associé à une valeur propre (une "fréquence") et un vecteur propre (une "forme").

C'est exactement comme une corde de guitare : sa vibration physique se décompose en un mode fondamental + des harmoniques. Pour un graphe, c'est la même idée — mais au lieu de modes vibratoires, on a des modes de **regroupement** : les premiers vecteurs propres révèlent les communautés, les suivants raffinent la structure.

Cette décomposition s'appelle l'**analyse spectrale** d'un graphe, et elle est étonnamment puissante : à partir d'un simple graphe pondéré, elle révèle automatiquement clusters, ponts, chunks atypiques. Tout Eigenmind repose là-dessus.

## Les trois matrices clés

### La matrice d'adjacence A

Si on a `n` nœuds, A est une matrice n × n où :

```
A[i, j] = poids de l'arête (i, j) si elle existe, 0 sinon
```

Pour notre graphe k-NN avec poids = similarité cosinus, A est symétrique (graphe non-orienté) et toutes ses valeurs sont entre 0 et 1.

### La matrice de degrés D

Matrice diagonale n × n où :

```
D[i, i] = somme des poids des arêtes incidentes au nœud i
        = "degré pondéré" du nœud i
```

Tous les autres éléments de D sont 0. D mesure à quel point chaque nœud est connecté.

### Le Laplacien L = D − A

**LA** matrice centrale. Une combinaison astucieuse :

- Les éléments diagonaux `L[i, i] = D[i, i]` mesurent la connexion locale.
- Les éléments hors-diagonale `L[i, j] = −A[i, j]` retirent les similarités.

Pourquoi cette définition ? Parce que L a une propriété magique : pour tout vecteur x,

```
x^T L x = Σ_{(i,j) ∈ E} w_{ij} × (x_i − x_j)²
```

Cette quantité mesure **à quel point x est lisse sur le graphe** — petite quand les valeurs voisines sont proches, grande quand les valeurs voisines diffèrent. Le Laplacien encode donc la **régularité spatiale** sur le graphe.

C'est la même idée que le Laplacien continu en physique : il mesure la "dispersion" d'une fonction. Sur un graphe c'est discret, mais l'intuition est identique.

### Le Laplacien normalisé

Variante très utilisée :

```
L_norm = I − D^(−1/2) A D^(−1/2)
```

Avantages :
- Toutes les valeurs propres sont dans [0, 2].
- Les nœuds très connectés n'écrasent pas l'analyse.
- Plus stable numériquement quand les degrés varient beaucoup.

On utilisera `L_norm` par défaut dans Eigenmind. C'est ce que `nx.normalized_laplacian_matrix` calcule.

## Valeurs propres et vecteurs propres

Pour une matrice symétrique L, une **paire propre** `(λ, v)` satisfait :

```
L v = λ v
```

Autrement dit : appliquer L à v ne change pas sa direction, juste sa longueur (multipliée par λ).

Une matrice n × n a **n** paires propres. Pour L (Laplacien d'un graphe connexe), elles sont toutes :

- **Réelles** (parce que L est symétrique).
- **Non-négatives** : λ ≥ 0 pour tout λ.
- **Orthogonales** : v_i · v_j = 0 quand i ≠ j.

On les trie classiquement par ordre croissant :

```
0 = λ_1 ≤ λ_2 ≤ λ_3 ≤ ... ≤ λ_n
```

Chaque vecteur propre v_i a n composantes — une par nœud du graphe.

## Ce que disent les premières valeurs propres

### λ_1 = 0 (toujours)

Pour tout Laplacien, la plus petite valeur propre est exactement 0. Son vecteur propre est constant (tous les nœuds ont la même valeur). C'est trivial et sans information.

**Sauf si le graphe a plusieurs composantes connexes** : alors λ_1 = 0 a une multiplicité égale au nombre de composantes. C'est un test diagnostique automatique : si λ_2 ≈ 0, ton graphe est presque déconnecté.

### λ_2 — l'**eigenvalue de Fiedler**, la quantité la plus importante

Aussi appelée **connectivité algébrique** du graphe.

- λ_2 grand → graphe bien connecté, difficile à séparer en deux.
- λ_2 petit → graphe avec un **bottleneck**, facilement séparable en deux clusters.
- λ_2 = 0 → graphe déconnecté.

Pour Eigenmind, λ_2 te dit si ton corpus est homogène (λ_2 élevé) ou s'il a des thématiques très distinctes (λ_2 bas).

### Le vecteur de Fiedler v_2 — la magie

Le vecteur propre associé à λ_2 a une propriété spectaculaire : ses composantes positives et négatives définissent **le meilleur découpage en deux clusters** du graphe.

```
nœud i va dans cluster A  si v_2[i] > 0
nœud i va dans cluster B  si v_2[i] < 0
```

C'est ce qu'on appelle le **spectral clustering bipartite**. Sans aucune supervision, sans k-means, sans seuil arbitraire, le simple signe du Fiedler vector révèle la coupure thématique la plus "naturelle" du corpus.

**Plus une composante v_2[i] est grande en valeur absolue, plus le nœud i est "central" dans son cluster.** Les deux **extrema** de v_2 (le max et le min) sont les **antipodes spectraux** — les deux nœuds les plus séparés le long de l'axe thématique dominant. Ce sont, dans le mémo officiel, les **Singular nodes** (pôles thématiques, cf. 2.3).

> ⚠️ Attention : dans une version antérieure de ces docs, on associait la **frontière** de v_2 (composantes ≈ 0) aux Hinge nodes. Le mémo officiel définit en réalité les Hinge par un **champ géodésique** (cf. 2.4), pas par le vecteur de Fiedler. Le Fiedler sert surtout aux **Singular** (ses extrema).

### L'inégalité de Cheeger — le théorème qui ancre λ_2

λ_2 n'est pas qu'un diagnostic intuitif : il est **formellement** relié à la "coupabilité" du graphe par l'**inégalité de Cheeger** (Chung, *Spectral Graph Theory*). En notant `h(G)` la constante isopérimétrique (conductance) du graphe :

```
λ_2 / 2  ≤  h(G)  ≤  √(2 λ_2) ,    h(G) = min_S  cut(S, S̄) / min(vol(S), vol(S̄))
```

- Une **petite** λ_2 garantit l'existence d'une coupe peu coûteuse → séparation thématique métastable.
- C'est un vrai **théorème** (avec preuve), à distinguer des interprétations sémantiques heuristiques (cf. 2.7).
- **Subtilité** : une petite λ_2 ne signifie **pas** une structure bipartie. La bipartité se lit à l'autre bout du spectre du Laplacien **normalisé** : des valeurs propres proches de **2**.

### λ_3, λ_4, ... — raffinements successifs

Chaque vecteur propre suivant subdivise les clusters précédents. Avec les k premiers vecteurs propres, on peut faire du **spectral clustering à k clusters** :

1. Calculer les k premiers vecteurs propres v_1, ..., v_k.
2. Représenter chaque nœud i par son "embedding spectral" : (v_1[i], v_2[i], ..., v_k[i]).
3. Appliquer k-means sur ces embeddings.

C'est l'algorithme de **Ng-Jordan-Weiss** (2001), l'un des spectral clustering les plus utilisés.

### Les dernières valeurs propres — le **bruit**

Les modes propres de haute fréquence (λ_n, λ_{n−1}, ...) correspondent à des oscillations très rapides sur le graphe. Ils captent les détails locaux et les anomalies.

> ⚠️ Une version antérieure de ces docs en faisait la base des **Singular nodes**. C'est une **erreur** corrigée : le mémo officiel définit les Singular par les modes **basses** fréquences (les extrema du vecteur de Fiedler et des modes suivants), **pas** par les hautes fréquences. Les hautes fréquences restent surtout du détail/bruit (et la bande proche de λ=2 signale la bipartité).

## Décodage : que représentent les vecteurs propres ?

Visualise : un vecteur propre v_i associe une valeur à chaque nœud.

- v_1 (associé à λ_1 = 0) : tous les nœuds reçoivent la même valeur. Pas d'info.
- v_2 (Fiedler) : valeurs positives d'un côté du graphe, négatives de l'autre. Coupure binaire.
- v_3 : raffine en trois zones, etc.

On peut **colorier** les nœuds du graphe selon v_2 : les nœuds très positifs en rouge, très négatifs en bleu, proches de zéro en jaune. La visualisation révèle instantanément la structure du corpus.

C'est exactement ce qu'on fera dans le Graph Explorer en phase 3.

## Lien avec la régularité

Souviens-toi : x^T L x = Σ w_{ij} (x_i − x_j)².

Quand x est un vecteur propre v_i :

```
v_i^T L v_i = λ_i
```

(parce que L v_i = λ_i v_i, donc v_i^T L v_i = λ_i v_i^T v_i = λ_i × 1 = λ_i).

Donc **λ_i mesure la "non-régularité" de v_i sur le graphe** :

- λ petit → v change lentement entre nœuds voisins → v "respecte" la structure du graphe → bonne fonction de regroupement.
- λ grand → v oscille rapidement → v capture des détails locaux.

C'est l'analogue exact de l'analyse de Fourier en signal : les basses fréquences captent la structure globale, les hautes fréquences captent le détail. **Le Laplacien définit la "transformée de Fourier sur graphe".**

## Coût computationnel

La décomposition propre complète de L coûte **O(n³)** opérations et O(n²) en mémoire.

- n = 100 chunks : instantané (~ms).
- n = 1000 chunks : ~1s, ~8 Mo RAM.
- n = 10 000 chunks : ~15 min, 800 Mo RAM. Limite raisonnable.
- n > 50 000 : il faut basculer sur des solveurs propres **partiels** (Lanczos, ARPACK) qui ne calculent que les k premiers vecteurs propres en O(n × k × itérations). Disponibles via `scipy.sparse.linalg.eigsh`.

Pour Eigenmind à l'échelle d'un utilisateur, la décomposition complète est OK. On y reviendra en optimisation phase 5 si nécessaire.

## Numpy / scipy : la pratique

Deux fonctions clés :

```python
import numpy as np
from scipy.sparse.linalg import eigsh

# Décomposition complète d'une matrice symétrique dense
eigenvalues, eigenvectors = np.linalg.eigh(L)
# eigenvalues : (n,) triées par ordre croissant
# eigenvectors : (n, n), eigenvectors[:, i] est le i-ème vecteur propre

# Décomposition partielle d'une matrice sparse (pour gros graphes)
eigenvalues_k, eigenvectors_k = eigsh(L_sparse, k=20, which='SM')
# k premiers vecteurs propres (SM = "smallest magnitude")
```

`eigh` (avec h pour Hermitien/symétrique) est crucial. **Ne pas utiliser `eig`** sur un Laplacien — `eig` est pour les matrices générales, lent et numériquement instable sur les matrices symétriques. `eigh` exploite la symétrie : 2× plus rapide et garantit des résultats réels.

## Le signe des vecteurs propres

Subtilité numérique : un vecteur propre est défini **à un signe près**. Si v est vecteur propre, −v l'est aussi (même λ). Les implémentations numpy/scipy choisissent arbitrairement un signe, qui peut **changer d'une exécution à l'autre** ou d'une machine à l'autre.

Conséquence pratique : si tu écris "cluster A = nœuds avec v_2 > 0", la définition de A et B peut s'inverser entre deux runs. Pas grave pour la coupure (elle reste la même), mais à savoir pour les tests.

Mitigation : on peut "normaliser le signe" en forçant `v[0] > 0`, par convention.

## Le piège du graphe non-connexe

Si le graphe a plusieurs composantes, on a λ_2 = 0 (ou très proche), et le Fiedler vector ne sépare plus deux clusters mais **isole une composante** du reste. La maths reste valide mais l'interprétation change.

Test à faire : avant la décomposition, vérifier `nx.is_connected(G)`. Si False, soit on traite chaque composante séparément, soit on rend le graphe connexe (ajouter quelques arêtes faibles).

Comme notre k-NN graph avec k=10 produit en général un graphe connexe, on assume cette condition.

## Récapitulatif pour la suite

Les éléments qu'on va exploiter dans les étapes 2.3 à 2.6 :

| Élément | Ce qu'il code | Utilisation Eigenmind (mémo officiel) |
|---|---|---|
| λ_2 (Fiedler) + Cheeger | Connectivité algébrique, conductance | Diagnostic du corpus ; ancre les axes thématiques |
| **Extrema** des modes basses fréquences (v_2, v_3, …) | Antipodes spectraux | Identifier les **Singular nodes** = pôles thématiques (2.3) |
| Champ géodésique sur −log W (pas le spectre) | Relief de log-similarité | Identifier les **Hinge nodes** = connecteurs (2.4) |
| Relaxation SDP de Lovász-θ sur H = 1[W≥τ] | Code non-confusable | Identifier les **Theta nodes** = info unique (2.5) |
| Valeurs propres proches de 2 | Bipartité | Diagnostic structurel |

⚠️ **Important** : seuls les **Singular** dérivent directement de la décomposition spectrale (modes basses fréquences). Les **Hinge** reposent sur des **géodésiques** (2.4) et les **Theta** sur une **relaxation SDP** (2.5) — pas sur les modes propres. Le mémo (`260522_Eigenmind_Cognitive_Maps.pdf`) fait foi.