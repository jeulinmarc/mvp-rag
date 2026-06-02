# 2.4 — Hinge nodes : les connecteurs (champ géodésique)

> **Référence faisant foi.** Mémo officiel §4.3 (*Strategy 2 — Hinge Connectors*, module
> `connectivity.py`). ⚠️ **Cette définition remplace la version précédente**, qui identifiait les
> Hinge par la **betweenness centrality** + la frontière de Fiedler. Le mémo fait foi : les Hinge
> sont définis par un **champ géodésique** construit sur des distances de log-similarité. Note
> d'implémentation en fin de doc.

## Qu'est-ce qu'un Hinge node ?

Un Hinge node est un chunk qui **fait le pont entre les deux extrêmes thématiques** du corpus. Il
est à mi-chemin, connecté aux deux pôles, sans appartenir franchement à aucun.

Question d'analyste : *« Quels concepts relient les deux extrêmes thématiques ? »*

Exemples (§4.3) :

- Dans un corpus compétition US–Chine, les Hinge pourraient être des passages sur les **chaînes
  d'approvisionnement de semi-conducteurs** ou la **logistique des terres rares** — présents à la
  fois dans les arguments de confinement et d'interdépendance.
- Dans un corpus de politique monétaire, un Hinge pourrait discuter le **mou du marché du
  travail** — pertinent pour les *hawks* comme pour les *doves*.

## La construction : un champ géodésique sur le graphe de similarité

L'idée : transformer les similarités en **distances**, calculer le « relief » du graphe vu depuis
son point le plus périphérique, puis repérer les nœuds situés à **mi-pente entre les deux pôles
de ce relief**.

### 1. Transformation en longueurs (log-similarité)

```
ℓ_ij = − log W_ij ∈ [0, +∞)        ℓ_ij = +∞ si W_ij = 0
```

Plus deux chunks sont similaires (`W_ij → 1`), plus la longueur d'arête est courte (`ℓ_ij → 0`).

### 2. Distances géodésiques toutes paires

```
d(u, v) = min_{γ : u → v} Σ_{(i,j) ∈ γ} ℓ_ij
```

C'est le plus court chemin pondéré (Dijkstra toutes paires). Les paires déconnectées reçoivent une
pénalité `10 · max_finite d`. Complexité `O(n² log n + nm)`, praticable `n ≲ 500`.

### 3. Source périphérique et champ géodésique

On prend comme source le nœud le **plus périphérique** (celui dont la somme des distances aux
autres est maximale), puis on en mesure le champ de distance, qu'on **centre et normalise** :

```
v* = argmax_v Σ_j d(v, j)                          # source périphérique
x_i^raw = d(v*, i)                                  # champ brut
x̄ = (Σ_i d_i x_i^raw) / (Σ_i d_i)                  # moyenne pondérée par les degrés
x_i = (x_i^raw − x̄) / max_i |x_i^raw − x̄| ∈ [−1, 1]   # champ normalisé
```

`x_i` situe chaque nœud sur un axe `[−1, +1]` : `−1` côté source périphérique, `+1` à l'opposé,
`0` au milieu.

### 4. Définition des pôles

```
P₊ = { i : x_i ≥ q_{0.9} }        P₋ = { i : x_i ≤ q_{0.1} }      (fallback 80/20)
```

(`q_{0.9}`, `q_{0.1}` = quantiles du champ `x`.)

### 5. Score Hinge

```
S_±(i) = Σ_{j ∈ P_±} W_ij              # affinité de i vers chaque pôle
B(i)   = min(S₊(i), S₋(i))             # affinité duale équilibrée
H(i)   = B(i) · (1 − |x_i|)            # score Hinge final
```

Les Hinge sont les nœuds de **score `H` maximal** : forte affinité **vers les deux** pôles
(`B(i)` grand) **et** position centrale dans le champ (`1 − |x_i|` grand, donc `x_i ≈ 0`).

## Fondement mathématique

### Motivation formelle : géodésiques de log-similarité

Un chemin de longueur `Σ (− log W_ij)` est minimisé exactement quand il **maximise `∏ W_ij`** —
c'est-à-dire la **route sémantique la plus cohérente** entre deux chunks. Les géodésiques suivent
donc les chemins de plus haute affinité.

### Intuition mathématique : champ géodésique comme proxy ℓ∞ (honnête)

L'analogue `ℓ∞` du problème de Fiedler chercherait la fonction la plus lisse saturant l'hypercube
`‖x‖∞ = 1` avec contraintes de pôles (`x_{i⁺} = +1`, `x_{i⁻} = −1`). Mais le problème non
contraint `argmin_{‖x‖∞ ≤ 1, x ⊥_w 1} xᵀ L_sym x` a pour minimiseur trivial `x = 0` ; une solution
non triviale exigerait une égalité de norme saturante ou des conditions de bord explicites.

**Le champ géodésique normalisé `x ∈ [−1,1]^n` doit donc être compris comme un *proxy
heuristique*** : il est sur la sphère `ℓ∞` par construction et lisse au sens géodésique, mais ce
n'est **pas** la solution d'un problème variationnel ℓ∞ bien posé. Les Hinge sont les nœuds proches
du **niveau zéro** de ce champ.

### Fonctionnelle Hinge

`B(i) = min(S₊, S₋)` récompense l'**affinité duale équilibrée** ; le facteur `(1 − |x_i|)`
pénalise les nœuds trop proches d'un extrême. Un vrai connecteur est central **et** également
attiré par les deux pôles.

### Limites connues

Sensibilité au seuil de sparsification `τ`, **hubness sémantique** (les nœuds de haut degré
dominent les chemins géodésiques et biaisent le champ), et **anisotropie de l'embedding**.

## Interprétation sémantique

> *(Heuristique)* Les connecteurs sont des **ponts conceptuels plausibles.**

Un Hinge est le concept que les deux camps d'un débat invoquent tous les deux. Dans un dossier
contradictoire, ce sont les passages charnières où les deux thèses se rencontrent — exactement ce
qu'il faut injecter dans le contexte du LLM pour une réponse **inter-thématique** cohérente, plutôt
qu'un top-k enfermé dans un seul cluster.

## ⚠️ Note d'implémentation MVP

Notre `hinge.py` combine **betweenness centrality (0.5) + frontière de Fiedler `1−|v_2|` (0.3) +
diversité des voisins (0.2)**, plus un bypass sur les *articulation points*. C'est une approche
classique de théorie des graphes, mais **différente du mémo** :

| | Notre `hinge.py` | Mémo officiel |
|---|---|---|
| Objet central | betweenness + Fiedler `v_2` | **champ géodésique** sur `ℓ_ij = −log W_ij` |
| Pôles | signe de `v_2` | quantiles `q_{0.9}`/`q_{0.1}` du champ `x` |
| Score | somme pondérée de 3 critères | `H(i) = B(i)·(1 − \|x_i\|)`, `B = min(S₊,S₋)` |

Les deux capturent l'idée « nœud-pont », mais **le mémo fait foi**. À l'adoption du repo complet,
`hinge.py` devra implémenter le champ géodésique (Dijkstra toutes paires sur `−log W`, source
périphérique, score `H`).
