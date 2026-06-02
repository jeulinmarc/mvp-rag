# 2.5 — Theta nodes : les thèmes-frontière (relaxation de Lovász-θ)

> **Référence faisant foi.** Mémo officiel §4.4 (*Strategy 3 — Frontier Themes*, module
> `theta.py`). ⚠️ **Cette définition remplace la version précédente**, qui présentait les Theta
> comme une « appellation maison » basée sur les **modes propres intermédiaires**. C'était faux :
> le « θ » de Theta renvoie au **nombre de Lovász `ϑ(H)`**, et la stratégie est une **relaxation
> SDP** de ce nombre. C'est l'écart le plus important entre notre MVP et le mémo. Note
> d'implémentation en fin de doc.

## Qu'est-ce qu'un Theta node ?

Un Theta node est un **thème-frontière** : un chunk qui porte de l'information **unique,
non-redondante** par rapport au reste du corpus. Là où Singular donne les axes et Hinge les ponts,
Theta révèle les **signaux faibles et angles morts** — ce dont parle un seul passage, sans écho.

Question d'analyste : *« Quels chunks portent une information unique, non-redondante ? »*

Exemples (§4.4) : dans un corpus Moyen-Orient, des passages sur la **diplomatie de l'eau** ou les
**implications du shipping arctique** — présents mais non amplifiés par le discours dominant. En
macro globale : la **rareté du collatéral** ou la **réallocation des fonds de pension**.

## L'idée centrale : indépendance = non-confusabilité

Deux chunks directement reliés dans `W` sont **informationnellement redondants**. Trouver un
ensemble de chunks porteurs d'info unique revient à chercher un **grand ensemble indépendant**
dans un graphe d'interdépendance — un problème NP-difficile qu'on **relâche** via le nombre de
Lovász `ϑ`.

### 1. Graphe d'interdépendance

```
H_ij = 1[W_ij ≥ τ]        H_ii = 0
```

On réutilise **le même seuil `τ`** que le graphe de similarité (cf. `02-1`). Une arête de `H`
signifie : « ces deux chunks sont si proches qu'ils sont mutuellement redondants. »

### 2. Le nombre de Lovász `ϑ(H)` et le sandwich

Le **problème primal** est le nombre d'indépendance :

```
α(H) = max_{x ∈ {0,1}^n} { 1ᵀx  :  x_i x_j = 0  ∀ (i,j) ∈ E(H) }
```

Le **nombre de Lovász** `ϑ(H)` est une relaxation SDP qui l'encadre par le **théorème sandwich de
Lovász** :

```
α(H)  ≤  ϑ(H)  ≤  χ(H̄)
```

(`H̄` = graphe complémentaire, `χ` = nombre chromatique). `ϑ(H)` est calculable en temps polynomial
(SDP), contrairement à `α` et `χ`.

### 3. La relaxation lagrangienne (Lemaréchal–Oustry)

Le mémo implémente une **relaxation lagrangienne / SDP** du primal (référence Lemaréchal–Oustry,
2001). Le lagrangien introduit des multiplicateurs `λ` (contraintes binaires `x_i − x_i² = 0`) et
`μ` (contraintes d'indépendance `x_i x_j = 0`) :

```
L(x, λ, μ) = 1ᵀx + Σ_i λ_i (x_i − x_i²) − ½ Σ_{(i,j) ∈ E(H)} μ_ij x_i x_j
           = (1 + λ)ᵀx − xᵀ A(λ, μ) x
```

La fonction duale `g(λ, μ) = ¼ (1+λ)ᵀ A⁻¹ (1+λ)` est **convexe** ; sa minimisation donne
`min g = ϑ(H) ≥ α(H)`.

### 4. Le schéma sous-gradient

Comme la SDP exacte est coûteuse, on minimise `g` par **sous-gradient projeté** :

```
A^(k) = diag(λ^(k)) + ¼ (μ^(k) + μ^(k)ᵀ) ⊙ H
x^(k) = ½ (A^(k))⁻¹ (1 + λ^(k))

∇_λ g = x^(k) − (x^(k))°²                    # descente libre sur λ
∇_μ g = −½ x^(k) (x^(k))ᵀ ⊙ H               # descente projetée (μ ≥ 0) sur μ
```

(`⊙` = produit de Hadamard, `°²` = carré élément par élément.) Le pas `t_k = t_0 / √k` est la
**règle de pas décroissant** de Polyak. On retient la meilleure borne duale `θ*`.

> **Statut honnête (§4.4).** Le schéma sous-gradient **n'offre aucun certificat de convergence en
> nombre fini d'itérations**. Les scores produits sont des **proxies `ϑ`-inspirés**, pas des
> optima SDP exacts. Une validation contre un solveur exact sur petits corpus est recommandée.

### 5. Matrice frontière et scores

À partir de l'optimum approché `A*` :

```
F = (A*)⁻¹ / tr[(A*)⁻¹]  ⪰ 0           F ≈ Y Yᵀ
FS(i) = F_ii = ‖y_i‖²                   # score frontière du chunk i
```

`F_ij ≈ 0` pour les paires interdépendantes ; `FS(i) = ‖y_i‖²` mesure le **proxy d'information
indépendante** du chunk `i`. Les Theta nodes sont ceux de **score frontière élevé**.

### 6. Sélection par point le plus éloigné (farthest-point)

Pour garantir la **diversité** des Theta retenus :

```
ŷ_i = y_i / ‖y_i‖                                  # normalisation
seed = chunk de plus fort SVD leverage
i^(t+1) = argmax_i  min_{j ∈ S^(t)} (1 − ŷ_iᵀ ŷ_j)   # le plus loin du déjà-choisi
```

On ajoute itérativement le chunk le plus éloigné (au sens cosinus) de l'ensemble déjà sélectionné.

## L'analogie de Shannon (capacité de canal)

Lovász a introduit `ϑ` pour borner la **capacité d'erreur-zéro** d'un canal `C(G) = sup_k
ϑ(G^⊠k)^{1/k}` (produit fort `⊠`). Sous cette analogie : les chunks interdépendants (`H_ij = 1`)
sont des **« symboles confusables »**, et les thèmes-frontière correspondent à un **code
non-confusable de capacité maximale** du corpus — l'ensemble de chunks qu'on peut « émettre » sans
ambiguïté sémantique mutuelle. (Analogie, non équivalence formelle : requiert que `τ` modélise
correctement la structure du canal.)

## Interprétation sémantique

> *(Heuristique)* Les thèmes-frontière sont des **signaux faibles et angles morts plausibles.**

Concrètement, quand l'utilisateur pose une question exploratoire (« qu'est-ce que je rate ? »),
injecter des Theta dans le contexte force le LLM à couvrir l'information unique du corpus, pas
seulement le discours dominant.

## ⚠️ Note d'implémentation MVP

Notre `theta.py` calcule `theta_score(i) = max_{k=3..K} |v_k[i]|` sur les **modes propres
intermédiaires** du Laplacien, avec sélection par extrêmes de mode. Notre doc précédent admettait
même qu'il s'agissait d'une « appellation maison sans définition standard » — **c'est faux** : le
mémo donne une définition précise (relaxation de Lovász-θ).

| | Notre `theta.py` | Mémo officiel |
|---|---|---|
| Objet | modes propres intermédiaires `v_3..v_K` | **relaxation SDP du nombre de Lovász `ϑ(H)`** |
| Graphe | Laplacien de `W` | graphe d'interdépendance `H = 1[W ≥ τ]` |
| Score | `max_k \|v_k[i]\|` | `FS(i) = ‖y_i‖²` via dual de Lemaréchal–Oustry |
| Sélection | extrêmes par mode | **farthest-point** (cosinus) |
| Sémantique | sous-clusters | **info non-redondante / signaux faibles** |

C'est **l'écart le plus profond** entre le MVP et le mémo. **Le mémo fait foi** : à l'adoption du
repo complet, `theta.py` devra implémenter la relaxation SDP (sous-gradient sur le dual, matrice
frontière `F`, scores `‖y_i‖²`, sélection farthest-point). La cible n'est pas « représentants de
sous-clusters » mais **« chunks d'information unique »**.
