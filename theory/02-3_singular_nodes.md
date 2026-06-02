# 2.3 — Singular nodes : les pôles thématiques (Thematic Diversity)

> **Référence faisant foi.** Mémo officiel §4.2 (*Strategy 1 — Thematic Diversity*, module
> `singular.py`). ⚠️ **Cette définition remplace la version précédente de ce document**, qui
> décrivait les Singular comme des chunks « atypiques » détectés par les **hautes** fréquences
> spectrales. Le mémo fait foi : les Singular sont en réalité les **extrema des modes basses
> fréquences** — les pôles des axes thématiques. Voir la note d'implémentation en fin de doc.

## Qu'est-ce qu'un Singular node ?

Un Singular node n'est **pas** un chunk marginal. C'est un **représentant thématique** : l'un des
deux pôles opposés d'un **axe de variation thématique** du corpus. Le module s'appelle
`singular.py` parce qu'il travaille sur les valeurs *singulières* de la matrice d'embedding, mais
sa classe sémantique officielle est **« Thematic Representatives »**.

Question d'analyste associée : *« Quels sont les axes dominants de variation thématique dans ce
corpus ? »*

Exemples (corpus de stratégie financière, §4.2) :

- Sur un corpus dé-dollarisation, les deux pôles d'un axe pourraient être : `i⁻` = une analyse
  banque centrale sur la profondeur du marché des Treasuries US, et `i⁺` = un passage sur
  l'infrastructure CBDC transfrontalière des BRICS.
- Sur un corpus inflation, l'axe pourrait opposer fragmentation des chaînes
  d'approvisionnement vs normalisation monétaire.

Les pôles sont les **archétypes** entre lesquels se déploie le corpus, pas des outliers.

## La machinerie : décomposition spectrale du Laplacien normalisé

### 1. Laplacien normalisé

À partir de la matrice de similarité `W` (cf. `02-1`), on forme le **Laplacien normalisé** :

```
L_sym = I − D^(−1/2) W D^(−1/2) ,    D = diag(d) ,    d_i = Σ_j W_ij
```

Les nœuds isolés (`d_i = 0`) reçoivent une entrée `D^(−1/2)` nulle.

### 2. Décomposition propre complète

On calcule **tout** le spectre via un solveur symétrique (`np.linalg.eigh`, `O(n³)`, praticable
pour `n ≲ 1000`) :

```
0 = λ_0 ≤ λ_1 ≤ … ≤ λ_{n−1}   et leurs vecteurs propres u_0, u_1, …
```

### 3. Sélection des modes basses fréquences

Le code compte les valeurs propres quasi-nulles :

```
k = |{ j : λ_j < 10⁻³ }|
```

Formellement, la **multiplicité de `λ = 0` est le nombre de composantes connexes** de `W`. En
pratique, les sous-graphes issus du BFS sont presque toujours connexes (`k = 1`) ; la structure
thématique signifiante vit dans les **petites valeurs propres non nulles** et leurs **gaps
spectraux**. On sélectionne donc les premiers vecteurs propres **basses fréquences** non triviaux
(à partir du vecteur de Fiedler `u_1`).

### 4. Extraction des pôles

Pour chaque vecteur propre basse fréquence sélectionné `u_j` (`j = 1, …, k`) :

```
i⁺_j = argmax_i u_j(i)        i⁻_j = argmin_i u_j(i)
```

La paire `{i⁺_j, i⁻_j}` constitue les **Thematic Representatives du mode j** — les nœuds les plus
éloignés le long de l'axe de variation `j`. Ce sont les **Singular nodes**.

## Le fondement mathématique : Fiedler, énergie de Dirichlet, Cheeger

### Le vecteur de Fiedler minimise une énergie de Dirichlet normalisée

Pour `L_sym`, les vecteurs propres basse fréquence minimisent l'**énergie de Dirichlet
normalisée** :

```
E(u) = (uᵀ L_sym u) / ‖u‖²₂ = [ Σ_{i,j} W_ij (u_i − u_j)² / √(d_i d_j) ] / Σ_i u_i²
```

Ils encodent les fonctions qui **varient le plus lentement** sur le graphe pondéré par les degrés.
Le premier minimiseur non trivial `u_1` (vecteur de Fiedler) a ses **extrema aux antipodes
spectraux** — les deux nœuds les plus séparés le long de l'axe dominant de variation thématique.
C'est exactement `{i⁺_1, i⁻_1}`.

### Théorème formel : inégalité de Cheeger

L'inégalité de Cheeger (Chung, *Spectral Graph Theory*) relie la première valeur propre non
triviale à la **constante isopérimétrique** (conductance) `h(G)` du graphe :

```
λ_1 / 2  ≤  h(G)  ≤  √(2 λ_1) ,    h(G) = min_S  cut(S, S̄) / min(vol(S), vol(S̄))
```

Conséquence : une **petite** `λ_1` signale une coupe faiblement connectée — une séparation
thématique métastable — **pas** une structure bipartie. (La structure bipartie correspond aux
valeurs propres proches de **2** dans le spectre du Laplacien normalisé.) Un gap prononcé entre
`λ_j` et `λ_{j+1}` signale une partition thématique naturelle en `j` parties.

> C'est un vrai **théorème** (niveau de preuve le plus fort, cf. `02-7`). En revanche,
> l'interprétation « ces nœuds sont des thèmes représentatifs » est une **interprétation
> sémantique** : sélectionner les extrema d'un vecteur propre identifie des **antipodes
> spectraux**, pas des « thèmes représentatifs » au sens formel. Que ces nœuds soient
> sémantiquement représentatifs dépend de la qualité de l'embedding et doit être validé par des
> experts du domaine (précision importante du mémo, §4.2).

## Interprétation sémantique

> *(Heuristique)* Les Thematic Representatives **ancrent les axes de la diversité thématique.**

Visuellement, les pôles `i⁻` et `i⁺` se situent aux deux bouts de l'axe `u_1`, chacun entouré de
son propre voisinage de chunks similaires. Ils donnent à l'analyste les **deux extrémités** du
spectre de discours du corpus.

L'implication stratégique est directe : la qualité de l'output analytique est **majorée par la
fidélité** avec laquelle l'encodeur aligne la sémantique textuelle sur la géométrie angulaire.
Choisir l'encodeur est donc une décision éditoriale, pas technique.

## Coût et limites

- **Coût** : décomposition propre complète `O(n³)`, praticable `n ≲ 1000`. Au-delà, solveurs
  partiels (Lanczos/ARPACK via `scipy.sparse.linalg.eigsh`).
- **Signe des vecteurs propres** : `u_j` est défini au signe près. L'échange `i⁺ ↔ i⁻` entre deux
  runs est sans conséquence (l'axe est le même), mais à savoir pour les tests.
- **Mode de défaillance** : *embedding collapse* — un encodeur hors-domaine mappe des chunks
  distincts sur des vecteurs proches, ce qui dégrade tous les modes et donc les pôles.

## ⚠️ Note d'implémentation MVP

Notre `singular.py` calcule un **score d'atypisme hautes fréquences**
`singular_score(i) = Σ_{k=n−K+1}^{n} v_k[i]²` (chunks « qui ne ressemblent à rien »). **C'est
l'inverse spectral de la définition officielle**, qui utilise les **basses** fréquences et prend
les **extrema** des vecteurs propres comme pôles thématiques.

| | Notre `singular.py` | Mémo officiel |
|---|---|---|
| Bande spectrale | hautes fréquences (`λ` grands) | **basses** fréquences (Fiedler & co.) |
| Critère | forte énergie `Σ v_k²` | **extrema** `argmax`/`argmin u_j` |
| Sémantique | outlier atypique | **pôle d'un axe thématique** |

Les deux sont défendables, mais **le mémo fait foi**. À l'adoption du repo complet, `singular.py`
devra extraire les antipodes basses fréquences. La cible conceptuelle est *« axes de variation
thématique »*, pas *« chunks isolés »*.
