# 2.7 — Épistémologie, régime opératoire et validation

> **Référence faisant foi.** Mémo officiel §1 (*Purpose and Reading Guide*), §4.6 (*Practical
> Scope, Assumptions, and Failure Modes*), §4.8 (*What Is Formal, What Is Heuristic, What Is
> Empirical*) et §4.9 (*Summary*). Ce chapitre rassemble le matériel **transversal** du mémo,
> absent des docs précédents.

## L'épistémologie à trois niveaux (le fil rouge du mémo)

Le mémo **inverse** l'ordre habituel : il part de ce qui est *réellement implémenté* dans le code,
puis demande *« pourquoi ça marche ? »* et *« qu'est-ce que ça veut dire pour un corpus
stratégique ? »*. Surtout, il distingue **trois niveaux de revendication** — et c'est crucial pour
ne pas surinterpréter les résultats :

| Niveau | Définition | Exemples dans Eigenmind |
|---|---|---|
| **Théorème formel** | Énoncé avec preuve | Multiplicité de 0 = nb de composantes connexes ; inégalité de Cheeger ; sandwich de Lovász `α ≤ ϑ ≤ χ(H̄)` ; dualité forte SDP |
| **Intuition mathématique** | Argument heuristique ancré dans la théorie, mais sans garantie formelle dans ce contexte | Les vecteurs propres basses fréquences encodent la variation thématique ; les géodésiques log-similarité suivent les routes de haute affinité ; le champ comme proxy ℓ∞ ; les gaps spectraux signalent des partitions |
| **Interprétation sémantique** | Mapping heuristique d'un objet mathématique vers un concept stratégique ; suggestif, non prouvé | « pôles = archétypes thématiques » ; « hinge = pont de négociation » ; « theta = signal faible » ; analogie de Shannon pour les frontières |

> **À retenir.** Les boîtes « interprétation sémantique » du mémo doivent être lues comme des
> **hypothèses de travail à valider empiriquement**, pas comme des théorèmes. Un pôle spectral est
> *formellement* un antipode du vecteur de Fiedler ; qu'il soit *sémantiquement* un thème
> représentatif dépend de l'encodeur et du corpus.

### Le tableau consolidé (§4.8)

| Type de revendication | Statut & caveat |
|---|---|
| **Théorème formel** | Prouvé. S'applique aux objets mathématiques tels que définis, sous hypothèses standard (graphe connexe, `A ≻ 0`, régularité SDP). |
| **Intuition mathématique** | Ancrée dans la théorie établie mais **sans garantie formelle** dans ce contexte. Dépend de la qualité d'embedding, de la connectivité et de la calibration de `τ`. |
| **Interprétation sémantique** | Hypothèses de travail seulement. Chaque revendication exige une validation empirique spécifique au corpus (revue d'expert, analyse de stabilité, comparaison à des baselines). |
| **Approximation heuristique** | Le schéma sous-gradient (theta) **n'a aucun certificat de convergence** en itérations finies. Scores `ϑ`-inspirés, pas optima SDP exacts. Valider contre solveur exact sur petits corpus. |

Conclusion du mémo : le pipeline est défendable comme **système d'attention géométrique-sémantique**
— chaque étage a une motivation mathématique claire, des limites connues et un mode de défaillance
bien défini. L'écart entre garanties formelles et comportement déployé se réduit progressivement
via le programme de validation (ci-dessous).

## Régime opératoire et complexité (§4.6.1)

Eigenmind opère sur des **sous-graphes sémantiques locaux** à l'échelle analyste, pas sur des
corpus entiers — d'où les plafonds :

| Opération | Complexité | Limite pratique |
|---|---|---|
| Décomposition propre complète (Singular) | `O(n³)` | `n ≲ 1000` |
| Dijkstra toutes paires (Hinge) | `O(n² log n)` | `n ≲ 500` |
| Sous-gradient dual `T` itérations (Theta) | `O(T n²)` | `n ≲ 300–500` |

C'est pourquoi l'**exploration BFS** (`exploration.py`) borne le sous-graphe à
`MAX_CHUNKS_FOR_CONTEXT` : les trois stratégies ne sont calculables que sur quelques centaines de
nœuds.

## Hypothèses formelles (§4.6.2)

1. **Lissité de l'embedding** : des documents sémantiquement proches ont une similarité cosinus
   élevée.
2. **Connectivité du graphe** : `W` doit avoir peu de composantes connexes.
3. **Calibration du seuil** : `τ` gouverne **à la fois** `W` et le graphe d'interdépendance `H` ;
   il doit être réglé avec conscience du domaine.

## Modes de défaillance connus (§4.6.3)

- **Embedding collapse** : un encodeur hors-domaine mappe des documents distincts sur des vecteurs
  proches → les trois stratégies dégénèrent.
- **Hubness sémantique** : des nœuds de très haut degré dominent les chemins géodésiques et biaisent
  le champ Hinge.
- **Fragmentation du graphe** : un `τ` trop agressif déconnecte le graphe ; le Laplacien a alors
  plusieurs valeurs propres nulles.
- **Corpus adversariaux/bruités** : des documents répétitifs gonflent `H` et saturent les scores
  frontière (Theta).

## Programme de validation empirique (§4.6.4 — travaux futurs)

1. **Analyse de stabilité** : sensibilité aux perturbations de prompt, à la suppression d'arêtes,
   au changement d'encodeur.
2. **Évaluation humaine** : jugement d'expert sur les labels thématiques / hinge / frontière.
3. **Comparaison de baselines** : PageRank, betweenness centrality, HDBSCAN, k-means, **MMR**, RAG
   standard.
4. **Nouveauté quantitative** : couverture sémantique, réduction de redondance, scoring de
   nouveauté par expert.

> C'est exactement le genre de validation à mener si tu branches le repo complet sur **tes propres
> données** : ne pas prendre les tags Singular/Hinge/Theta pour argent comptant sans une passe de
> sanity-check humain sur ton corpus.

## Résumé : les trois stratégies, géométriquement complémentaires (§4.9)

| Module | Opération cœur | Objet mathématique | Classe sémantique | Niveau |
|---|---|---|---|---|
| `singular.py` | Vecteurs propres de `L_sym` ; antipodes spectraux | Vecteur de Fiedler, Cheeger | Thematic Representatives | **formel** |
| `connectivity.py` | Champ géodésique ; niveau zéro ℓ∞ ; score hinge | Géodésiques log-similarité, quotient de Rayleigh ℓ∞ | Hinge Connectors | **intuition** |
| `theta.py` | Dual de Lemaréchal–Oustry ; `F = YYᵀ` | Lovász `ϑ`, relaxation SDP (approx.) | Frontier Themes | **approx. heuristique** |
| `exploration.py` | BFS sur index ANN approché | Voisinage local (approché) | Sous-graphe de travail | — |
| `similarity_graph.py` | Dispatch parallèle mémoïsé | Union des trois rankings | Singular / Hinge / Theta | — |

> Ensemble, ils fournissent une **géométrie de l'attention analytique** : les **pôles** orientent
> l'analyste dans l'espace narratif ; les **hinges** suggèrent où le dialogue est possible ; la
> **frontière** avertit de ce qui est en train d'être manqué.

## Références clés du mémo

Bibliographie sélective (numérotation du mémo) :

- **[1]** Malkov & Yashunin (2018), *HNSW* — index ANN. **[3]** Reimers & Gurevych (2019),
  *Sentence-BERT*. **[4]** Chung (1997), *Spectral Graph Theory* — Cheeger.
- **[5]** Fiedler (1973), *Algebraic connectivity*. **[6]** von Luxburg (2007), *Spectral
  clustering tutorial*. **[7]** Coifman & Lafon (2006), *Diffusion maps*.
- **[10]** Tenenbaum et al. (2000), *Isomap* — géodésiques. **[12]** Carbonell & Goldstein (1998),
  *MMR*.
- **[13]** Lovász (1979), *Shannon capacity / nombre ϑ*. **[14]** Grötschel, Lovász, Schrijver
  (1988). **[15]** Lemaréchal & Oustry (2001), *SDP relaxations / Lagrangian*. **[16]** Polyak
  (1987), pas sous-gradient. **[18]** Boyd & Vandenberghe (2004), *Convex Optimization*.

Le PDF complet est dans `theory/260522_Eigenmind_Cognitive_Maps.pdf`.
