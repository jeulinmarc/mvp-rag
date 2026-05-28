# 2.4 — Hinge nodes : les chunks-pivots

## Qu'est-ce qu'un Hinge node ?

Un Hinge node est un chunk qui **fait le pont entre plusieurs clusters thématiques**. Il appartient à la **frontière** entre deux ou plusieurs zones du corpus, pas au cœur d'aucune.

Exemples concrets :

- Dans un rapport multi-thème : la phrase de transition qui passe du chapitre "ventes" au chapitre "RH" en évoquant les deux.
- Dans une thèse interdisciplinaire : le paragraphe qui relie le contexte théorique (sciences sociales) à la méthode quantitative (statistiques).
- Dans un papier de ML : la section qui connecte la formulation mathématique au protocole expérimental.

En graphe-speak : un Hinge node est traversé par beaucoup de **chemins courts** entre paires de nœuds qui ne sont pas voisins directs. Si on le supprimait, plusieurs chemins du graphe deviendraient soudainement très longs — ou disparaîtraient.

## Pourquoi c'est précieux en RAG

Trois utilités principales :

**1. Réponses inter-thématiques.** Question : "Comment la stratégie commerciale s'aligne-t-elle avec la politique RH ?" Aucun chunk du cluster "ventes" ni du cluster "RH" ne répond directement. Mais le Hinge entre ces deux clusters est probablement la clé. Sans lui dans le contexte, le LLM doit "deviner" la connexion. Avec lui, elle est explicite.

**2. Cohérence globale.** Quand le LLM voit uniquement le top-k cosinus, il voit un sous-ensemble cohérent mais étroit du corpus. Ajouter un Hinge donne une **vue d'ensemble** : "voilà comment ce sujet s'articule avec le reste du doc". Améliore les synthèses.

**3. Exploration.** Quand un utilisateur explore un corpus inconnu, partir des Hinge nodes lui montre la **structure narrative** du document — comment les idées s'enchaînent. Plus utile que partir de chunks aléatoires.

## Distinction Singular vs Hinge

À ne pas confondre, même si les deux sont des nœuds "spéciaux" :

| | Singular | Hinge |
|---|---|---|
| Position dans le graphe | En marge | À la frontière |
| Similarité aux voisins | Faible | Moyenne |
| Centralité de degré | Basse | Souvent élevée |
| Rôle | Atypique, original | Connecteur |
| Risque | Bruit (artefacts) | Chunk générique (peu informatif) |
| Intuition | "Ne ressemble à rien" | "Ressemble à tout le monde" |

Un Singular est isolé. Un Hinge est connecté à tout (ou à plusieurs zones distinctes).

## Comment les identifier formellement

Plusieurs métriques classiques en théorie des graphes, à combiner.

### Critère 1 — Betweenness centrality (le plus important)

Pour chaque paire de nœuds `(u, v)` du graphe, on calcule le **plus court chemin** entre eux. Pour chaque nœud `i`, on compte la **fraction de ces chemins qui passent par `i`**.

```
betweenness(i) = Σ_{u≠v≠i} (nb_shortest_paths(u,v) passant par i) / nb_shortest_paths(u,v)
```

Plus la betweenness est élevée, plus le nœud est un point de passage obligé du graphe. Les Hinges ont une **betweenness très élevée** par construction.

Coût : O(n·m) pour un graphe sparse avec l'algorithme de Brandes. Sur 1000 nœuds et 5000 arêtes, ça prend ~1s en NetworkX. C'est la métrique la plus coûteuse, on l'utilise quand même parce qu'elle est canonique.

### Critère 2 — Faible projection sur le Fiedler vector

Souviens-toi du Fiedler vector (vu en 2.2) : il sépare le graphe en deux clusters selon le signe. Les nœuds avec **v_2[i] proche de 0** sont sur la frontière entre les deux clusters — ils ne penchent ni d'un côté ni de l'autre.

```
hinge_fiedler_score(i) = 1 - |v_2[i]|
```

Plus le score est haut, plus le nœud est sur la frontière.

### Critère 3 — Diversité des voisins (multi-cluster membership)

Un Hinge a des voisins **dans plusieurs clusters**. Pour formaliser ça, on prend le spectral embedding (les `k` premiers vecteurs propres) et on regarde la **dispersion** des coordonnées spectrales des voisins.

```
neighbor_diversity(i) = std({embedding_spectral(j) for j ∈ N(i)})
```

Plus la dispersion est grande, plus les voisins sont éclatés dans plusieurs clusters → plus le nœud est un pont.

### Critère 4 — Articulation points (cas extrême)

Un **articulation point** est un nœud dont la suppression **déconnecte le graphe**. C'est le Hinge maximal. NetworkX a `nx.articulation_points(G)` qui les liste.

En pratique sur un k-NN graph bien connecté, il y en a peu (le graphe est trop redondant). Mais quand il y en a, ce sont des Hinges **certains**. On les boost en priorité.

## Pondération des critères

Choix Eigenmind par défaut :

| Critère | Poids |
|---|---|
| Betweenness centrality | 0.5 |
| 1 − \|v_2\| (frontière Fiedler) | 0.3 |
| Neighbor diversity spectrale | 0.2 |

La betweenness est le pilier — c'est *la* mesure classique des nœuds-ponts en théorie des graphes. Les autres affinent.

Si un nœud est articulation point, on lui donne automatiquement un score = 1.0 (bypass).

## Pourquoi pas seulement betweenness ?

Question naturelle : pourquoi ne pas se contenter de la betweenness ?

Deux raisons :

**1. Coût.** Betweenness exact est O(n·m). Sur 10 000 nœuds et 50 000 arêtes, ça prend ~10s. Combiné avec d'autres métriques moins coûteuses (Fiedler vector déjà calculé), on a un score plus robuste sans coût additionnel significatif.

**2. Robustesse.** La betweenness pure peut "élire Hinge" un nœud central d'un cluster homogène simplement parce que beaucoup de chemins passent par lui (effet "hub"). Le critère Fiedler corrige : un vrai Hinge est à la frontière, pas au centre d'un cluster. La combinaison des deux est plus discriminante.

## Le piège : Hinge ≠ chunk générique

Un chunk avec du contenu très général ("Cette section présente nos résultats principaux") peut artificiellement avoir une betweenness élevée parce qu'il "ressemble à tout le monde un peu". Mais il n'apporte pas vraiment de connexion thématique informative.

Mitigations :

- **Filtrer par longueur** comme pour Singular — un Hinge trop court (<150 chars) est suspect.
- **Vérifier le Fiedler score** — un vrai Hinge doit avoir v_2 proche de 0, pas juste une betweenness élevée.
- **Sanity check humain** : afficher les top Hinges et vérifier que ce sont des passages charnières et non du remplissage.

## Approximations pour grands graphes

Pour des graphes >10 000 nœuds, betweenness exact devient prohibitif. NetworkX propose `nx.betweenness_centrality(G, k=1000)` qui calcule sur un échantillon de **k nœuds sources**. Précision réduite mais beaucoup plus rapide. Pour Eigenmind à l'échelle d'un utilisateur, on garde exact.

## Utilisation downstream

Les Hinge nodes seront exploités en 2.6 et phase 3 :

- **Boost contextuel** : quand le top-k Qdrant ramène des chunks de plusieurs clusters distincts, on ajoute le Hinge qui les connecte. La réponse devient plus cohérente.
- **Visualisation** : dans le Graph Explorer, les Hinges seront colorés différemment (rouge/orange) pour montrer la structure narrative du document.
- **Résumés inter-thématiques** : quand l'utilisateur demande "donne-moi un résumé global", on bourre le contexte de Hinges plutôt que de chunks redondants.

## Lien avec d'autres notions

- **Communities detection** (Louvain, Leiden) : les Hinges sont à la frontière entre communautés détectées. Mais on n'a pas besoin de Louvain : la décomposition spectrale fait le job indirectement via le Fiedler vector.
- **Bridges** (théorie des graphes) : une arête bridge est une arête dont la suppression déconnecte le graphe. Un nœud articulation est l'équivalent côté nœud. Concepts duaux.
- **Brokerage** (réseaux sociaux) : Burt's structural holes — un Hinge est l'analogue d'un "broker" qui exploite les trous structurels d'un réseau social.

C'est tout l'avantage d'utiliser des concepts standards de théorie des graphes : on hérite d'un demi-siècle de littérature.
