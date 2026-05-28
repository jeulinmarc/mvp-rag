# 2.5 — Theta nodes : extraction par modes propres intermédiaires

## Préambule honnête

La notion de "Theta node" est **spécifique à Eigenmind**. Elle n'a pas de définition standard dans la théorie des graphes — c'est une appellation maison qui rassemble une idée précise : utiliser les **modes propres intermédiaires** du Laplacien pour identifier des chunks qui ne sont ni Singular (marges spectrales hautes) ni Hinge (frontière Fiedler), mais portent des **sous-structures locales** intéressantes.

Concrètement, on va définir une heuristique qui s'inspire de la décomposition spectrale au-delà du Fiedler vector pour repérer des chunks "représentants" de sous-clusters.

## L'intuition

Souviens-toi du paysage spectral (vu en 2.2) :

| Bande spectrale | Vecteurs propres | Capture |
|---|---|---|
| Basses fréquences (λ_1, λ_2) | v_1, v_2 | Structure globale, bipartition |
| **Fréquences intermédiaires** (λ_3 à λ_K) | **v_3 à v_K** | **Sous-clusters, raffinements** |
| Hautes fréquences (λ_{n-K+1} à λ_n) | v_{n-K+1} à v_n | Atypisme local (Singular) |

Les **modes intermédiaires** sont rarement exploités directement, mais ils contiennent l'info de la structure multi-cluster du graphe. Chaque v_k pour k entre 3 et 10 peut être vu comme un "raffinement" du Fiedler vector : il subdivise les clusters précédents.

L'idée des Theta nodes : pour chaque mode intermédiaire v_k, identifier les nœuds avec une **forte projection positive ou négative** sur ce mode — ce sont les "représentants extrêmes" du sous-cluster correspondant.

## Définition formelle

Pour un nœud `i` et un mode propre `k`, on a une **projection** `v_k[i]`. On définit le score Theta comme :

```
theta_score(i) = max_{k=3..K} |v_k[i]|
```

Le `max` sur les K premiers modes intermédiaires (typiquement K=8). Si un nœud a une projection forte sur **au moins un** mode intermédiaire, il est un Theta — il représente fortement un sous-cluster.

C'est différent de Singular (qui regarde les **dernières** valeurs propres) et de Hinge (qui regarde la frontière de v_2 et la betweenness).

## Pourquoi pas juste faire du spectral clustering classique ?

Question légitime : on pourrait faire k-means sur l'embedding spectral, identifier des clusters, et prendre les centroïdes. Pourquoi cette approche par max-projection ?

Trois raisons :

**1. Pas besoin de fixer K à l'avance.** Le nombre de clusters d'un corpus est inconnu. La méthode par max-projection s'adapte automatiquement : si un mode propre intermédiaire ne sépare rien d'intéressant, il n'y a pas de nœud avec une forte projection dessus.

**2. Préservation de la hiérarchie.** Chaque mode v_k correspond à un niveau de granularité différent. v_3 capte une sous-structure plus large que v_8. La méthode max-projection laisse le nœud "choisir" le mode qui le caractérise le mieux.

**3. Compatibilité avec Singular et Hinge.** Comme Singular et Hinge sont déjà définis par d'autres bandes du spectre, Theta complète naturellement l'arsenal sans recouvrement.

## Le risque de recouvrement avec Singular et Hinge

Un nœud peut potentiellement satisfaire plusieurs critères. Solutions :

**1. Filtrage par exclusion.** Calculer d'abord Singular, puis Hinge, puis Theta sur ce qui reste. Les sets sont disjoints.

**2. Tagging multiple.** Un nœud peut être à la fois Singular et Theta. On accepte le recouvrement et on l'affiche dans la visualisation.

**3. Choix exclusif par score relatif.** Pour chaque nœud, on regarde lequel des trois scores est le plus élevé, et on lui assigne cette catégorie.

Eigenmind utilise l'option 1 (filtrage par exclusion). On calcule Singular, on retire ces nœuds, on calcule Hinge, on retire ces nœuds, **puis** on calcule Theta sur le reste. Trois ensembles disjoints, plus simples à interpréter.

## Combien de Theta extraire ?

On vise typiquement **10-20%** du corpus en Theta nodes, après retrait des Singular et Hinge. Si le corpus est très clusterisé, on aura beaucoup de Theta (un par sous-cluster). Si le corpus est homogène, on en aura peu.

## Critère de sélection raffiné

Plutôt que prendre les top-N globaux, on peut prendre **les top-K nœuds par mode propre** :

```
pour chaque k = 3, 4, ..., K:
    extraire les 2 nœuds avec max v_k[i] positif (extrême positif du mode k)
    extraire les 2 nœuds avec max v_k[i] négatif (extrême négatif du mode k)
```

Avec K=8, on prend ~28 nœuds qui couvrent toute la diversité des sous-clusters.

Cette approche garantit la **diversité** des Theta — pas tous concentrés sur le même mode propre. Eigenmind utilise cette stratégie.

## Utilisation downstream

Les Theta nodes sont les **représentants de sous-clusters**. Concrètement :

- **Vue d'ensemble** : la liste des Theta donne un aperçu rapide des sous-thématiques du corpus. "Voici 10 chunks qui résument la diversité du document."
- **Navigation** : dans le Graph Explorer (phase 3), cliquer sur un Theta saute vers le sous-cluster qu'il représente.
- **Diversification du retrieval** : quand l'utilisateur pose une question vague ("résume-moi le document"), on inclut des Theta dans le contexte pour garantir une couverture multi-thématique.

## Le piège : Theta très corrélés

Si le corpus a une structure très hiérarchique (gros cluster A subdivisé en A1, A2 ; gros cluster B subdivisé en B1, B2, B3), plusieurs modes propres peuvent **encoder la même information** (juste à des niveaux de granularité différents). Tu peux te retrouver avec des Theta très redondants — plusieurs nœuds qui représentent essentiellement A1.

Mitigation : déduplication post-hoc. Après avoir collecté tous les Theta candidates, on calcule leur similarité cosinus mutuelle (via leurs embeddings sémantiques d'origine, pas spectraux). Si deux Theta ont cos > 0.85, on garde celui avec le score plus haut et on retire l'autre.

## Quand les Theta sont inutiles

Deux cas où le concept perd son sens :

- **Corpus très petit** (<30 chunks) : les modes intermédiaires ne discriminent pas vraiment. v_3 ressemble à v_2, etc.
- **Corpus totalement homogène** : tous les modes intermédiaires ont des projections faibles partout. Les Theta extraits seraient quasi-aléatoires.

Dans ces cas, Eigenmind affiche un message d'info et passe les Theta. Le RAG continue de fonctionner avec Singular + Hinge.

## Récapitulatif : les trois types de nœuds

| Type | Bande spectrale | Critère principal | Rôle |
|---|---|---|---|
| **Singular** | Hautes fréquences (λ_n, λ_{n-1}, ...) | Forte projection sur les derniers v | Atypique global, original |
| **Hinge** | Basse fréquence (v_2 + betweenness) | Frontière Fiedler + betweenness | Pivot inter-cluster |
| **Theta** | Fréquences intermédiaires (v_3 à v_K) | Forte projection sur un mode intermédiaire | Représentant de sous-cluster |

Les trois ensembles sont disjoints (filtrage par exclusion) et couvrent ensemble les "chunks remarquables" du corpus. Le reste — la majorité — est du contenu "mainstream" qui forme la masse du document.

C'est cette anatomie en quatre catégories (Singular, Hinge, Theta, mainstream) qui structure tout Eigenmind : la visualisation, le retrieval hybride, et l'exploration.
