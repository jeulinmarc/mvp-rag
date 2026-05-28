# 2.3 — Singular nodes : les chunks originaux

## Qu'est-ce qu'un Singular node ?

Intuitivement, c'est un chunk qui **ne ressemble à aucun autre** dans le corpus. Il porte une information ou une formulation qu'on ne retrouve nulle part ailleurs.

Exemples concrets :

- Dans un rapport annuel : la section "événement exceptionnel" est typiquement Singular — tout le reste du doc parle de chiffres, RH, marché, mais cette section unique parle d'un litige juridique précis.
- Dans une thèse : la définition originale proposée par l'auteur est Singular — c'est ce qui distingue cette thèse de la littérature.
- Dans un corpus médical : le case report d'un patient rare est Singular — la majorité des chunks parlent de cas typiques.

En graphe-speak : un Singular node a des **liens faibles avec tout le monde**. Il n'appartient à aucun cluster, il flotte un peu en marge.

## Pourquoi c'est précieux en RAG

Un RAG top-k vectoriel classique retourne les k chunks les plus similaires à la question. Risque : les k chunks viennent tous du même cluster (ils se ressemblent entre eux, redondance), et l'info "rare" mais importante reste invisible.

Exemple :

> Question : "Quels sont les risques mentionnés dans le rapport ?"

Sans Singular boost :
- Top-5 retourne 5 chunks qui parlent tous des risques de marché (les plus fréquents dans le doc).
- L'unique chunk qui parle du risque de litige juridique est en position 12. Jamais vu.

Avec Singular boost :
- On garde 3 chunks du top-5 classique.
- On ajoute 2 chunks Singular pertinents pour la question.
- Le chunk sur le litige juridique remonte et est inclus.
- La réponse couvre l'ensemble du paysage de risques, pas juste le mainstream.

C'est exactement ce que fait MMR aussi, mais le boost Singular est **précalculé globalement** sur tout le corpus, pas par requête. Plus efficace, plus stable.

## Comment les identifier formellement

Plusieurs critères, qu'on combine pour robustesse.

### Critère 1 — Faible centralité

Un Singular node a peu de connexions fortes. On mesure ça avec la **centralité de degré pondéré** (somme des poids des arêtes incidentes) :

```
centrality(i) = Σ_j w_{ij}
```

Plus la centralité est basse, plus le nœud est isolé du reste.

### Critère 2 — Faible similarité moyenne avec ses voisins

Variante affinée : on regarde uniquement les **top-k voisins** et on calcule leur similarité moyenne :

```
avg_top_k_sim(i) = (1/k) × Σ_{j ∈ top_k(i)} w_{ij}
```

Si même tes meilleurs voisins ne te ressemblent que faiblement (~0.4), tu es Singular. Si tes meilleurs voisins te ressemblent fortement (~0.9), tu es dans un cluster dense.

### Critère 3 — Forte projection sur les hautes fréquences spectrales

Le critère spectral, vraiment original. Souviens-toi de 2.2 :

- Les **basses fréquences** (premiers vecteurs propres, λ petits) captent la structure globale, les clusters.
- Les **hautes fréquences** (derniers vecteurs propres, λ grands) captent le détail local, les anomalies.

Un nœud Singular **ne participe pas aux modes basse fréquence** (il n'appartient à aucun cluster) mais **émerge fortement dans les modes haute fréquence** (il a sa propre identité atypique).

Formellement, on définit le **score spectral d'atypisme** :

```
singular_score(i) = Σ_{k=n-K+1}^{n} v_k[i]²
```

où `v_k` est le k-ième vecteur propre et on somme sur les K dernières valeurs propres (les plus grandes). Plus ce score est élevé, plus le nœud "résiste" à toute tentative de clustering — il a son propre comportement.

C'est une mesure profonde et plutôt rare. On la combinera avec les critères 1 et 2.

### Critère 4 — Faible appartenance à un cluster

Si on a calculé un spectral clustering en k clusters (via les K premiers vecteurs propres + k-means), un Singular node est celui dont la **distance au centroid de son cluster** est maximale. Il est "membre par défaut" mais ne ressemble pas vraiment au reste.

On ne l'utilisera pas en MVP (k-means est une dépendance en plus), mais à connaître.

## Comment on combine les critères

Chaque critère donne un score continu. Pour identifier les Singular nodes :

1. Calculer les 3 critères (centralité, sim moyenne aux voisins, score spectral) pour chaque nœud.
2. Normaliser chaque score (z-score ou min-max scaling) pour les rendre comparables.
3. Combiner via somme pondérée ou rang composite.
4. Trier et garder le top-N.

Eigenmind utilise typiquement les 10-20% du corpus comme "Singular set" — ni trop (alors c'est plus du Singular), ni trop peu (manque de diversité).

## Pondération des critères

Choix par défaut basé sur l'expérience :

| Critère | Poids |
|---|---|
| Score spectral (haute fréquence) | 0.5 |
| 1 − centralité normalisée | 0.3 |
| 1 − sim moyenne voisins | 0.2 |

Pourquoi cette pondération : le score spectral est le plus "intelligent" (il capte vraiment l'atypisme global). Les autres sont des proxies plus directs mais moins fins.

## Le piège : Singular ≠ outlier de bruit

Attention : un chunk Singular n'est pas un chunk **mauvais**. Si tu as un chunk avec du texte parasite (header de page, footer, char Unicode bizarre), il sera mécaniquement Singular parce qu'il ne ressemble à rien de propre. Mais ce n'est pas un chunk précieux — c'est du bruit.

Mitigations :

- **Nettoyer le texte** en amont : enlever les en-têtes répétés, normaliser l'Unicode.
- **Filtrer par longueur** : un chunk Singular très court (< 100 chars) est probablement un artefact.
- **Sanity check humain** : afficher les 5 chunks les plus Singular et vérifier qu'ils ont du sens. Si tu vois "Page 12 — Confidentiel", tu sais qu'il faut nettoyer en amont.

On garde simple en MVP : on filtre uniquement par longueur minimale.

## Utilisation downstream

Les Singular nodes seront exploités en 2.6 (Hybrid Retrieval) de deux façons :

1. **Boost à la requête** — quand on récupère le top-k Qdrant, on regarde si des Singular nodes pertinents (cos avec la question > seuil) existent en dehors du top-k. On les ajoute.

2. **Diversification automatique** — on remplace les chunks redondants du top-k par des chunks Singular pertinents. C'est notre version Eigenmind du MMR.

En phase 3 (Streamlit), on les afficher avec une couleur spéciale dans le Graph Explorer pour les rendre visuellement repérables.

## Cas extrêmes

- **Corpus trop petit** (n < 20) : les Singular nodes n'ont pas beaucoup de sens, le concept de "majorité du corpus" est mal défini. On les calcule quand même mais avec une remise en contexte.
- **Corpus très homogène** (tous les chunks parlent de la même chose) : aucun nœud n'est vraiment Singular. Le score d'atypisme est faible partout. C'est normal et informatif — ton corpus n'a pas de diversité interne.
- **Corpus très diversifié** (multi-sujets sans lien) : presque tout est Singular. Là le concept perd son sens — il faut clusteriser d'abord, puis chercher des Singular **dans chaque cluster**.

Pour Eigenmind, on assume un corpus thématiquement cohérent avec quelques zones atypiques. C'est le cas le plus courant.