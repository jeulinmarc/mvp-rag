# 2.6 — Agrégation : du sous-graphe aux trois labels (SimilarityGraph)

> **Référence faisant foi.** Mémo officiel §4.5 (*The Pipeline in One Picture*) et §4.7
> (*Aggregation — The SimilarityGraph Class*). ⚠️ **Cette définition remplace la version
> précédente**, qui décrivait un retrieval « hybride » par **boosts additifs** (`+0.10 / +0.07 /
> +0.05`). Ce mécanisme de boost **n'existe pas dans le mémo** : l'agrégation officielle produit
> un **tagging à trois labels**, pas un re-ranking pondéré. Note d'implémentation en fin de doc.

## Le pipeline en une image

Les deux premiers étages sont **séquentiels**, les trois stratégies tournent ensuite **en
parallèle** sur le même `W`, et leurs sorties sont fusionnées par `selection_tags` :

```
exploration.py        → récupère le sous-graphe de travail (BFS sur l'index ANN)
similarity_graph.py   → construit et mémoïse W
        │
        ├── singular.py       → pôles thématiques  (tag "Singular")
        ├── connectivity.py   → connecteurs        (tag "Hinge")
        └── theta.py          → thèmes-frontière    (tag "Theta")
        │
selection_tags        → fusionne en Singular / Hinge / Theta
```

La classe `SimilarityGraph` **détient `W`** et dispatche paresseusement vers les trois modules,
en **mémoïsant** les résultats intermédiaires (spectre, champ `x*`, matrice frontière `Y`) :
chacun n'est calculé qu'**une seule fois** par instance.

## `selection_tags(top_k)` : la fonction d'agrégation

```
selection_tags(top_k):
  1. singular_chunks()              → tag "Singular"
  2. hinge_ranking()   (x* mémoïsé) → tag "Hinge"  sur le top-k
  3. theta_diversity(k) (Y mémoïsé) → tag "Theta"  sur le top-k
```

Le résultat n'est **pas** un score fusionné : c'est un **étiquetage** des chunks du sous-graphe.
Chaque chunk peut porter **zéro, un, ou plusieurs** des trois labels.

### Les trois labels

| Tag | Ce que ça signifie | Question d'analyste |
|---|---|---|
| **Singular** | antipode spectral sur un mode basse fréquence | Quels sont les axes dominants de variation thématique ? |
| **Hinge** | proche du niveau zéro `ℓ∞`, connecté aux deux pôles | Quels concepts relient les deux extrêmes ? |
| **Theta** | proxy d'interdépendance faible, score frontière `FS(i)` élevé | Quels chunks portent une info unique, non-redondante ? |

## Ce que révèlent les chunks multi-labels

La richesse vient des **combinaisons** (§4.7.3) :

- **Singular + Hinge** : un antipode spectral **également connecté au pôle opposé** → un **concept
  contesté**, revendiqué par les deux camps.
- **Hinge + Theta** : un pont **aussi informationnellement auto-suffisant** → une **idée
  potentiellement fédératrice** (coalition-enabling).
- **Les trois** : densité structurelle maximale → **à lire en premier**.

## Les deux régimes de retrieval (rappel de 02-1)

L'agrégation ci-dessus opère sur le sous-graphe **BFS** (couverture). Elle est complémentaire du
**top-k dense** (`similarity_search` dans `pipelines/rag.py`), qui répond directement « quels
chunks sont les plus proches du prompt ? ». Le mémo (§3.2.2) insiste : ces deux régimes ne sont
**pas redondants**.

- Top-k dense → *pertinence au prompt* (RAG classique de question-réponse).
- BFS + tags → *couverture du voisinage du prompt* (détection de contexte ambiant, concepts
  contestés, angles morts).

## Le principe de design : des chunks nommés, pas des vecteurs latents (§4.7.4)

Chaque tag est attaché à un **chunk nommé et lisible** (filename + chunk_number + texte). Un
analyste peut **lire, vérifier et annoter** chaque chunk tagué. Le système n'opère jamais sur des
vecteurs latents opaques : il maintient un **contrôle épistémique** sur un pipeline mathématique.

> C'est le fil rouge de tout Eigenmind, déjà rencontré dès le chunking (`01-3` / `01-8`) : *« quel
> est le plus petit morceau de texte que je peux encore lire, attribuer et sur lequel raisonner ? »*

## ⚠️ Note d'implémentation MVP

Notre `hybrid_retrieve.py` implémente une **fusion par boosts** : il récupère un top-k dense large,
ajoute les nœuds Singular/Hinge/Theta dont le cosinus à la question dépasse `0.30`, applique des
boosts (`+0.10 / +0.07 / +0.05`) et re-trie. **Ce mécanisme n'apparaît pas dans le mémo.**

| | Notre `hybrid_retrieve.py` | Mémo officiel |
|---|---|---|
| Sortie | re-ranking dense + boosts | **tagging** Singular/Hinge/Theta (`selection_tags`) |
| Sous-graphe | k-NN global mis en cache | sous-graphe **BFS** local par requête |
| Rôle des nœuds spéciaux | bonus de score additif | **labels** sur des chunks lisibles |
| Objectif | un seul classement fusionné | deux régimes (top-k vs couverture) + labels |

Notre approche par boosts reste un moyen pédagogique commode de **voir l'effet** des nœuds spéciaux
sur le retrieval (et c'est exactement ce que le mode `compare` du CLI illustre). Mais **le mémo
fait foi** : à l'adoption du repo complet, le retrieval devra séparer les deux régimes et exposer
les trois labels plutôt que de fusionner en un score unique.

## (Note) MMR vs Eigenmind

Le mémo cite MMR [Carbonell & Goldstein 1998] uniquement comme **baseline de comparaison** pour la
validation empirique (§4.6.4), pas comme l'analogue d'Eigenmind. La diversité chez Eigenmind n'est
pas un re-ranking glouton à la requête : elle émerge de la **structure du graphe** (antipodes,
ponts, code non-confusable) calculée sur le sous-graphe de travail.
