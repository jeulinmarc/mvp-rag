# 2.6 — Retrieval hybride : fusionner cosinus et signaux graphe

## Le problème du retrieval pur dense

Avec le retrieval Qdrant classique, on récupère les **k chunks les plus similaires** à la question. C'est simple, rapide, efficace dans la majorité des cas. Mais trois limites :

**1. Redondance.** Si 5 chunks du doc disent presque la même chose, ils remontent tous ensemble. Le LLM reçoit 5 fois la même info et rate la diversité.

**2. Tunnel vision.** Le top-k est concentré autour de la question — toutes les nuances latérales (info connexe, contexte historique, exception) sont invisibles. Le LLM répond précisément mais sans recul.

**3. Pas de "remarquables".** Un chunk Singular ou Hinge pertinent pour la question peut se trouver hors du top-k strict (score cosinus #15 par exemple). Il est perdu.

C'est exactement ce que la phase 2 corrige. On a maintenant trois ensembles de nœuds "remarquables" : Singular, Hinge, Theta. Le retrieval hybride les exploite.

## Principe du retrieval hybride Eigenmind

```
top_k_dense  = Qdrant.search(question, k=K_DENSE)
candidates   = top_k_dense ∪ Singulars ∪ Hinges ∪ Thetas
rescored     = pour chaque c dans candidates :
                 score = cos(c, question) + boost selon c.type
sorted_top_k = top K finals après tri par score
```

C'est une **réunion + rescoring**. Pas un remplacement du dense par le graphe — un enrichissement.

## Boost : combien et pourquoi

Chaque type de nœud reçoit un boost différent qui s'ajoute au score cosinus :

| Type | Boost | Justification |
|---|---|---|
| Top-k dense (normal) | 0 | Baseline |
| Singular pertinent | +0.10 | Récompense l'originalité informative |
| Hinge pertinent | +0.07 | Récompense la connexion inter-thématique |
| Theta pertinent | +0.05 | Récompense la diversité de couverture |

Les valeurs sont **petites** par rapport à l'écart de cosinus typique entre chunks (~0.1-0.2). Un Singular avec un cosinus de 0.4 sera moins prioritaire qu'un chunk dense avec 0.65. Mais un Singular à 0.55 dépassera un chunk dense à 0.62.

C'est l'arbitrage délicat : on veut **enrichir** sans **noyer**. Trop de boost → on retourne des chunks hors-sujet. Trop peu → on n'a rien changé.

## Seuil de pertinence pour les Singular/Hinge/Theta

Un Singular qui a un cosinus de 0.10 avec la question n'a aucune raison d'être ajouté — c'est juste un chunk atypique du corpus, sans rapport avec la question. On filtre donc par **seuil minimal** :

```
si cos(node, question) < SEUIL_PERTINENCE :
    on ignore ce nœud, même s'il est Singular/Hinge/Theta
```

Eigenmind utilise typiquement `SEUIL_PERTINENCE = 0.30`. En dessous, c'est du bruit.

## Pseudocode complet

```python
def hybrid_retrieve(question, k_final=5):
    q_vec = embed(question)

    # Dense top-k (avec un fetch large pour avoir de la marge)
    dense_candidates = qdrant.search(q_vec, limit=K_DENSE_FETCH)  # ex 15

    # Special nodes (déjà calculés)
    singulars = load_singular_ids()
    hinges = load_hinge_ids()
    thetas = load_theta_ids()

    candidates = {}  # node_id → {score, payload, type}

    # Add dense top-k
    for d in dense_candidates:
        candidates[d.id] = {
            "score": d.score,
            "payload": d.payload,
            "type": "dense",
        }

    # Add special nodes if relevant enough
    for special_id, type_name, boost in [
        (singulars, "singular", 0.10),
        (hinges, "hinge", 0.07),
        (thetas, "theta", 0.05),
    ]:
        for nid in special_id:
            if nid in candidates:
                # already in dense, just bump up
                candidates[nid]["score"] += boost
                candidates[nid]["type"] = f"dense+{type_name}"
            else:
                # fetch this node and check relevance
                point = qdrant.retrieve(nid)
                cos = dot(q_vec, point.vector)
                if cos >= SEUIL_PERTINENCE:
                    candidates[nid] = {
                        "score": cos + boost,
                        "payload": point.payload,
                        "type": type_name,
                    }

    # Sort and return top K
    sorted_candidates = sorted(candidates.values(), key=lambda c: -c["score"])
    return sorted_candidates[:k_final]
```

## Diversification implicite

Cette approche fait deux choses élégantes :

**1. Boost cumulatif.** Si un chunk est à la fois dense top-k **et** Singular, il reçoit le score dense + boost Singular. Il monte donc dans le ranking. C'est le signe qu'on a un chunk "doublement pertinent" — la requête pointe vers du contenu **original**.

**2. Couverture multi-thématique.** Si la question concerne un sujet à cheval sur deux clusters du doc, le top-k dense ramènera probablement des chunks d'un seul cluster (le plus proche cosinus). En boostant les Hinges, on rajoute la passerelle vers l'autre cluster. La réponse devient plus riche.

C'est essentiellement le **rôle de MMR**, mais résolu différemment : avec MMR la diversité est calculée à la requête, à chaque appel. Avec Eigenmind elle est **précalculée** dans le graphe une fois pour toutes — plus rapide à la requête.

## Comparaison avec MMR

| | MMR | Eigenmind hybrid |
|---|---|---|
| Diversité calculée | À la requête | Précalculée (offline) |
| Coût par requête | O(k_fetch²) | O(k_fetch) |
| Notion de "spécial" | Relativiste (vs résultats déjà choisis) | Globale (rôle du nœud dans le corpus) |
| Réutilisable | Non, doit recompute | Oui, les Singular/Hinge/Theta servent partout |
| Visualisation | Non | Oui (Graph Explorer) |

L'approche Eigenmind est plus lourde à mettre en place (toute la phase 2) mais une fois en place, elle est plus puissante et plus rapide à l'inférence.

## Réglage des hyperparamètres

Les valeurs par défaut conviennent à la plupart des cas, mais tu peux ajuster :

- **`K_DENSE_FETCH`** (15) : combien de candidats dense récupérer avant rescoring. Plus c'est haut, plus la marge de manœuvre est grande, mais plus la requête à Qdrant est lente. 15-30 est raisonnable.
- **`SEUIL_PERTINENCE`** (0.30) : sous ce score cosinus, on ignore. Plus haut → moins de boost effectif, plus conservatif. Plus bas → plus de chunks atypiques inclus, plus de risque de bruit.
- **`BOOST_SINGULAR`** (0.10) : à augmenter si tu veux que les Singular dominent davantage (corpus très répétitif). À baisser si trop de bruit.
- **`BOOST_HINGE`** et **`BOOST_THETA`** : similaire. Hinge plutôt à valoriser pour des questions de synthèse, Theta pour des questions exploratoires.

Calibrer ces valeurs avec un eval set est l'étape clé pour passer en prod. En MVP on garde les défauts.

## Logging et audit

Une bonne pratique : à chaque retrieval hybride, **logger le type de chaque chunk inclus** dans le contexte. Ça permet de diagnostiquer :

- Si le LLM hallucine sur un point, on peut voir si c'est venu d'un chunk dense, Singular, Hinge ou Theta.
- Si les réponses sont systématiquement biaisées vers un type, on ajuste les boosts.
- On peut afficher les types dans l'UI (phase 3) pour transparence utilisateur.

## Cas où le hybrid n'apporte rien

Deux cas où le retrieval dense suffit largement :

**1. Corpus très petit** (<30 chunks) : la couche graphe a peu de matière. Le top-k dense couvre déjà la moitié du corpus. Skip phase 2 dans ce cas.

**2. Questions très factuelles** ("Quel est le CA 2024 ?") : il n'y a qu'un seul chunk qui répond, et le retrieval dense le trouve sans souci. Le boost n'aide pas, peut même nuire (un Singular non pertinent fait du bruit).

Le hybrid brille sur les questions **synthétiques, exploratoires, multi-aspect** : "Quels sont les risques mentionnés ?", "Résume les contributions principales", "Comment X et Y sont-ils liés ?".

## Le piège : ne pas re-fetch tout le corpus

Tentation naïve : pour calculer cos(Singular, question) sur tous les Singular, on pourrait re-fetch tous les vecteurs Singular depuis Qdrant à chaque requête. Coûteux.

Solution : **cacher les vecteurs Singular/Hinge/Theta** en RAM au démarrage de l'app. Quelques centaines de vecteurs 384-d = quelques Mo, ridicule. À la requête, le calcul de cosinus est juste un produit scalaire local, instantané.

C'est ce qu'on fait dans le code : `vectors_cache` est chargé une fois et réutilisé.

## Évolution Phase 4+

En phase 4 (refactor), le `hybrid_retrieve.py` devient `pipelines/rag.py` ou `vectordb/retrieval.py`. Le pattern reste identique, on structure juste mieux.

En phase 5 (avancé), on pourra ajouter :
- Reranker cross-encoder par-dessus pour affiner.
- Hybrid avec BM25 (sparse + dense + graph signals).
- Learnable boosts via une mini régression sur eval set.

Eigenmind 2.0 quoi.
