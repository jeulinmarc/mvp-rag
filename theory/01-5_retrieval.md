# 1.5 — Retrieval

## Le principe

On a embeddé et stocké N chunks. Pour répondre à une question, on suit trois étapes :

1. Embed la question avec le **même** modèle que les chunks (sinon les vecteurs ne sont pas dans le même espace, ils sont incomparables).
2. Demander à Qdrant les **k chunks les plus proches** de ce vecteur question, au sens de la similarité cosinus.
3. Récupérer le payload de chacun (texte + métadonnées) pour les passer ensuite au LLM.

C'est tout. La complexité technique du retrieval est dans l'index spatial de Qdrant — côté client, c'est un appel de fonction.

Mais "c'est tout" cache un univers de techniques d'amélioration. Cette section les couvre toutes — même celles qu'on n'utilise pas en MVP. Comprendre le paysage te permettra de savoir où aller quand tu voudras pousser la qualité.

## k-NN approximé : ANN

L'algorithme exact "k-Nearest Neighbors" compare la question à *tous* les vecteurs : O(N × d) opérations. Inutilisable au-delà de quelques milliers de vecteurs.

L'algorithme **HNSW** (vu en 01-1) approxime ce résultat avec une recherche en O(log N), avec un rappel typique de 95-99 % par rapport à l'exact. Pour un RAG, perdre un voisin de temps en temps n'a aucun impact perceptible — le LLM s'accommode du bruit.

### Le paramètre `ef` au runtime

À la recherche, HNSW utilise un paramètre `ef` qui contrôle la qualité-vs-vitesse. Plus `ef` est grand, plus la queue de candidats est large, meilleur le rappel, plus la latence augmente. Tu peux le passer par requête dans Qdrant :

```python
client.query_points(
    collection_name="documents",
    query=vec,
    limit=5,
    search_params={"hnsw_ef": 128},  # défaut 64
)
```

Pour Eigenmind on garde les défauts. Si tu remarques que le retrieval rate des chunks que tu sais pertinents, monter `ef` à 128 ou 256 avant de penser à des techniques plus avancées.

## Choix de k

Combien de chunks ramener ? Compromis classique :

- **k trop petit** (1-2) : si le bon passage n'est pas dans le top-2, la réponse est ratée.
- **k trop grand** (20+) : on noie le LLM dans du contexte non pertinent, on dépasse la fenêtre, on consomme des tokens.
- **Sweet spot** : 3 à 8 pour la plupart des cas. Eigenmind utilise typiquement 5.

Règle de pouce : si chaque chunk fait 500 caractères ≈ 150 tokens, et que tu vises 1500-2000 tokens de contexte, ça te donne k = 10-13. Mais avec un LLM moderne, fenêtre 128k tokens, k peut être plus généreux.

### Le compromis fondamental

Plus tu mets de chunks, plus tu réduis le risque de rater l'info, mais plus tu augmentes le risque de bruit. Le LLM doit alors filtrer lui-même le pertinent du non-pertinent dans son contexte, et il n'est pas toujours bon à ça (phénomène "lost in the middle"). À k=20, tu peux paradoxalement avoir des réponses **moins bonnes** qu'à k=5.

C'est ce qui motive les rerankers et MMR.

## Score threshold

Qdrant renvoie chaque résultat avec un score (similarité cosinus, entre -1 et 1, ou 0 et 1 pour des vecteurs normalisés positifs).

Tu peux imposer un seuil minimal : "renvoie le top-k, mais ignore tout ce qui a un score < 0.3". Utile pour deux raisons :

1. **Questions hors-sujet** : si la question n'a aucun rapport avec le corpus, on récupère quand même k chunks (les "moins éloignés"), avec des scores faibles. Sans threshold, on les passe au LLM qui essaie de répondre avec n'importe quoi. Avec threshold, on peut détecter le hors-sujet et répondre "je ne trouve rien sur ce sujet".

2. **Bruit de fond** : éviter d'inclure des chunks juste "moyennement" pertinents qui diluent le contexte.

En pratique pour du français avec MiniLM, un seuil entre 0.25 et 0.40 marche bien. Trop haut → on rate des chunks pertinents formulés différemment de la question. On ne mettra pas de threshold dans le MVP mais on en discutera en phase 2.

### Calibrer le threshold

Le bon threshold dépend de ton embedder, de ta langue, et de ton domaine. Méthode pour le calibrer :

1. Construis un eval set de 30-50 questions avec leur chunk-vérité.
2. Pour chaque question, calcule le score du chunk-vérité.
3. Le 10e percentile de ces scores te donne un threshold raisonnable (tu acceptes de rater 10% des cas).

## MMR — Maximal Marginal Relevance

Limite du top-k pur : si tu as 5 chunks qui disent presque la même chose dans le doc, ils vont tous remonter ensemble. Le LLM reçoit 5 fois la même info, et rate les nuances qui sont ailleurs dans le doc.

MMR résout ça en pénalisant la redondance.

### La formule

À chaque pick, on choisit le chunk qui maximise :

```
MMR(d_i) = λ · sim(d_i, q) − (1−λ) · max_{d_j ∈ S} sim(d_i, d_j)
```

Où :
- `q` : la question
- `d_i` : un chunk candidat encore non-choisi
- `S` : ensemble des chunks déjà choisis
- `sim(d_i, q)` : similarité entre le chunk et la question (notre métrique cosinus)
- `sim(d_i, d_j)` : similarité entre le chunk candidat et un chunk déjà choisi
- `λ ∈ [0, 1]` : poids relatif

Décodage : on récompense un chunk qui ressemble à la question, on pénalise un chunk qui ressemble trop à un chunk déjà choisi.

- `λ = 1` → MMR = top-k pur (pas de pénalité de redondance).
- `λ = 0` → on ne regarde plus la pertinence, juste la diversité (mauvaise idée).
- `λ = 0.7` → bon compromis par défaut.

### Implémentation

Tu récupères d'abord un large fetch (k_fetch = 20-30) via Qdrant, puis tu fais l'algorithme MMR côté Python pour réduire à k_final = 5.

```python
def mmr(query_vec, candidates, k=5, lam=0.7):
    """candidates: liste de (vec, payload, score) du top-k_fetch."""
    selected = []
    remaining = list(candidates)

    # Premier pick : le plus pertinent
    first = max(remaining, key=lambda c: c[2])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        def score(c):
            relevance = c[2]
            redundancy = max(cosine(c[0], s[0]) for s in selected)
            return lam * relevance - (1 - lam) * redundancy

        next_pick = max(remaining, key=score)
        selected.append(next_pick)
        remaining.remove(next_pick)

    return selected
```

Coût : O(k × k_fetch × d). Pour k=5, k_fetch=30, d=384 → ~58k flops, négligeable.

### Pourquoi Eigenmind n'utilise pas MMR

Eigenmind résout le même problème différemment, via les **Singular nodes** du graphe (phase 2). C'est plus élégant parce que la diversité est calculée *globalement* sur tout le corpus, pas seulement par rapport aux résultats déjà sélectionnés à chaque requête.

MMR reste un outil simple et efficace pour un RAG sans graphe.

## Rerankers — la qualité supérieure

Le retrieval HNSW + cosinus utilise des **bi-encoders** : la question et les chunks sont embeddés indépendamment, on compare leurs vecteurs. Rapide mais limité : le modèle ne voit jamais la question et le chunk ensemble.

Un **cross-encoder** prend la paire `(question, chunk)` en entrée simultanée et produit un score de pertinence direct. Beaucoup plus précis, mais coûteux : il faut un appel par paire à scorer.

### Le pipeline avec reranker

1. **Retrieve top-K large** (k_fetch = 30) avec bi-encoder + Qdrant. Rapide.
2. **Rerank** les K candidats avec un cross-encoder. Coûteux mais limité.
3. Garde le **top-k final** (k=5) selon le score cross-encoder.

Modèles populaires de rerankers en 2026 :
- `BAAI/bge-reranker-base` (open-source, 110M params, multilingue)
- `BAAI/bge-reranker-v2-m3` (open-source, qualité state-of-art)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (anglais, léger)
- Cohere Rerank (payant, API, excellent)
- Voyage Rerank (payant)

### Combien ça améliore ?

Sur des benchmarks de retrieval comme BEIR :
- Bi-encoder seul : nDCG@10 ≈ 0.40
- Bi-encoder + reranker BGE-base : nDCG@10 ≈ 0.50 (+25%)

Ça vaut le coup si la qualité est critique. Coût : ~50-200ms supplémentaires par requête.

### Quand le mettre en place

- MVP / apprentissage : non, simplicité d'abord.
- Si tu remarques que les bons chunks ne remontent pas en top-3 alors qu'ils sont en top-15 : oui, le reranker va aider.
- Phase 4-5 d'Eigenmind, candidat naturel d'ajout.

## Hybrid search : combiner vector et BM25

Les embeddings excellent en sémantique mais ratent les **termes rares** : noms propres, sigles, numéros, codes. Une recherche sur "Article R412-37" peut être mal embedded — le modèle n'a jamais vu ce code spécifique.

**BM25** (Best Matching 25) est la version moderne de TF-IDF, une recherche **par mots-clés** statistique. Excellent sur les termes rares, naze en sémantique. Complément parfait des embeddings.

### Reciprocal Rank Fusion (RRF)

Comment combiner les deux résultats ? Le standard est RRF :

```
score_RRF(d) = sum_over_results( 1 / (k + rank(d)) )
```

Où `rank(d)` est le rang du chunk dans une des deux recherches (vector ou BM25), et `k=60` est une constante d'amortissement. On somme sur toutes les recherches.

Le chunk qui est bien classé dans **les deux** recherches obtient un meilleur score que ceux qui ne sont bien que dans une.

### Implémentation Qdrant

Depuis Qdrant 1.10, support natif du sparse vector pour BM25 :

```python
from qdrant_client.models import VectorParams, SparseVectorParams

client.create_collection(
    "docs",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),  # BM25-style
    },
)

# upsert avec les deux types
# query avec fusion
client.query_points(
    "docs",
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=20),
        Prefetch(query=sparse_vec, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
)
```

Eigenmind ne l'utilise pas en MVP, mais c'est typiquement la première amélioration à apporter sur un RAG de prod.

## HyDE — Hypothetical Document Embeddings

Technique élégante : au lieu d'embedder la question (souvent courte et différente du style des documents), on demande à un LLM de **générer une réponse hypothétique** à la question, puis on embed cette réponse pour la recherche.

### Pipeline

1. Question : "Quel est le CA 2024 du groupe ?"
2. LLM génère : "Le groupe a réalisé un chiffre d'affaires de X milliards en 2024, en hausse de Y%..."  *(toutes les valeurs inventées)*
3. Embed cette réponse, recherche dans Qdrant.
4. Le chunk réel ressemble à la réponse hypothétique (mêmes mots, même style) → match supérieur.

### Pourquoi ça marche

Les chunks dans la DB sont du style "documentaire" (paragraphes affirmatifs). Les questions sont courtes et interrogatives. L'écart de style fait que l'embedding de la question matche imparfaitement les chunks. HyDE comble cet écart.

Coût : un appel LLM supplémentaire par requête (+ latence + tokens). Mais souvent net positif sur la qualité.

### Limites

- Si la question est sur un sujet hors-corpus, HyDE génère du texte plausible mais hors-sujet, et la recherche dérive.
- Pour des questions très factuelles ("Quel est le CA ?"), HyDE inventerait un chiffre — pas grave parce qu'on cherche du **style**, pas la vérité.

Eigenmind ne fait pas HyDE. Bon ajout potentiel.

## Query expansion

Variante encore plus simple : on demande au LLM de **reformuler** la question en plusieurs variantes, puis on cherche avec chaque variante et on fusionne (RRF).

```
Question : "CA 2024 du groupe"
Variantes générées :
- "Chiffre d'affaires 2024 de l'entreprise"
- "Revenu annuel 2024"
- "Total des ventes en 2024"
```

Chaque variante peut matcher des chunks différents (rédigés avec d'autres synonymes). Union via RRF.

Plus simple à implémenter que HyDE, gain de qualité similaire.

## Self-query retrieval

Pour des corpus structurés (chunks avec date, auteur, type...), on peut demander au LLM de **transformer la question en filtre + recherche** :

```
Question : "Quels rapports écrits par Alice en 2024 parlent du Cloud ?"

LLM extrait :
- filter: author=Alice, year=2024, doc_type=report
- query: "Cloud"
```

Le filtre Qdrant restreint l'espace de recherche, l'embedding cherche dans ce sous-espace. Beaucoup plus précis pour des questions multi-critères.

Implémentation : prompt-engineering + JSON output structuré + parsing.

## Métadonnées exploitées au retour

Quand on récupère un chunk, on a accès à tout son payload — donc `filename`, `page`, `text`. C'est ça qui permet de citer les sources dans la réponse finale ("d'après *rapport.pdf*, page 12, …"). Pas de sources = pas de RAG sérieux.

Le payload peut aussi servir à **dédupliquer** : si trois chunks du top-5 viennent du même filename à des pages consécutives, c'est probablement une longue section unique éclatée en chunks. Tu peux les fusionner avant de les passer au LLM.

## Métriques d'évaluation

Comment savoir si ton retrieval est bon ? Construis un eval set : `(question, chunk_id_attendu)` annoté à la main, 30-100 exemples. Puis mesure :

**Recall@k** — dans combien de cas le chunk attendu est dans le top-k retourné ?
```
recall@5 = (nb questions où chunk_attendu ∈ top_5) / (total questions)
```
Le KPI principal du retrieval. Vise >80% à k=5.

**MRR (Mean Reciprocal Rank)** — moyenne de 1/(rang du chunk attendu).
```
question 1 : chunk attendu en rang 2 → 1/2 = 0.5
question 2 : chunk attendu en rang 5 → 1/5 = 0.2
question 3 : chunk attendu pas trouvé → 0
MRR = (0.5 + 0.2 + 0) / 3 = 0.233
```
Pénalise plus fort si le bon chunk est loin. Vise >0.5.

**Precision@k** — dans le top-k, combien sont pertinents ? (suppose plusieurs chunks pertinents par question). Plus rare à évaluer parce que demande plus d'annotation.

**nDCG@k** — Normalized Discounted Cumulative Gain. Le standard académique. Gradue la pertinence (très/moyennement/peu pertinent) et pondère par le rang. Plus riche mais plus coûteux à annoter.

## Le test "needle in a haystack"

Un test diagnostic : injecte volontairement dans ton corpus un chunk unique avec une info bizarre ("Le mot de passe magique est XYZ123"). Pose la question correspondante. Le système doit retrouver ce chunk dans le top-1.

Si non, ton retrieval a un problème :
- Ton chunk est mal embeddé → vérifie qu'il est bien indexé.
- Ta question matche mal → essaie une reformulation.
- Le bruit du corpus écrase le signal → besoin de filtres ou rerankers.

C'est un debug essentiel.
