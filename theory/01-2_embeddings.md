# 1.2 — Sentence Embeddings

## De quoi on parle exactement

Un embedding de phrase est une fonction qui prend une string et renvoie un vecteur dense de dimension fixe (384 dans notre cas). Deux propriétés clés :

1. **Sémantique** : deux phrases au sens proche → deux vecteurs proches.
2. **Continuité** : de petites variations de formulation → de petites variations vectorielles.

C'est différent d'un *word embedding* (Word2Vec, GloVe) qui n'embeddait que des mots isolés sans contexte. Un sentence embedding tient compte du contexte global de la phrase grâce à un transformer.

## Le transformer en 2 minutes

Pour comprendre les embeddings, il faut comprendre l'architecture qui les produit. Un transformer est composé de **couches répétées**, chacune contenant deux blocs :

1. **Self-attention** — chaque token "regarde" tous les autres tokens de la séquence et décide lesquels sont pertinents pour comprendre son rôle. C'est l'innovation clé de l'article "Attention is All You Need" (2017). Sans elle, on en serait encore aux RNN.
2. **Feed-forward network** — un MLP appliqué à chaque position indépendamment, qui raffine la représentation.

Entre les deux : des **residual connections** et de la **normalization** pour stabiliser l'entraînement.

### L'attention en plus de détail

Pour chaque token, on calcule trois vecteurs : **Query** (Q), **Key** (K), **Value** (V). Tous obtenus par multiplication matricielle du vecteur d'entrée avec trois matrices apprises.

L'attention de chaque token est :

```
attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

En clair : pour chaque token (chaque Q), on calcule un score avec tous les K (à quel point ce token est "intéressé" par les autres), on normalise via softmax, et on en fait une combinaison pondérée des V. La sortie est une nouvelle représentation du token, enrichie par le contexte qu'il a jugé pertinent.

C'est répété en **multi-head** : on apprend plusieurs systèmes Q/K/V en parallèle (typiquement 8 ou 12 "têtes"), chacun captant un type différent de relation (syntaxique, sémantique, longue distance...).

### Position embeddings

L'attention pure est *permutation-invariante* : "le chat mange la souris" et "la souris mange le chat" donneraient les mêmes attention scores. On rajoute donc des **position embeddings** au vecteur de chaque token : sinusoïdaux (BERT, original), appris (GPT), ou relatifs (T5, RoPE pour Llama et Qwen). MiniLM utilise des positions apprises.

## Comment un sentence embedding est calculé

Pipeline complet sur `all-MiniLM-L6-v2` (qui est un BERT-style encoder réduit) :

1. **Tokenization** — la string est découpée en sous-mots via WordPiece. "transformer" devient `["transform", "##er"]`. Les `##` indiquent une continuation. Vocabulary size ~30k.
2. **Padding / truncation** — la séquence est tronquée à 256 tokens si plus longue, paddée à droite si plus courte. Un masque d'attention indique quels tokens sont réels vs padding.
3. **Embedding lookup** — chaque token id est mappé à un vecteur appris de dim 384.
4. **6 transformer layers** — chaque layer applique self-attention + FFN. À la sortie, chaque token a son propre vecteur 384-d enrichi de tout le contexte de la phrase.
5. **Mean pooling** — on moyenne tous les vecteurs de tokens, pondérés par le masque d'attention (le padding est exclu). Résultat : un seul vecteur 384-d pour toute la phrase.
6. **L2 normalization** — on divise par la norme pour avoir ||v|| = 1.

Le pooling est crucial. Variantes possibles :

- **CLS pooling** — utiliser uniquement le vecteur du token spécial `[CLS]`. Classique en classification BERT. Moins bon pour la similarité.
- **Mean pooling** (notre cas) — moyenne des tokens. Le défaut de sentence-transformers.
- **Max pooling** — max par dimension. Rarement utilisé.

Mean est ce qui donne les meilleurs scores sur les benchmarks de similarité.

## Pourquoi normaliser ?

Quand les vecteurs sont normalisés, **cosine similarity = produit scalaire**. C'est massivement plus rapide à calculer (juste un dot product, sans division par les normes). Toute la chaîne RAG suppose cette normalisation.

Formule de la similarité cosinus :

    cos(a, b) = (a · b) / (||a|| × ||b||)

Avec a et b normalisés (||a|| = ||b|| = 1), ça se simplifie en :

    cos(a, b) = a · b

Range théorique : -1 à 1. En pratique sur du texte naturel, les valeurs descendent rarement sous 0 — deux textes vraiment opposés sémantiquement donnent ~0, pas -1.

## Pourquoi cosinus et pas euclidienne ?

La distance euclidienne pénalise la magnitude. Deux textes sur le même sujet mais de longueurs différentes peuvent avoir des normes différentes — la direction du vecteur reste la même, mais la longueur change. Le cosinus, lui, ne regarde que la direction.

Avec des vecteurs L2-normalisés, euclidienne et cosinus sont équivalents à une transformation près : `dist_euclid² = 2 - 2·cos`. Donc choisir l'un ou l'autre n'a pas d'impact sur le **ranking** des résultats. Mais cosinus est plus interprétable (entre 0 et 1 pour des vecteurs unitaires positifs).

## L'entraînement : la loss contrastive

Comment force-t-on un modèle à mapper des phrases proches sur des vecteurs proches ? Par une **loss contrastive**.

Le dataset d'entraînement consiste en triplets `(anchor, positive, negative)` ou paires `(anchor, positive)` :

- **anchor** : une phrase de référence.
- **positive** : une phrase au sens proche (paraphrase, traduction, paire question-réponse...).
- **negative** : une phrase au sens éloigné.

La loss force `cos(anchor, positive) > cos(anchor, negative)` avec une marge. La variante moderne **MultipleNegativesRankingLoss** utilise les autres exemples du même batch comme négatifs implicites — gain énorme en efficacité.

`all-MiniLM-L6-v2` a été entraîné sur **>1 milliard de paires** (Reddit, Stack Exchange, S2ORC, WikiAnswers, et bien d'autres) avec cette loss. C'est le scale qui fait la qualité, pas l'architecture en soi.

## MTEB — le benchmark à connaître

**MTEB** (Massive Text Embedding Benchmark) est le standard pour évaluer les modèles d'embedding. Il agrège **56 tâches** sur 8 catégories :

- **Retrieval** (15 tâches) — la plus importante pour nous
- **Reranking**
- **Clustering**
- **Pair Classification**
- **Classification**
- **STS** (Semantic Textual Similarity)
- **Summarization**
- **Bitext Mining** (multilingue)

Chaque modèle obtient un score moyen, et tu peux regarder les sous-scores par catégorie. La leaderboard sur Hugging Face (`mteb/leaderboard`) est mise à jour en continu.

Quelques chiffres clés pour notre choix :

- `all-MiniLM-L6-v2` : MTEB ~56 — bon pour sa taille (22M params).
- `all-mpnet-base-v2` : MTEB ~57.8 — mieux mais 5x plus gros (110M).
- `bge-large-en-v1.5` : MTEB ~63 — top open-source, 335M params.
- `text-embedding-3-large` (OpenAI, payant) : MTEB ~64.6.
- `e5-mistral-7b-instruct` : MTEB ~66 — mais 7B params, lourd.

L'écart entre MiniLM (56) et BGE-large (63) est notable mais pas énorme. Pour un MVP qui tourne en local, MiniLM est le compromis raisonnable. En prod sur GPU, BGE-large ou supérieur.

## Choix du modèle : pourquoi `all-MiniLM-L6-v2` pour Eigenmind

- **22M paramètres** — petit, tourne sur CPU en quelques millisecondes par phrase.
- **384 dimensions** — compromis taille/qualité.
- Entraîné sur **>1 milliard de paires** avec une loss contrastive.
- Performance proche des modèles bien plus gros pour les tâches de retrieval.

**Limite principale** : entraîné majoritairement sur de l'anglais. Pour du multilingue, des modèles alternatifs :

| Modèle | Dim | Multilingue | MTEB-FR retrieval |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Non | ~30 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Oui (50 langues) | ~42 |
| `multilingual-e5-base` | 768 | Oui (100 langues) | ~48 |
| `BAAI/bge-m3` | 1024 | Oui (100+) + dense/sparse/colbert | ~54 |
| `intfloat/multilingual-e5-large-instruct` | 1024 | Oui + instruction-tuned | ~57 |

Pour un Eigenmind sérieusement utilisé en français, **`multilingual-e5-base`** ou **`bge-m3`** seraient de meilleurs choix. On reste sur MiniLM en MVP pour la légèreté CPU, mais c'est un swap à envisager.

Important si tu changes de modèle : **il faut tout ré-embedder**. Les vecteurs de deux modèles différents sont dans des espaces différents et ne sont pas comparables.

## Matryoshka embeddings

Innovation 2024 popularisée par OpenAI (`text-embedding-3-large`) et reprise en open-source (Nomic, BGE). L'idée : entraîner le modèle pour que les **k premières dimensions** d'un embedding 1024-d soient déjà un embedding utilisable de dim k.

Tu peux ainsi tronquer un vecteur de 1024 à 512 ou 256 pour économiser stockage et latence, sans avoir à re-train. Légère perte de qualité mais beaucoup plus flexible.

`all-MiniLM-L6-v2` n'est pas Matryoshka. À garder en tête si tu passes à BGE ou OpenAI plus tard.

## Caching et performance

Au premier appel, sentence-transformers télécharge le modèle (~90 MB pour MiniLM) depuis Hugging Face dans `~/.cache/huggingface/hub/`. Les appels suivants réutilisent ce cache. Pour bosser hors-ligne après le premier download : `HF_HUB_OFFLINE=1`.

### Performances réelles sur CPU

Sur un Mac M3 (notre cas) :

| Modèle | Latence 1 phrase | Throughput batch=32 |
|---|---|---|
| `all-MiniLM-L6-v2` | ~15 ms | ~800 phrases/s |
| `all-mpnet-base-v2` | ~50 ms | ~250 phrases/s |
| `bge-large-en-v1.5` | ~120 ms | ~80 phrases/s |
| `multilingual-e5-base` | ~60 ms | ~200 phrases/s |

Pour 10 000 chunks à indexer : MiniLM en ~12 secondes, BGE-large en ~2 minutes. À ton échelle, peu importe lequel. À l'échelle de 1M de chunks, c'est 20 minutes vs 3.5 heures.

### Batching

Pour embed un seul texte, c'est rapide. Pour 10 000 chunks, on **batch** : on passe 32 ou 64 phrases à la fois au modèle, qui les traite en parallèle. Sans batching, on a N× l'overhead d'inférence (chargement des tenseurs, allocation mémoire) ; avec batching c'est ~50-150× plus rapide.

Sentence-transformers fait le batching automatiquement quand on lui passe une liste, avec une `batch_size` configurable. Le défaut est 32.

### MPS sur Apple Silicon

PyTorch supporte MPS (Metal Performance Shaders) sur Mac M1/M2/M3/M4 pour accélérer l'inférence via le GPU intégré. Pour MiniLM, le gain est modeste (~2x) parce que le modèle est petit. Pour les modèles plus gros (BGE-large), MPS peut donner 5-10x.

```python
model = SentenceTransformer(MODEL_NAME, device="mps")  # au lieu de "cpu"
```

Attention : MPS a parfois des bugs sur certaines ops, et la consommation RAM peut être plus élevée. On garde `device="cpu"` par défaut en MVP, on testera MPS en phase 5.

## Pourquoi delayed loading dans Eigenmind

Le modèle prend ~200 MB de RAM une fois chargé. Sur une VM 4 GB, on ne le charge donc qu'au moment où on en a besoin (premier embed appelé) plutôt qu'à l'import du module. C'est le pattern *lazy initialization*.

Code typique :
```python
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(...)
    return _model
```

C'est ce qu'on a fait dans `embed_text.py`. En phase 5, on ajoutera aussi un `gc.collect()` après les gros traitements pour libérer la RAM rapidement.

## Évaluer son propre setup

À un moment tu te demanderas : "est-ce que mes embeddings sont bons pour mon corpus ?" La réponse honnête : il faut **un eval set**.

Construis 20-50 paires `(question, chunk_attendu)` à la main à partir de ton corpus. Mesure :
- **Recall@5** : dans combien de cas le chunk attendu est dans le top-5 ?
- **MRR** : Mean Reciprocal Rank = moyenne de 1/(rang du chunk attendu).

Si recall@5 < 70%, ton modèle d'embedding est inadapté à ton domaine. Solutions : changer de modèle, fine-tuner sur ton domaine, ajouter du BM25 en hybrid search.

On reverra l'évaluation en phase 4.
