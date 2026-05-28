# 1.2 — Sentence Embeddings

## Intuition de départ

Avant le calcul, l'idée. On veut transformer chaque morceau de texte en un point dans un espace géométrique tel que :

- des textes au sens proche tombent **près** les uns des autres,
- des textes au sens éloigné tombent **loin**,
- la "distance" se mesure avec une opération simple (un produit scalaire).

Une fois ce mapping `texte → point` fait, beaucoup de problèmes deviennent géométriques :
- **Recherche** = "quels points sont les plus proches de la requête ?"
- **Clustering** = "quels paquets de points y a-t-il ?"
- **Détection d'atypique** = "quels points sont isolés ?" (c'est ce qu'on fera en Phase 2 avec les Singular nodes)
- **Pivots structurels** = "quels points relient deux régions denses ?" (Hinge nodes)

Tout l'enjeu est donc la qualité du mapping. C'est là que les transformers entrent en scène.

## De quoi on parle exactement

Un embedding de phrase est une fonction qui prend une string et renvoie un vecteur dense de dimension fixe (384 dans notre cas). Deux propriétés clés :

1. **Sémantique** : deux phrases au sens proche → deux vecteurs proches.
2. **Continuité** : de petites variations de formulation → de petites variations vectorielles.

C'est différent d'un *word embedding* (Word2Vec, GloVe) qui n'embeddait que des mots isolés sans contexte. Un sentence embedding tient compte du contexte global de la phrase grâce à un transformer.

**Pourquoi "dense" ?** Un vecteur dense a (presque) toutes ses composantes non nulles, par opposition à un vecteur *sparse* (TF-IDF, BM25) qui a un 0 partout sauf pour les mots présents. Dense = sémantique compressée ; sparse = match lexical exact. Les deux ont leurs forces (on verra l'hybride lexical+dense plus tard).

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

### Exemple numérique minimal de self-attention

Pour ancrer la formule, prenons une mini-séquence de 3 tokens : `["le", "chat", "dort"]`. Imaginons des vecteurs Q, K, V de dimension `d = 2` (au lieu de 64 en vrai). On suppose qu'on les a déjà calculés :

```
Q = [[1, 0],     K = [[1, 0],     V = [[10, 0],
     [0, 1],          [1, 1],          [0, 10],
     [1, 1]]          [0, 1]]          [5,  5]]
```

**Étape 1** : `Q × K^T` (matrice 3×3 de scores bruts) :
```
[[1*1+0*0, 1*1+0*1, 1*0+0*1],     [[1, 1, 0],
 [0*1+1*0, 0*1+1*1, 0*0+1*1], =    [0, 1, 1],
 [1*1+1*0, 1*1+1*1, 1*0+1*1]]      [1, 2, 1]]
```

**Étape 2** : on divise par `√d = √2 ≈ 1.41`. Ce scaling empêche les scores d'exploser quand d est grand (sinon softmax sature et le gradient meurt).

**Étape 3** : softmax ligne par ligne. Pour la 3e ligne `[1/√2, 2/√2, 1/√2]` ≈ `[0.71, 1.41, 0.71]` :

```
softmax = [e^0.71, e^1.41, e^0.71] / somme ≈ [0.27, 0.46, 0.27]
```

Cette ligne dit : le token "dort" s'intéresse pour 46% à "chat", 27% à "le", 27% à lui-même.

**Étape 4** : on multiplie ces poids par V pour obtenir la nouvelle représentation de "dort" :

```
0.27 × [10, 0] + 0.46 × [0, 10] + 0.27 × [5, 5] ≈ [4.05, 5.95]
```

Voilà la nouvelle représentation de "dort", enrichie par le contexte. Tout le reste du transformer empile ce même mécanisme couche après couche, avec des résidus et des MLP entre.

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

### Tokenizers : WordPiece, BPE, SentencePiece

Trois familles de sous-mot tokenizers — utile à connaître car le choix affecte ce que le modèle "voit".

| Tokenizer | Modèles | Comment ça marche | Particularité |
|---|---|---|---|
| **WordPiece** | BERT, MiniLM | Démarre du vocab des caractères, fusionne récursivement les paires les plus fréquentes pour maximiser la vraisemblance. Marque les continuations avec `##`. | Sensible aux espaces : "Paris" et " Paris" peuvent tokeniser différemment. |
| **BPE** (Byte-Pair Encoding) | GPT-2, RoBERTa | Idem mais maximise la compression (paires les plus fréquentes). Pas de marqueur de continuation. | Byte-level BPE (GPT) gère n'importe quel UTF-8 sans OOV. |
| **SentencePiece** | T5, Llama, Qwen | Traite l'espace comme un caractère normal (`▁`). Indépendant de la langue / segmentation préalable. | Le seul vraiment language-agnostic. Idéal multilingue. |

Pour MiniLM (WordPiece), un mot français accentué peut éclater en plusieurs sous-mots, ce qui dilue l'attention. C'est une des raisons pour lesquelles MiniLM monolingue est moyen sur du français — le tokenizer lui-même est sous-optimal.

### Pooling

Le pooling est crucial. Variantes possibles :

- **CLS pooling** — utiliser uniquement le vecteur du token spécial `[CLS]`. Classique en classification BERT. Moins bon pour la similarité.
- **Mean pooling** (notre cas) — moyenne des tokens. Le défaut de sentence-transformers.
- **Max pooling** — max par dimension. Rarement utilisé.
- **Last-token pooling** — utilisé pour les embedding models décodeurs (E5-Mistral, GTE-Qwen). On prend l'état du dernier token.

Mean est ce qui donne les meilleurs scores sur les benchmarks de similarité avec des encoders BERT-style.

## Géométrie de l'espace d'embedding

Un point souvent passé sous silence : **un espace 384-d n'est pas un espace 3-d en plus gros**. Quelques propriétés contre-intuitives à connaître.

### Anisotropie

Empiriquement, les embeddings de BERT/MiniLM ne remplissent pas uniformément la sphère unité : ils se concentrent dans un **cône étroit**. Conséquence : même deux textes très différents ont un cosinus assez élevé (souvent 0.3-0.5), pas 0. Cette anisotropie est un biais d'entraînement, pas une propriété fondamentale.

Sentence-transformers (entraîné avec une loss contrastive) atténue ce problème — c'est une des raisons pour lesquelles `all-MiniLM-L6-v2` est bien meilleur que le BERT original sur les tâches de similarité, à architecture quasi équivalente.

### Hubness

En haute dimension, certains points deviennent **hubs** : ils apparaissent dans le top-K des plus proches voisins de beaucoup d'autres points, sans raison sémantique évidente. C'est un effet purement géométrique de la haute dimension (le phénomène disparaît en 3-d).

Symptôme : "ce chunk générique revient tout le temps dans mes résultats, peu importe la question". Mitigations : Mutual K-NN (deux nœuds connectés ssi chacun est dans les K voisins de l'autre — on l'utilisera implicitement avec la symétrisation par union en Phase 2), ou re-scoring (MMR, cross-encoder).

### Pourquoi 384 et pas 50 ou 4096 ?

- En dessous de ~100, on perd trop d'information sémantique fine.
- Au-dessus de ~1024, les gains s'amenuisent et le coût (stockage, latence) explose.
- 384, 512, 768, 1024 sont les sweet spots actuels.

Anecdotique : Matryoshka (voir plus bas) permet maintenant d'avoir un seul modèle qui sert plusieurs dimensions.

## Pourquoi normaliser ?

Quand les vecteurs sont normalisés, **cosine similarity = produit scalaire**. C'est massivement plus rapide à calculer (juste un dot product, sans division par les normes). Toute la chaîne RAG suppose cette normalisation.

Formule de la similarité cosinus :

    cos(a, b) = (a · b) / (||a|| × ||b||)

Avec a et b normalisés (||a|| = ||b|| = 1), ça se simplifie en :

    cos(a, b) = a · b

Range théorique : -1 à 1. En pratique sur du texte naturel, les valeurs descendent rarement sous 0 — deux textes vraiment opposés sémantiquement donnent ~0, pas -1. C'est l'anisotropie résiduelle dont on parlait plus haut.

## Pourquoi cosinus et pas euclidienne ?

La distance euclidienne pénalise la magnitude. Deux textes sur le même sujet mais de longueurs différentes peuvent avoir des normes différentes — la direction du vecteur reste la même, mais la longueur change. Le cosinus, lui, ne regarde que la direction.

Avec des vecteurs L2-normalisés, euclidienne et cosinus sont équivalents à une transformation près : `dist_euclid² = 2 - 2·cos`. Donc choisir l'un ou l'autre n'a pas d'impact sur le **ranking** des résultats. Mais cosinus est plus interprétable (entre 0 et 1 pour des vecteurs unitaires positifs).

## Que veulent dire les scores en pratique ?

Une "échelle" empirique pour `all-MiniLM-L6-v2`, basée sur l'observation. Les valeurs varient selon le modèle mais l'ordre de grandeur tient :

| Cosinus | Interprétation typique | Exemple |
|---|---|---|
| **> 0.85** | Quasi-duplicat / paraphrase serrée | "Le chat dort." ↔ "Le chat est endormi." |
| **0.65 – 0.85** | Même sujet, formulation différente | "Le chat dort." ↔ "Mon félin fait la sieste." |
| **0.45 – 0.65** | Sujet connexe / même domaine | "Le chat dort." ↔ "Les félins chassent la nuit." |
| **0.25 – 0.45** | Lien sémantique faible / vague | "Le chat dort." ↔ "Le chien aboie dans le jardin." |
| **< 0.25** | Topics étrangers | "Le chat dort." ↔ "La Bourse de Paris a clôturé en hausse." |

Les seuils du retrieval hybride d'Eigenmind (cos > 0.30 pour ajouter un Singular/Hinge/Theta) tombent volontairement dans la zone "lien faible mais pas nul" — on veut récupérer des chunks structurellement importants même s'ils ne sont pas le match parfait.

**Important** : ces seuils sont calibrés par modèle. Si on swap pour BGE ou E5, l'échelle se décale (E5 produit des cosinus en moyenne plus élevés que MiniLM). À chaque changement de modèle, refaire l'étalonnage avant de réutiliser les mêmes seuils.

## Failure modes connus des sentence embeddings

Les sentence embeddings ne sont pas magiques. Cas où ils plantent :

- **Négation** — "Le contrat est valide" et "Le contrat n'est pas valide" sortent souvent un cosinus > 0.85. Le modèle voit les mêmes mots avec la même structure et ignore le "ne…pas". Connu, pas vraiment résolu en 2024.
- **Nombres** — "facture de 100 €" et "facture de 100 000 €" sont quasi identiques pour le modèle. Les nombres sont tokenisés comme des suites de chiffres sans représentation numérique. Si ton domaine est numéricky (finance, médical), prévoir du post-processing.
- **Entités nommées rares** — "Eigenmind" tokenise en `["eig", "##en", "##mind"]` qui n'a aucun ancrage sémantique. Solution : fine-tuning ou ajouter des tokens au vocabulaire.
- **Code source** — MiniLM est entraîné sur du texte naturel. Pour embedder du Python/SQL, prendre un modèle dédié (`CodeBERT`, `e5-code`).
- **Très longs textes** — la troncature à 256 tokens (~1000 caractères) coupe brutalement. C'est précisément pour ça qu'on chunk en amont (voir 1.3).
- **Langues sous-représentées** — MiniLM est ~95% anglais. Tout passage français est dégradé. Multi-lingual models obligatoires si corpus français.

Connaître ces failure modes oriente le debug : si un retrieval rate de manière systématique sur des chunks contenant un nombre clé, c'est probablement le modèle, pas le retrieval.

## L'entraînement : la loss contrastive

Comment force-t-on un modèle à mapper des phrases proches sur des vecteurs proches ? Par une **loss contrastive**.

Le dataset d'entraînement consiste en triplets `(anchor, positive, negative)` ou paires `(anchor, positive)` :

- **anchor** : une phrase de référence.
- **positive** : une phrase au sens proche (paraphrase, traduction, paire question-réponse...).
- **negative** : une phrase au sens éloigné.

La loss force `cos(anchor, positive) > cos(anchor, negative)` avec une marge.

### In-batch negatives

La variante moderne **MultipleNegativesRankingLoss** utilise les autres exemples du même batch comme négatifs implicites. Si on a un batch de 64 paires `(q_i, p_i)`, chaque `q_i` a un positif `p_i` et 63 négatifs gratuits `p_j` pour `j ≠ i`. Gain énorme en efficacité : on apprend sur 64×64 = 4096 comparaisons au lieu de 64.

### Hard negatives

Les in-batch negatives sont en moyenne *faciles* (deux paires aléatoires sont en général sémantiquement éloignées). Les modèles modernes ajoutent du **hard negative mining** : on cherche activement des négatifs qui *ressemblent* à des positifs mais ne le sont pas (par ex., même topic mais réponse à une question différente). C'est ce qui pousse vraiment le modèle à apprendre les distinctions fines.

`all-MiniLM-L6-v2` a été entraîné sur **>1 milliard de paires** (Reddit, Stack Exchange, S2ORC, WikiAnswers, et bien d'autres) avec cette loss. C'est le scale qui fait la qualité, pas l'architecture en soi.

## Bi-encoders vs cross-encoders

Distinction structurante pour la suite. Sentence-transformers couvre les deux.

### Bi-encoder (notre cas)

`embed(query)` et `embed(chunk)` sont calculés **indépendamment**. Puis on compare via cosinus. Conséquence cruciale : on peut **pré-calculer** tous les embeddings du corpus une fois pour toutes, puis ne faire qu'un dot product par requête.

```
query ─→ embed ─→ q
                  │
chunk ─→ embed ─→ c  ─→  cos(q, c)
```

Coût online : 1 forward pass (la query) + N dot products. Scale à des millions de chunks sans souci.

### Cross-encoder (reranker)

On passe `[query, chunk]` *concaténés* dans le modèle. Le modèle voit les deux ensemble et sort un score de pertinence directement. Beaucoup plus précis (les deux textes "se parlent" via l'attention), mais **non factorisable** : il faut un forward pass par paire (query, chunk).

```
[query | chunk] ─→ modèle ─→ score
```

Coût online : N forward passes par requête → impossible à grande échelle.

**Pattern usuel** : bi-encoder pour le retrieval top-K (rapide, large rappel), puis cross-encoder pour reranker les K candidats (lent mais précis). On y reviendra probablement quand on raffinera le retrieval en phase 4-5.

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

**Avertissement** : MTEB est saturé. Beaucoup de modèles récents over-fittent sur le benchmark (data leakage des datasets MTEB dans le training set). Un score MTEB élevé ne garantit pas la qualité sur ton domaine. D'où l'importance de ton propre eval set (voir plus bas).

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

Important si tu changes de modèle : **il faut tout ré-embedder**. Les vecteurs de deux modèles différents sont dans des espaces différents et ne sont pas comparables. C'est pour ça qu'en Phase 4 on ajoutera le nom du modèle dans le payload Qdrant — pour pouvoir détecter une incohérence collection / modèle au démarrage.

## Matryoshka embeddings

Innovation 2024 popularisée par OpenAI (`text-embedding-3-large`) et reprise en open-source (Nomic, BGE). L'idée : entraîner le modèle pour que les **k premières dimensions** d'un embedding 1024-d soient déjà un embedding utilisable de dim k.

Tu peux ainsi tronquer un vecteur de 1024 à 512 ou 256 pour économiser stockage et latence, sans avoir à re-train. Légère perte de qualité mais beaucoup plus flexible.

`all-MiniLM-L6-v2` n'est pas Matryoshka. À garder en tête si tu passes à BGE ou OpenAI plus tard.

## Quantization : int8 et binary embeddings

Un float32 384-d pèse `384 × 4 = 1536 octets` par chunk. À 1M de chunks → ~1.5 Go. Sur 10M, ~15 Go. La RAM devient le bottleneck.

Solutions :

- **int8** — quantiser chaque composante sur 8 bits au lieu de 32. Compression ×4, perte de qualité < 1% sur MTEB. Supporté nativement par Qdrant (`quantization_config`).
- **Binary** — chaque composante devient un bit (1 si > 0, 0 sinon). Compression ×32. Distance = Hamming (XOR + popcount, extrêmement rapide). Perte de qualité ~5-10%, mais souvent acceptable si on rerank ensuite.
- **Product Quantization** — découper le vecteur en sous-vecteurs et quantiser chacun via un codebook. Compromis intermédiaire utilisé par FAISS.

Pour Eigenmind à l'échelle MVP (< 100K chunks), on reste en float32 pour simplicité. À envisager quand le corpus dépasse 1M.

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
- **nDCG@10** : si plusieurs chunks sont pertinents par question, gradué par position.

Si recall@5 < 70%, ton modèle d'embedding est inadapté à ton domaine. Solutions :
1. **Changer de modèle** (BGE, E5) — solution la plus simple.
2. **Fine-tuner** sur ton domaine — sentence-transformers a un trainer dédié, ~1h sur GPU avec 1000 paires.
3. **Hybrid search** — combiner dense (embeddings) et sparse (BM25). Très efficace quand le corpus a beaucoup de termes techniques / noms propres mal embeddés.
4. **Reranker** — ajouter un cross-encoder en seconde passe sur les K candidats.

On reverra l'évaluation en phase 4.

## Lien avec la suite

Trois choses à garder en tête pour les étapes suivantes :

- **1.3 — Chunking** : ce qu'on embedde n'est *jamais* une phrase au sens grammatical, mais un chunk de ~500 caractères. Le mean pooling sur un chunk hétérogène (plusieurs paragraphes, plusieurs sujets) dilue le signal. Le chunking est donc la moitié du jeu — un chunking foireux ruine les meilleurs embeddings.
- **1.4 — Stockage Qdrant** : les vecteurs partent en base avec une `distance=Cosine`. Cette distance suppose la normalisation L2 — on la fait côté `embed_text.py` plutôt que d'attendre que Qdrant le fasse, pour rester explicite.
- **Phase 2 — Graphe spectral** : tout le graphe k-NN est construit sur ces embeddings. Si l'embedding est mauvais, le graphe l'est aussi, et les Singular/Hinge/Theta nodes ne veulent rien dire. Tester la qualité du retrieval dense avant de s'engager dans la couche spectrale.
