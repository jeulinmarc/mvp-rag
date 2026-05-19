# 1.4 — Stockage des chunks dans Qdrant

## Le workflow d'ingestion

Une fois qu'on a une liste de chunks (sortie de `load_pdf.py`), l'ingestion en vector DB suit toujours les mêmes 4 étapes :

1. **Créer la collection** (une fois) avec sa dimension et sa distance.
2. **Embedder les chunks** en batch — récupérer une matrice (N, 384).
3. **Construire les "points"** en associant chaque vecteur à son payload.
4. **Upsert** : insérer ou mettre à jour ces points dans la collection.

"Upsert" = "update or insert". Si un point avec le même id existe déjà, il est écrasé ; sinon il est créé. C'est l'opération idempotente par excellence — tu peux relancer l'ingestion 10 fois, le résultat est le même.

## Anatomie d'un PointStruct Qdrant

```python
PointStruct(
    id=42,                              # int ou str (UUID typique)
    vector=[0.12, -0.04, ..., 0.07],    # liste de floats de la dim de la collection
    payload={                           # dict de métadonnées libre
        "filename": "rapport.pdf",
        "page": 12,
        "text": "Le chiffre d'affaires...",
    }
)
```

Le `payload` est searchable par filtre (égalité, range, in/not in) — c'est ce qui permettra le namespacing multi-user et le Smart Resume plus tard.

## Choix du type d'id

Trois options principales :

**int auto-incrémenté** — simple, mais pas de garantie d'unicité si tu réingères depuis plusieurs sources. Risque de collisions à long terme.

**UUID** — universellement unique (`uuid.uuid4()`), mais **non-déterministe** : chaque ingestion du même chunk crée un nouveau point. Tu finis avec des doublons si tu réingères.

**Hash déterministe** (notre choix) — `hash(filename + page + chunk_index)`. Même chunk → même id. C'est ça qui rend l'upsert idempotent : relancer l'ingestion ne duplique pas les points, elle les écrase à l'identique.

### Pourquoi pas `hash()` Python

`hash()` est randomisé entre les runs Python (protection contre les attaques DoS sur les dicts). Il te donnerait des ids différents à chaque lancement. **SHA-1 / SHA-256** sont déterministes : mêmes input → mêmes output, toujours.

Qdrant accepte les ids en `int` (jusqu'à 64-bit non signé) ou `str` (UUID au format string). On va utiliser un hash int 64-bit dérivé d'une string déterministe.

### Risque de collision

SHA-1 sur 64 bits → 2^64 valeurs possibles. Probabilité de collision sur 1M de chunks ≈ 2.7e-8. Négligeable. Pour un milliard de chunks tu commences à voir des collisions (~5%), il faudrait alors passer à 128 bits.

## Distance metric : Cosine

Au moment de créer la collection, on doit spécifier la métrique de distance. Trois options :

- **Cosine** (notre choix) — standard pour les embeddings de texte.
- **Dot** — produit scalaire pur, équivalent à Cosine si les vecteurs sont normalisés. Plus rapide en théorie de quelques %.
- **Euclid** — distance euclidienne, peu utilisée en NLP.

Comme nos vecteurs sont déjà L2-normalisés par `sentence-transformers` (`normalize_embeddings=True`), Cosine et Dot donneraient les mêmes scores. On garde Cosine pour rester compatible avec des vecteurs potentiellement non-normalisés provenant d'autres sources (par exemple si quelqu'un ajoute des embeddings d'images avec un autre modèle).

**Manhattan / L1** existe aussi mais quasi jamais pour du texte.

## Batching de l'upsert

Pour 50 chunks, on peut tout envoyer d'un coup. Pour 50 000 chunks, on doit batcher — typiquement 64 ou 128 points par requête. Trois raisons :

1. **Timeout HTTP** — par défaut le client Qdrant timeout à 60s. Un gros upsert peut dépasser.
2. **RAM côté client** — construire 50k PointStruct en mémoire d'un coup peut consommer plusieurs Go.
3. **Granularité des erreurs** — si une requête échoue, tu sais quels points ont été affectés.

On va prévoir le batching dès le départ, ça coûte 3 lignes.

### Wait vs non-wait

Qdrant supporte `wait=False` sur l'upsert : la requête revient immédiatement et l'indexation se fait en arrière-plan. Plus rapide pour des très gros batches. Inconvénient : si une recherche arrive avant la fin de l'indexation, certains points ne sont pas encore visibles.

Pour le MVP on garde `wait=True` (défaut), c'est plus prévisible.

## Recreate vs ensure

Deux philosophies à l'initialisation d'une collection :

**`recreate_collection`** (raccourci `delete + create`) — efface tout et repart de zéro. Pratique en dev quand tu changes ton schéma, dangereux en prod où tu perds tes données.

**get + create-if-absent** (le pattern *ensure*) — vérifier si la collection existe, ne créer que si absente. Idempotent, safe.

On part sur le pattern *ensure* pour ne pas effacer le travail entre deux runs.

## Sanity check : vérifier la dimension

Si tu changes de modèle d'embedding et que la nouvelle dim ne match pas celle de la collection existante, Qdrant refuse l'upsert avec une erreur claire. Mais c'est plus propre de vérifier côté client avant d'envoyer — d'où une assertion au début de l'ingestion.

Idéalement, en plus, tu pourrais stocker la dimension et le nom du modèle dans une **collection-level metadata** Qdrant — ça t'évite tout doute à l'avenir.

## Index HNSW et seuils d'indexation

Qdrant ne construit pas l'index HNSW à chaque upsert — ce serait trop coûteux. Il accumule les points dans un "segment indexé linéairement" (équivalent d'un brouillon), et déclenche la construction HNSW quand un segment dépasse un seuil (par défaut 20 000 points ou 4h d'inactivité).

Conséquences :

- **Recherche pendant l'ingestion** : Qdrant cherche dans HNSW + scan linéaire des segments non encore indexés. Lent si tu fais des recherches pendant que tu ingères massivement.
- **Pour les corpus petits** (<20k points), il se peut que Qdrant ne construise jamais HNSW et reste sur du scan exhaustif. C'est ok à cette échelle.
- **Tu peux forcer l'indexation** via `optimizers_config` au moment de créer la collection.

À l'échelle d'Eigenmind en MVP, on ne se préoccupe pas de ça. Les défauts sont bons.

## Payload indexing — la magie des filtres rapides

Par défaut, les filtres sur payload sont scannés linéairement. Pour 100k points c'est ok, pour 10M ça devient lent. Qdrant supporte les **payload indexes** : tu déclares qu'un champ doit être indexé.

```python
client.create_payload_index(
    collection_name="documents",
    field_name="user_id",
    field_schema="keyword",
)
```

Schemas supportés :
- `keyword` — string égalité (le plus courant)
- `integer` — int + range
- `float` — float + range
- `geo` — coordonnées
- `text` — full-text search (sur des mots)
- `bool`
- `datetime`

Le payload indexing est **différent** de l'index HNSW. HNSW indexe les vecteurs pour la recherche par similarité. Payload index indexe les métadonnées pour les filtres rapides. Les deux coexistent.

Pour Eigenmind on indexera `user_id` et `filename` (besoin pour le Smart Resume) en phase 4.

## Filtres au moment de la recherche

L'intérêt principal du payload. Exemples de filtres :

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Filtre simple : un user
query_filter = Filter(
    must=[FieldCondition(key="user_id", match=MatchValue(value="alice"))]
)

# Filtre composé
query_filter = Filter(
    must=[
        FieldCondition(key="user_id", match=MatchValue(value="alice")),
        FieldCondition(key="filename", match=MatchValue(value="rapport.pdf")),
    ],
    should=[
        FieldCondition(key="language", match=MatchValue(value="fr")),
        FieldCondition(key="language", match=MatchValue(value="en")),
    ],
)

# Range
Filter(must=[FieldCondition(key="year", range=Range(gte=2024))])
```

`must` = AND. `should` = OR. `must_not` = NOT. Tu peux nester.

Important : Qdrant applique les filtres **pendant** la recherche HNSW (filterable HNSW), pas après. Tu ne perds pas les performances d'index. C'est un avantage concurrentiel par rapport à d'autres vector DBs où le filtre se fait en post-processing.

## Quantization au moment du stockage

Évoqué dans 01-1, ici en pratique. Activer scalar quantization au moment de créer la collection :

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            always_ram=True,  # quantized vectors in RAM, originals on disk
        )
    ),
)
```

Avec `always_ram=True`, les vecteurs quantisés (4x plus petits) sont en RAM, les originaux peuvent partir sur disque. Qdrant utilise les quantisés pour le candidat-screening et raffine avec les originaux pour le top-K final. **Gain x4 RAM, perte de qualité <2%**.

Pour Eigenmind en MVP, pas nécessaire. À activer si tu scales.

## Persistance et write-ahead log

Qdrant utilise un **write-ahead log** (WAL) : chaque upsert est d'abord écrit dans un journal sur disque, puis appliqué à l'index en RAM. En cas de crash, le WAL est rejoué au démarrage. **Cohérence forte** garantie sur les opérations.

Le storage dans le volume Docker monté contient :

```
qdrant_storage/
├── collections/
│   └── documents/
│       ├── 0/                  # shard 0
│       │   ├── segments/       # segments indexés
│       │   └── wal/            # write-ahead log
│       └── config.json
└── raft_state.json             # cluster state (Raft)
```

Tu peux backuper le dossier `qdrant_storage/` à chaud — Qdrant garantit qu'un snapshot du disque est restaurable.

## Snapshots et backups

Pour les backups propres, Qdrant supporte les **snapshots** explicites :

```python
client.create_snapshot(collection_name="documents")
# génère un .snapshot dans qdrant_storage/collections/documents/snapshots/
```

Un snapshot est un dump **consistent** de la collection à un instant T. Il inclut les vecteurs, payloads, et l'index HNSW (pas besoin de tout réindexer à la restauration).

Pour restaurer ailleurs :

```python
client.recover_snapshot(
    collection_name="documents",
    location="http://...path/to/snapshot.snapshot",
)
```

En prod, c'est ce que tu schedules quotidiennement et envoies sur S3.

## Le dashboard Qdrant — usage avancé

`http://localhost:6333/dashboard` propose plus que de la visualisation :

- **Collections** — liste, count, taille disque, taille RAM.
- **Points** — browser, click pour voir payload + vecteur (les 384 floats si tu veux).
- **Console** — un éditeur de requêtes pour faire des recherches au format JSON.
- **Visualize** — projection t-SNE / UMAP des vecteurs en 2D. Tu peux **voir** ton corpus comme un nuage de points coloré par payload. Très utile pour comprendre la structure de tes données.
- **Métriques** — latence, throughput, RAM, segments.

La visualisation 2D vaut le coup d'œil quand on en arrivera à la phase 2 (graphes) — on peut souvent y identifier des clusters thématiques à l'œil.

## Monitoring en prod (à savoir)

Métriques Qdrant accessibles via Prometheus à `/metrics` :

- `app_info` — version
- `collections_total` — nombre de collections
- `collections_vector_total` — nombre total de vecteurs
- `app_status_recovery_mode` — récupération en cours ?
- HTTP request latencies, throughput
- gRPC latencies

À monitorer en prod : latence p99 des recherches, RAM utilisée, taille des segments non-indexés.

## Multi-tenancy : implémentation concrète

Trois patterns (rappel de 01-1), avec leur trade-off implémentation :

**Collection-per-tenant**

```python
collection_name = f"{user_id}_documents"
client.create_collection(collection_name, ...)
```

Simple. Eigenmind fait ça. Limite à 1000-10000 tenants raisonnable.

**Single collection + payload filter**

```python
client.create_collection("documents", ...)
client.create_payload_index("documents", "user_id", "keyword")

# à la recherche
client.query_points(
    "documents",
    query=vec,
    query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=current_user))]),
)
```

Scale à des millions. **Vulnérabilité applicative** : un bug qui oublie le filter expose les données. Sécurité côté code, pas côté DB.

**Sharded collection (Qdrant 1.7+)**

```python
client.create_collection(
    "documents",
    vectors_config=...,
    shard_number=8,
    sharding_method="custom",
)

# à l'upsert et au query, passer shard_key=user_id
client.upsert(..., shard_key_selector=user_id)
```

Isolation physique au niveau shards. Compromis prod-grade.

## Maintenance : delete, update, count

```python
# Compter
info = client.get_collection("documents")
print(info.points_count)  # nombre approximatif (rapide)
print(info.exact_count)   # exact, plus lent

# Supprimer un point
client.delete(collection_name="documents", points_selector=[42])

# Supprimer par filtre (Smart Resume reverse — par exemple "supprime tous les chunks d'un filename")
client.delete(
    collection_name="documents",
    points_selector=Filter(must=[FieldCondition(key="filename", match=MatchValue(value="old.pdf"))]),
)

# Modifier un payload sans toucher au vecteur
client.set_payload(
    collection_name="documents",
    payload={"reviewed": True},
    points=[42, 43, 44],
)

# Supprimer une collection entière
client.delete_collection("documents")
```

## Le piège à éviter : oublier `with_payload=True`

Par défaut au query, `with_payload=False` — Qdrant ne renvoie que l'id et le score. Si tu oublies de mettre `True`, tu n'as pas le texte pour appeler le LLM. Erreur de débutant qu'on retrouve dans tous les tutos.

Idem pour `with_vectors=True` si tu veux récupérer le vecteur original (rare, mais utile pour construire le graphe en phase 2).

## Atomicité

Les upserts Qdrant sont **atomiques au niveau de chaque point**, mais pas au niveau d'un batch. Si tu envoies 100 points et que le 50e échoue, les 49 premiers sont bien insérés, le 50e ne l'est pas, et les suivants peuvent ou non l'être selon l'erreur. Ne suppose pas l'atomicité du batch entier.

Pour des cas critiques (rare en RAG), tu peux faire un upsert par point — coûteux mais sûr.
