# 1.1 — Vector Databases & Qdrant

## L'embedding sémantique : le concept de base

Un RAG repose sur une idée simple : on transforme un texte (un chunk de document) en un vecteur de nombres flottants — typiquement 384 ou 768 dimensions — qui capture son **sens**. Deux textes qui parlent de la même chose se traduisent par deux vecteurs proches dans cet espace, au sens de la **similarité cosinus** (cosinus de l'angle entre les deux vecteurs, valeur entre -1 et 1).

C'est ce qu'on appelle un *embedding sémantique*. Le modèle qu'on utilise, `sentence-transformers/all-MiniLM-L6-v2`, mappe n'importe quelle string vers un vecteur 384-dimensionnel.

## Pourquoi une base de données dédiée

Quand on a 10 000 chunks, comparer un vecteur de question à tous les chunks par produit scalaire coûte O(N) — linéaire dans la taille du corpus. À l'échelle d'un million de chunks ça devient prohibitif : 1M × 384 multiplications par requête, soit ~50ms minimum sur CPU moderne, et il faut compter le réseau et le parsing en plus.

Une **vector database** résout ça avec un index spatial. L'idée fondamentale : précalculer une structure de données qui permet de **sauter** la plupart des comparaisons inutiles à l'inférence. On accepte un coût de construction d'index (one-shot) en échange d'une recherche logarithmique.

L'algorithme dominant en 2026 est **HNSW** (Hierarchical Navigable Small World). C'est celui qu'utilise Qdrant par défaut.

## HNSW — comment ça marche vraiment

HNSW construit un **graphe multi-couches** où chaque nœud est un vecteur de ta base.

- **Couche 0** : tous les vecteurs sont présents, connectés à leurs voisins les plus proches (par défaut M=16 voisins par nœud).
- **Couche 1** : ~1/e des nœuds (sélection aléatoire), connectés entre eux.
- **Couche 2** : encore ~1/e des nœuds de la couche 1.
- **Etc.** : moins de nœuds à chaque couche, jusqu'à 1 seul tout en haut.

Visuellement, c'est une pyramide où le haut est très clairsemé (autoroutes longue distance) et le bas est dense (routes locales).

**À la recherche**, on fait une descente "greedy" :

1. On entre par le sommet de la pyramide.
2. À chaque couche, on saute vers le voisin le plus proche de la cible. Si aucun voisin n'est plus proche que le nœud courant, on descend d'une couche.
3. Arrivé en couche 0, on raffine en explorant les voisins jusqu'à converger.

Le résultat : recherche en O(log N) en pratique, avec un rappel typique de 95-99 % par rapport à l'exhaustif. Pour 1M de vecteurs, ça prend ~1-5ms au lieu de ~50ms en exhaustif.

### Les paramètres HNSW qui comptent

- **M** (par défaut 16) — nombre de voisins par nœud en couche 0. Plus c'est élevé, meilleur le rappel mais plus l'index est gros et long à construire. Sweet spot 16-32.
- **ef_construct** (défaut 100) — taille de la queue de candidats pendant la construction. Plus c'est élevé, meilleur l'index mais plus la construction est lente. 100-500.
- **ef** (défaut 64, paramètre de requête) — taille de la queue à la recherche. Le **knob qualité-vitesse runtime**. Tu peux le monter à 256 pour du rappel max, le baisser à 32 pour de la latence min.

Compromis pratiques : recall@10 vs latence sur 1M de vecteurs 384-d :

| M | ef | Recall@10 | Latence par requête |
|---|---|---|---|
| 16 | 32 | 91% | 0.8 ms |
| 16 | 64 | 95% | 1.5 ms |
| 16 | 128 | 98% | 3 ms |
| 32 | 128 | 99% | 4 ms |

Pour Eigenmind on garde les défauts, qui sont déjà bons.

### Alternatives à HNSW

- **IVF** (Inverted File) — découpe l'espace en clusters via k-means, on cherche dans les clusters les plus proches. Utilisé par FAISS. Bon pour les très gros volumes (>10M) mais moins fin que HNSW.
- **LSH** (Locality Sensitive Hashing) — hash les vecteurs pour que les proches collisionnent. Historique, peu utilisé en prod aujourd'hui.
- **ANNOY** (Spotify) — basé sur des forêts d'arbres aléatoires. Lisible, mais surpassé par HNSW.
- **DiskANN** (Microsoft) — variante HNSW optimisée disque, pour les très très gros corpus qui ne tiennent pas en RAM.

Pour <100M de vecteurs et de la RAM dispo, HNSW gagne sur tous les fronts.

## Le paysage des vector DBs

| DB | Lang | Open Source | Best for |
|---|---|---|---|
| **Qdrant** | Rust | Apache 2.0 | Filtres avancés, perfs, simplicité |
| **Weaviate** | Go | BSD | Hybrid search natif, modules ML |
| **Milvus** | Go/C++ | Apache 2.0 | Très gros corpus, GPU search |
| **Pinecone** | (fermé) | SaaS uniquement | Setup minimal, pricing élevé |
| **Chroma** | Python | Apache 2.0 | Prototypes, embedded |
| **pgvector** | C | Postgres ext | Si déjà sur Postgres |
| **Elasticsearch** | Java | Apache 2.0 | Hybrid avec full-text mature |
| **FAISS** | C++ | MIT | Lib in-process, pas un serveur |

Qdrant gagne sur trois points pour notre cas : filtres riches sur payload (essentiel pour le multi-tenancy d'Eigenmind), perf par défaut sans tuning, Docker en 1 commande.

## Vocabulaire Qdrant

Une **collection** est l'équivalent d'une table SQL. Elle a une dimension fixe (ici 384) et une distance fixe (ici Cosine).

Un **point** est une ligne dans cette collection. Il a un **id**, un **vector**, et un **payload** — dictionnaire de métadonnées libres, par exemple `{"filename": "rapport.pdf", "page": 12, "text": "..."}`.

On peut **filtrer** les recherches par payload — par exemple "donne-moi les 5 chunks les plus proches sémantiquement de cette question, mais uniquement parmi les fichiers de l'utilisateur Alice". C'est exactement comme ça qu'Eigenmind fait :
- son **namespacing multi-user** : collections nommées `<user>_<collection>`
- son **Smart Resume** : skip des filenames déjà présents avant ré-ingestion

## Indexation des payloads

Par défaut, les filtres sur payload sont scannés linéairement. Pour 100k points c'est ok, pour 10M ça devient lent. Qdrant supporte les **payload indexes** : tu déclares qu'un champ (`filename`, `user_id`...) doit être indexé, et il construit une structure de données dédiée (B-tree pour ranges, hashmap pour égalité).

```python
client.create_payload_index(
    collection_name="documents",
    field_name="user_id",
    field_schema="keyword",
)
```

Schemas supportés : `keyword` (string égalité), `integer`, `float`, `geo`, `text` (full-text), `bool`, `datetime`.

Pour Eigenmind, on indexera `user_id` et `filename` en phase 4-5.

## Quantization — économiser la mémoire

Les vecteurs float32 prennent 4 octets par dimension. Pour 1M de vecteurs 384-d : 1.5 Go en RAM. Pour 100M : 150 Go — inacceptable.

Trois techniques de **quantization** réduisent ça :

**Scalar quantization (8-bit)** — chaque dimension passe de float32 (32 bits) à uint8 (8 bits). **Gain x4 mémoire**, perte de rappel typiquement <2%. Le défaut pratique pour les gros corpus.

**Product quantization (PQ)** — découpe le vecteur en sous-vecteurs, et remplace chaque sous-vecteur par l'index du centroïde le plus proche dans un codebook. **Gain x8 à x32**, perte de rappel 5-10%. Pour les corpus massifs.

**Binary quantization** — chaque dimension devient 1 bit (signe positif/négatif). **Gain x32**, mais rappel divisé par 2-3. Utilisable seulement avec un **re-ranking** où on récupère un large top-K en binaire puis on re-classe avec les float32. Très utilisé dans le state-of-the-art 2025-2026.

Pour Eigenmind à l'échelle d'un utilisateur (quelques milliers à dizaines de milliers de chunks), pas besoin de quantization. À retenir si tu scales un jour.

## Mode d'exécution

Qdrant tourne en serveur. On lui parle en :
- **REST** sur le port 6333 (debug facile au curl, lisible humainement)
- **gRPC** sur le port 6334 (plus rapide en prod, payload binaire)

Le client Python `qdrant-client` encapsule les deux. Par défaut il utilise REST, on peut activer gRPC avec `prefer_grpc=True`. Gain typique : 30-50% sur le throughput d'upsert pour les gros batches.

Trois modes de déploiement :

- **Docker local** (notre choix) — isolé, reproductible, persistance via volume.
- **Cloud managé** — Qdrant Cloud, en quelques clics, payant.
- **Embedded** — `from qdrant_client import QdrantClient; client = QdrantClient(":memory:")`. Pas de serveur, tout dans le process Python. Utile pour tests unitaires.

## Persistance et backups

Sans volume monté, toutes les données disparaissent à `docker compose down`. Notre `docker-compose.yml` monte `./qdrant_storage` sur `/qdrant/storage` dans le conteneur. Tout est sur ton disque local.

Qdrant utilise un **write-ahead log** (WAL) : chaque upsert est d'abord écrit dans un journal, puis appliqué à l'index. En cas de crash, le WAL est rejoué au démarrage. Cohérence forte garantie sur les opérations.

Pour les backups, Qdrant supporte les **snapshots** :

```python
client.create_snapshot(collection_name="documents")
# génère un .snapshot file dans qdrant_storage/collections/documents/snapshots/
```

Un snapshot est un dump consistent de la collection. Tu peux le restaurer ailleurs, le copier sur S3, etc. En prod, c'est ce que tu schedules quotidiennement.

## Multi-tenancy : 3 stratégies

Quand plusieurs utilisateurs partagent une instance Qdrant, trois patterns :

**Collection-per-tenant** — `alice_docs`, `bob_docs`, etc. Isolation forte, simple. Inconvénient : explosion du nombre de collections si tu as 10 000 users. Chaque collection a un coût mémoire fixe (~10 Mo de structure HNSW vide). C'est le choix d'Eigenmind, valable jusqu'à quelques milliers d'utilisateurs.

**Single collection + payload filter** — une seule collection `documents`, avec `user_id` dans chaque payload. Tu filtres à la recherche : `must=[{"key": "user_id", "match": {"value": "alice"}}]`. Scale à des millions d'users. Risque : un bug d'application qui oublie le filtre exposerait les données d'un autre user. Sécurité applicative.

**Sharded collection** — depuis Qdrant 1.7+, support natif du sharding par tenant. `client.create_collection(..., shard_number=10, sharding_method="custom")`. Isole les données physiquement tout en restant logique. Le meilleur des deux mondes, mais plus complexe.

Eigenmind est sur le premier pattern. C'est ce qu'on implémentera en phase 4.

## Le dashboard Qdrant

`http://localhost:6333/dashboard` te donne une UI web. À ce stade tu pourras y voir :
- ta collection
- son nombre de points
- un browser pour cliquer sur un point et voir son payload + son vecteur
- un éditeur de requêtes Console
- des métriques (mémoire, latence, throughput)

Très utile pour debug visuellement. Tu peux aussi y faire des recherches manuelles plus tard.

## Quand ne PAS utiliser une vector DB

À ne pas oublier : la vector DB n'est pas toujours nécessaire.

- **< 10k chunks** : un simple `numpy.argsort(cosine_sim)` sur un array en mémoire suffit. Latence négligeable.
- **Recherche exacte** : si tu veux retrouver un mot exact, BM25 / Elasticsearch / SQLite FTS surpasse les embeddings.
- **Métadonnées structurées** : si la requête est "tous les docs de 2024 du dossier X", c'est du SQL, pas du vector search.
- **Données très structurées** : tableaux financiers, séries temporelles — pas pour les embeddings.

Le bon RAG mélange souvent vector + BM25 (hybrid search) — on en reparle en 1.5.
