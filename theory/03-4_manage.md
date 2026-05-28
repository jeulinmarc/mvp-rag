# 3.4 — Gestion des documents : filtres payload, suppression sélective

## Le besoin

Quand un utilisateur ingère plusieurs PDFs, il a besoin de :
- voir quels fichiers sont indexés et combien de chunks chacun a produit
- supprimer un fichier spécifique sans toucher au reste
- éventuellement tout purger pour repartir de zéro
- visualiser quand chaque fichier a été ajouté

C'est le rôle de la page Manage. Conceptuellement simple, mais touche aux mécaniques Qdrant qu'on n'a pas encore explorées : les **filtres payload** et la suppression sélective.

## Lister les fichiers indexés

Qdrant ne nous donne pas directement une "vue distincte" des filenames. Il faut **scroller** la collection et agréger côté client :

```python
def list_indexed_files(client, collection):
    files = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # important: skip vectors for speed
        )
        for p in points:
            fn = p.payload.get("filename")
            if fn not in files:
                files[fn] = {"chunks": 0, "pages": set()}
            files[fn]["chunks"] += 1
            files[fn]["pages"].add(p.payload.get("page"))
        if offset is None:
            break
    return files
```

Trois choses à noter :

**1. `with_vectors=False`.** Crucial. Récupérer 384 floats par point pour juste lister les filenames est un gaspillage massif de bande passante et de RAM. Pour 10 000 chunks, c'est ~15 Mo vs ~3 Mo. Toujours désactiver les vecteurs quand on n'en a pas besoin.

**2. Pagination via `offset`.** Le `scroll` Qdrant utilise un curseur d'offset qu'on passe à la requête suivante. Quand l'API renvoie `offset=None`, on a tout récupéré.

**3. Agrégation côté client.** Qdrant n'a pas de `GROUP BY` natif. On agrège en Python — acceptable jusqu'à quelques millions de points.

## Suppression par filtre

Le pattern critique : supprimer **tous les chunks d'un filename donné**. Si on devait stocker les IDs côté client pour rappeler `delete([id1, id2, ...])`, ce serait fragile. Qdrant supporte la **suppression par filtre**, beaucoup plus propre :

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

client.delete(
    collection_name="documents",
    points_selector=Filter(
        must=[FieldCondition(key="filename", match=MatchValue(value="rapport.pdf"))]
    ),
)
```

Qdrant scanne sa collection, identifie les points matchant, et les supprime. **Atomique** au niveau de l'opération.

Si tu as indexé `filename` en payload index (`client.create_payload_index`), c'est très rapide même sur de gros corpus. Sinon, scan linéaire.

## Soft delete vs hard delete

Deux philosophies dans les systèmes de stockage :

**Hard delete** (notre choix Qdrant) : le point est immédiatement effacé du disque (ou marqué pour effacement, et le compactage le supprime ensuite). Pas de récupération. Simple.

**Soft delete** : le point est marqué `deleted=True` dans son payload, mais reste physiquement présent. Les recherches ignorent les points marqués. Avantage : possible de "restaurer" un fichier supprimé par erreur, audit trail. Inconvénient : la base grossit indéfiniment, il faut un job de purge périodique.

Pour Eigenmind on fait hard delete — simplicité avant tout. Si un utilisateur veut restaurer un PDF, il le réingère.

## Confirmation utilisateur

Une suppression est **irréversible**. Toute UI de suppression doit demander une confirmation explicite. Pattern Streamlit classique :

```python
if st.button("Supprimer", type="primary"):
    st.session_state.pending_delete = filename

if st.session_state.get("pending_delete"):
    st.warning(f"Confirmer la suppression de '{st.session_state.pending_delete}' ?")
    c1, c2 = st.columns(2)
    if c1.button("✅ Oui, supprimer"):
        delete_file(st.session_state.pending_delete)
        st.session_state.pending_delete = None
        st.rerun()
    if c2.button("❌ Annuler"):
        st.session_state.pending_delete = None
        st.rerun()
```

Deux états : "demande initiale" et "demande de confirmation". L'utilisateur doit cliquer deux fois pour valider.

## Invalidation du cache

Comme pour l'ingest, après toute suppression on doit invalider le cache graphe. Sinon le `GraphAwareCache` continue de référencer des Singular/Hinge nodes qui n'existent plus.

```python
def delete_file(filename):
    client.delete(...)
    st.session_state.graph_cache = None
```

C'est la même règle qu'en page Ingest. Tout changement de la collection invalide le graphe.

## Effacement total : `delete_collection` vs `delete + filter`

Pour purger toute la collection, deux options :

**Option A — Delete collection** :
```python
client.delete_collection("documents")
```
Plus rapide (juste un drop). Mais la collection n'existe plus, il faudra la recréer à la prochaine ingestion (ce que fait `ensure_collection` dans `store_chunks.py`).

**Option B — Delete all points avec un filter vide** :
```python
client.delete(
    collection_name="documents",
    points_selector=Filter(must=[]),  # match all
)
```
Garde la collection mais la vide. Plus lent (Qdrant doit scanner tous les points).

Eigenmind utilise Option A par défaut — collection vide = collection inexistante, c'est cohérent.

## Affichage de la date d'ingestion

On n'a pas ajouté de timestamp dans le payload jusqu'ici. Pour le faire :

```python
import datetime
chunk_payload["indexed_at"] = datetime.datetime.utcnow().isoformat()
```

Phase 4-5 idée. En MVP, on liste juste les fichiers sans date — l'info est limitée.

## Sécurité dans un contexte multi-user (phase 5)

Si plusieurs utilisateurs partagent une instance Qdrant, la page Manage doit **scoper toutes les opérations au user courant** :

```python
client.delete(
    collection_name=f"{user_id}_documents",
    points_selector=Filter(must=[
        FieldCondition(key="filename", match=MatchValue(value=filename))
    ]),
)
```

Avec le namespacing par collection, on garantit qu'un user ne peut pas supprimer les docs d'un autre. Avec le pattern single-collection + payload filter, le filtre `must` doit toujours inclure `user_id` — un bug d'oubli expose toute la base.

En MVP single-user on n'a pas ce souci.

## Le piège de la pagination interrompue

Si tu liste les fichiers et qu'une ingestion concurrente ajoute des points pendant le scroll, certains nouveaux points peuvent apparaître dans ta vue ou non, selon le timing. Pour de la prod, il faut un snapshot consistent. Pour le MVP single-user, on accepte ce petit non-déterminisme.
