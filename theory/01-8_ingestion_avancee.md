# 1.8 — Ingestion avancée : ce que fait le mémo officiel en plus

> **Source.** Ce document s'appuie sur le mémo technique officiel de Merlin Intelligence,
> `theory/260522_Eigenmind_Cognitive_Maps.pdf` (sections 2 et 3). Il complète les chapitres
> `01-3` (chunking), `01-2` (embeddings) et `01-4` (stockage Qdrant) en exposant les briques
> que **notre MVP a simplifiées ou laissées de côté**. Rien de tout ça n'est requis pour que
> le MVP fonctionne : c'est ce qu'on retrouvera en clonant le repo complet.

## Vue d'ensemble du pipeline officiel

Le repo officiel découpe l'ingestion en quatre étages séquentiels, orchestrés par une classe
`Ingester` (`pipelines/ingest.py`) :

```
document_loaders.py → chunking.py → embeddings.py → vectordb/store.py
```

`Ingester` expose **deux stratégies** :

- `run_chunknorris` : ingestion *structure-aware* spécifique PDF.
- `run_multi_format` : pipeline générique PDF / DOCX / PPTX / XLSX / TXT / MD.

Notre MVP ne couvre que le chemin « PDF → splitter récursif », d'où les manques ci-dessous.

## 1. Extraction = un canal bruité (`document_loaders.py`)

### Le cadrage mathématique

Chaque fichier `f` porte un texte « auteur » sous-jacent `T*(f)`. L'extracteur `E` en réalise
un **décodage avec perte** :

```
T̂ = E(f) ≈ T*(f)
```

- Pour un document **nativement numérique**, `E` est quasi exact.
- Pour un **scan**, `E` se réduit à de l'OCR, qui est la reconstruction **maximum a posteriori**
  `T̂ = argmax_T p(T | I)` à partir d'une séquence d'images de pages `I = (I_1, …, I_np)`.

C'est une grille de lecture utile : l'extraction n'est pas neutre, elle **injecte du bruit** qui
se propage ensuite dans l'espace d'embedding.

### Le fallback OCR (✅ désormais implémenté dans le MVP)

Heuristique de détection des PDF scannés : soit `T_0` le texte de la couche texte du PDF et
`n_p` le nombre de pages. Si

```
|T_0| / n_p < 20   (moins de 20 caractères par page)
```

le fichier est considéré comme **image-only** et on bascule sur l'OCR :

```
T(f) = T_OCR(f)   si |T_0|/n_p < 20  et  |T_OCR| > |T_0|
       T_0        sinon
```

Chaîne technique : `pdf2image → pytesseract`. La branche OCR n'est invoquée **que si** le chemin
pas cher échoue — ça préserve le débit d'ingestion sur un corpus bien formé.

> **Implémentation MVP (étape 5.1).** C'est désormais codé dans `src/load_pdf.py` (`_ocr_page` +
> paramètre `ocr` de `load_pdf_pages`). Deux différences assumées avec le mémo :
> 1. Le test caractères/page est appliqué **par page** (et non au niveau document), pour gérer les
>    PDF **mixtes texte + images** : on n'OCR que les pages dont la couche texte est trop pauvre.
> 2. Un mode `--ocr always` force l'OCR de toutes les pages (utile quand chaque page porte des
>    images contenant du texte). `auto` (défaut) suit le seuil ; `never` désactive.
>
> Dépendances : `tesseract` + `poppler` (système, `brew` — voir `docs/INSTALL.md`) et
> `pytesseract` + `pdf2image` (pip). Dégradation gracieuse si absent (OCR ignoré, le reste tourne).
> Le reste de la Phase 5 (connecteurs, multi-user, etc.) **n'est volontairement pas fait dans le
> MVP** : il vit dans le repo de production `merlin-intelligence/eigenmind`.

### Le format dispatch et ses limites

`E` dispatche selon l'extension : PDF (`pypdf`), DOCX (`python-docx`), PPTX (`python-pptx`),
XLSX (`pandas`), texte brut. Limites héritées des libs sous-jacentes, à connaître :

- **PDF** : le re-flow casse l'ordre typographique sur les mises en page multi-colonnes.
- **PPTX** : l'itération sur les *shapes* rate le texte alternatif des images.
- **XLSX** : la sémantique tableur est aplatie en une sérialisation string.

Les extractions vides sont **filtrées** avant le chunking ; un échec sur un fichier est loggé et
**n'interrompt pas** le run (idempotence du pipeline).

## 2. Chunking : deux stratégies (`chunking.py`)

### Le splitter récursif (ce qu'on a fait — Langchain)

Liste de séparateurs par priorité `S = ("\n\n", "\n", ". ", …, "")` : on partitionne
récursivement au séparateur **le mieux classé** qui produit des fragments de taille `≤ L`
(défaut `L = 300` caractères), avec un overlap `o = 30` réinjecté entre chunks consécutifs.

Cadrage formel : un chunker réalise un **recouvrement** (overlapping cover) `κ : Σ* → (Σ*)^m`
tel que `T ⊆ ∪_i c_i`. L'overlap `o > 0` garantit que `c_i ∩ c_{i+1}` est non vide, donc
**toute phrase à cheval sur une coupe** apparaît en entier dans au moins un chunk. Si `L` et `o`
dépassent la longueur de phrase maximale, `κ` est un recouvrement *sentence-preserving*.

L'ordre des séparateurs encode une hiérarchie sémantique grossière→fine
(paragraphe > ligne > phrase > mot), et la descente gloutonne réalise une **coupe à disruption
minimale**.

### ChunkNorris : le splitter *structure-aware* (ce qu'on n'a pas)

Pour les PDF, ChunkNorris parse d'abord le fichier en un **intermédiaire markdown**
`M = π_PDF(f)`, puis découpe `M` le long de sa **hiérarchie de titres** (sections, listes,
tableaux). `M` est un arbre de paires `(heading, body)` ; on chunk le long des sous-arbres
plutôt que par offset de caractères.

- **Avantage** : cohérence topique supérieure (un chunk ≈ une section).
- **Coût** : latence plus élevée, stack de dépendances plus lourde, longueurs inégales.

**Limite commune aux deux** : le chunking est *length-driven*, pas *topic-driven*. Un argument
long peut être coupé en plein milieu d'une affirmation ; une section « titre seul » peut produire
un chunk dégénéré. Le chunking adaptatif *embedding-aware* est laissé en travaux futurs.

## 3. Embedding : conventions à connaître (`embeddings.py`)

### Device dispatch

La classe `EmbeddingModel` appelle un `detect_device()` qui renvoie le premier accélérateur
disponible dans l'ordre `cuda > mps > cpu`, et libère le modèle + vide le cache CUDA en fin de run
(protocole *context manager*).

> ⚠️ **Écart MVP.** Notre `embed_text.py` force `device="cpu"` en dur. Sur ton M3, basculer sur
> `mps` accélérerait l'embedding (cf. le troubleshooting du README). Le repo officiel le fait
> automatiquement.

### La convention de normalisation (le point clé)

Les sorties *mean-pooled* de SentenceTransformer pour `all-MiniLM-L6-v2` sont **approximativement
ℓ₂-unitaires** : `‖φ_i‖₂ ≈ 1`. Conséquence directe et importante :

> la matrice de Gram `W = ΦΦᵀ` **approxime directement les similarités cosinus**, sans étape de
> normalisation explicite. `W_ij = φ_iᵀ φ_j ≈ cos∠(φ_i, φ_j)`.

C'est ce qui rend l'analyse spectrale de la phase 2 cohérente « gratuitement ».

## 4. Indexation : HNSW comme index de sphère (`vectordb/store.py`)

### L'équivalence cosinus / euclidien / produit interne

Pour des vecteurs ℓ₂-unitaires :

```
‖φ_i − φ_j‖²₂ = 2 (1 − φ_iᵀ φ_j)
```

Donc **plus proche voisin euclidien = produit interne maximal = similarité cosinus maximale** :
les trois classements coïncident. Choisir `Distance.COSINE` dans la config de la collection est
donc cohérent avec la normalisation implicite de SBERT. HNSW (Hierarchical Navigable Small World)
retourne les voisins approchés en `O(log N)` en moyenne.

### Le payload comme index secondaire

Stocker `(filename, chunk_number, text, ingestion_date)` dans le payload signifie que l'index
vectoriel **et** l'index documentaire vivent dans le même store : un seul appel `retrieve` renvoie
à la fois la géométrie (le vecteur) et le contenu lisible. `ingestion_date` permet une
ré-indexation/suppression fenêtrée par jour (`delete_for_date`).

### Déduplication

`existing_filenames(collection)` scrolle la collection et renvoie l'ensemble des noms déjà
indexés ; l'ingester **skippe** tout fichier dont le basename est déjà présent.

> ⚠️ **Limite.** La dédup est au **niveau filename uniquement** : deux fichiers distincts
> contenant des chunks identiques seront tous deux indexés. La déduplication sémantique est
> laissée aux étages downstream.

### L'interprétation : « admission géométrique »

Une fois un fichier ingéré, chacun de ses chunks devient une coordonnée dans l'espace partagé,
atteignable en `O(log N)` depuis n'importe quelle requête. Un nouveau document ne se range pas
*à côté* du corpus : il entre immédiatement dans sa structure de voisinage. L'ingestion n'est pas
du *stockage de fichiers*, c'est une **admission dans une variété sémantique en évolution**.

Question d'analyste : *« Où, dans le corpus existant, ce nouveau document se situe-t-il ? »*

## Récap des écarts MVP ↔ officiel (ingestion)

| Brique | Notre MVP | Mémo officiel |
|---|---|---|
| Formats | PDF seul | PDF/DOCX/PPTX/XLSX/TXT/MD |
| Scans / images | ✅ OCR par page (auto/always/never) | OCR fallback (seuil 20 char/page) |
| Chunking | splitter récursif | + ChunkNorris (markdown-aware) |
| Device | `cpu` en dur | `cuda > mps > cpu` auto |
| Dédup | écrasement par ID | `existing_filenames` au niveau fichier |

Aucun de ces écarts n'est un bug : ce sont des choix de simplification pédagogique. Le reste
(formats supplémentaires, packaging…) vit dans le repo de production `merlin-intelligence/eigenmind`.
