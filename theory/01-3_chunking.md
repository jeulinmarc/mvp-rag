# 1.3 — Chargement de documents et chunking

## Pourquoi chunker ?

On ne peut pas embedder un document entier d'un coup pour deux raisons :

1. **Contrainte technique** : les modèles d'embedding ont une fenêtre de contexte limitée. `all-MiniLM-L6-v2` accepte 256 tokens — au-delà, il tronque silencieusement. Un PDF de 50 pages devient donc inutilisable comme bloc unique.

2. **Contrainte sémantique** : un vecteur 384-d représente le sens *global* du texte fourni. Plus le texte est long, plus ce vecteur devient un mélange flou de tous les sujets abordés. Pour un retrieval précis, on veut des unités sémantiquement focalisées — un paragraphe traitant *une* idée.

Le chunking est donc un découpage du document en blocs assez petits pour rester sémantiquement focalisés, assez grands pour porter une idée complète.

C'est probablement **l'étape qui a le plus d'impact sur la qualité finale d'un RAG**, et paradoxalement la moins glamour. Un bon embedder avec un mauvais chunking donne un RAG médiocre. L'inverse est plus pardonnable.

## Le compromis taille de chunk

### Trop petit (50-100 tokens)

```
Chunk 12 : "Il a dit que c'était impossible."
```

Inutilisable seul. Qui est "il" ? Qu'est-ce qui était impossible ? Le contexte est perdu. Le vecteur embedding est trop générique — n'importe quelle phrase de citation ressemblera à ça.

### Trop grand (1500+ tokens, plusieurs paragraphes hétérogènes)

```
Chunk 4 : "[paragraphe sur les ventes 2024]
          [paragraphe sur la stratégie RH]
          [paragraphe sur la conformité ESG]"
```

Le vecteur embedding représente une moyenne floue de tous ces sujets. Une question sur les ventes va mal matcher parce que le signal est dilué par le RH et l'ESG. C'est le problème dit de **dilution sémantique**.

### Sweet spot

300-800 tokens en pratique. Eigenmind utilise typiquement ~500 caractères (≈100-150 tokens en français).

Pour un texte en français, 1 token ≈ 3-4 caractères. Pour de l'anglais, 1 token ≈ 4-5 caractères. Compter en caractères est plus simple et stable que de tokenizer pour décider du découpage.

### Règle de pouce empirique

- Documents narratifs (rapports, articles) : **500-800 caractères**
- Documentation technique (API docs, code) : **300-500 caractères** (les concepts sont denses)
- Listes et FAQs : **un item = un chunk** (taille variable)
- Code source : **une fonction = un chunk** quand possible

## L'overlap (chevauchement)

Si on découpe à 500 caractères pile, on peut couper une phrase en plein milieu. Le bout de phrase qui termine le chunk N et le début de phrase qui commence le chunk N+1 deviennent tous les deux incompréhensibles.

Solution : on fait se chevaucher les chunks de 50-100 caractères. Le chunk N+1 commence un peu avant la fin du chunk N. Une information à cheval sur deux chunks est ainsi présente *entière* dans au moins un des deux. C'est une assurance contre les coupures malheureuses.

### Le paradoxe de l'overlap

On pourrait penser "plus d'overlap = mieux". Faux. Trop d'overlap (>30%) crée des **doublons quasi-identiques** dans la vector DB. Quand tu fais un top-5 retrieval, tu récupères 5 chunks qui sont en réalité 2 ou 3 contenus différents avec leurs variantes overlappées. Le contexte envoyé au LLM est redondant et tu rates de l'info utile.

Sweet spot : **10-20% d'overlap**. Pour chunk_size=500, overlap=50 à 100.

## Stratégies de chunking, du plus simple au plus malin

### 1. Fixed-size character splitting

On découpe tous les N caractères, point.

```
"Le chat dort. Il rê" | "ve de souris. La sourcier"
```

Rapide, brutal, ignore la structure. **À éviter**, sauf cas trivial.

### 2. Recursive character splitting (notre choix)

On essaie de couper d'abord aux séparateurs naturels (`\n\n`, puis `\n`, puis `. `, puis `, `, puis `' '`) avant de tomber sur un découpage forcé. Le splitter cherche le plus haut séparateur qui permet de respecter la taille max.

Algorithme :

1. Si le texte fait moins de `chunk_size`, c'est un chunk. Fin.
2. Sinon, essaye de couper sur le premier séparateur de la liste (`\n\n`).
3. Si chaque morceau résultant fait moins de `chunk_size`, c'est bon.
4. Sinon, applique récursivement avec le séparateur suivant (`\n`, puis `. `, etc.).
5. En dernier recours, coupe sur le caractère vide `""` (force le découpage à `chunk_size` exact).

Ça donne des chunks qui respectent les paragraphes, puis les phrases, puis les mots.

### 3. Semantic chunking

On embed chaque phrase individuellement, on calcule la similarité cosinus entre phrases consécutives, et on coupe aux frontières où la similarité chute brutalement (changement de sujet détecté).

```python
sentences = split_into_sentences(text)
embeddings = [embed(s) for s in sentences]
gaps = [
    1 - cosine(embeddings[i], embeddings[i+1])
    for i in range(len(embeddings)-1)
]
# coupe aux indices où gaps[i] > seuil (ex: 95e percentile)
```

Plus précis, beaucoup plus coûteux (un appel d'embedding par phrase). Utile pour des corpus très hétérogènes ou des textes très denses. Eigenmind ne l'utilise pas par défaut.

### 4. Structure-aware chunking

Pour Markdown ou HTML, on coupe aux headers (`#`, `##`, `<h1>`, `<h2>`). Chaque section devient un chunk.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter
```

Pour LaTeX, on coupe aux `\section`. Pour du code, on coupe aux frontières de fonctions/classes via tree-sitter.

Pour PDF avec table des matières détectable, on peut couper aux chapitres. Eigenmind reste sur du recursive — c'est le compromis standard.

### 5. Patterns avancés

**Parent-child chunking** — on indexe des petits chunks (200 caractères, précis) pour le retrieval, mais on récupère le **chunk parent** (1000 caractères, contexte large) pour l'envoyer au LLM. Best of both worlds : précision du match, richesse du contexte.

**Sentence-window retrieval** — variante : on indexe phrase par phrase, mais à la récupération on attache la ±2 phrases autour de chaque match.

**Hierarchical summarization** — chaque chunk porte un résumé d'une phrase, le résumé est aussi embedded. Permet du retrieval à deux niveaux (résumé pour trouver le bon doc, détail pour répondre).

Eigenmind 2.x pourrait évoluer vers parent-child. En MVP on reste simple.

## Pourquoi `RecursiveCharacterTextSplitter` de LangChain

C'est essentiellement l'implémentation de la stratégie 2 ci-dessus, paramétrable, robuste, testée. On lui donne `chunk_size`, `chunk_overlap` et éventuellement une liste de séparateurs custom. Il s'occupe du reste.

On utilise le package `langchain-text-splitters` plutôt que `langchain` lui-même : c'est un sous-package isolé qui ne tire pas tout le framework (chains, agents, retrievers). 100 lignes de pure logique de découpage, zéro dépendance lourde.

### Les séparateurs à customiser

Le défaut `["\n\n", "\n", " ", ""]` marche pour de l'anglais. Pour du français, on peut ajouter `". "` et `", "` :

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
)
```

Pour du Markdown, ajouter `["\n## ", "\n### ", ...]` en tête. Pour du code Python, `["\nclass ", "\ndef ", "\n\n"]`.

## Exemples concrets de bons vs mauvais chunks

### Mauvais : coupure en plein milieu de phrase

```
Chunk 5: "...Le chiffre d'affaires consolidé du groupe pour l'exercice"
Chunk 6: "2024 s'élève à 4.2 milliards d'euros, en hausse de 8% par..."
```

Une question "Quel est le CA 2024 ?" risque de matcher le chunk 5 (avec "chiffre d'affaires") mais sans le chiffre. Mitigation : overlap.

### Mauvais : chunk multi-sujets

```
Chunk 12: "La croissance des ventes en Europe a été soutenue.
          Par ailleurs, le conseil d'administration a nommé
          Mme X au poste de Chief Sustainability Officer."
```

Deux sujets indépendants. Une question sur Mme X ramènera ce chunk avec un signal dilué par la partie ventes.

### Bon : chunk auto-suffisant, un sujet

```
Chunk 8: "Au Q4 2024, le chiffre d'affaires de la division Cloud
          a atteint 1.8 milliards d'euros, en progression de 23%
          par rapport au Q4 2023. La marge opérationnelle s'est
          établie à 28%, en hausse de 3 points."
```

Un seul sujet (Cloud Q4 2024), tous les chiffres nécessaires sont dans le chunk, le contexte est explicite.

## Lecture de PDF : la complexité cachée

Un PDF est un format d'affichage, pas un format de données. Sous le capot, le texte est positionné par coordonnées (x, y) sur la page, sans notion de paragraphe ni d'ordre logique. Une lib comme `pypdf` reconstitue un best-effort de texte page par page en suivant l'ordre des objets PDF.

### Conséquences pratiques

**L'ordre du texte peut être bizarre** — colonnes multiples, encadrés flottants, en-têtes de page mélangés au corps. Tu peux voir un footer apparaître en plein milieu d'un paragraphe.

**Les PDF scannés renvoient une string vide** — c'est de l'image, pas du texte. Il faut OCR (phase 5 avec Tesseract). pypdf renvoie `""` ou des caractères aléatoires.

**Les tableaux sont catastrophiques avec pypdf**. Un tableau bien lisible à l'écran devient un magma textuel en extraction. Solutions :
- `pdfplumber` : reconstitue les tableaux par détection de bordures.
- `camelot-py` : spécialisé tableaux, qualité supérieure.
- `unstructured` : utilise du ML pour la structure.
- `LLamaParse` / `Reducto` : services payants haute qualité.
- LLM multimodaux (GPT-4V, Claude) : on convertit la page en image et on demande à un LLM de la transcrire.

Pour un MVP, pypdf suffit. Pour de la prod sérieuse sur des docs structurés, `unstructured` ou un service payant.

**Les caractères Unicode exotiques peuvent ressortir mal encodés** — caractères composés, ligatures, mathématiques.

**Les en-têtes et pieds de page polluent le texte** — la mention "Confidentiel — Page X / Y" qui apparaît sur chaque page va se retrouver dans chaque chunk. À filtrer en post-traitement si problème.

### Détection d'un PDF scanné

Un PDF avec >80% des pages renvoyant `len(text.strip()) < 50` est probablement scanné. Tu peux faire la vérification :

```python
text_density = sum(len(p.extract_text() or "") for p in reader.pages)
if text_density < 100 * len(reader.pages):
    print("PDF probablement scanné — bascule sur OCR")
```

On implémentera cette logique en phase 5.

## Au-delà du PDF — autres formats

Eigenmind doit aussi gérer (`document_loaders.py` en phase 4) :

- **DOCX** — via `python-docx`. Plus structuré que PDF, paragraphes propres, headers détectables.
- **XLSX** — via `openpyxl`. Subtil : un tableur n'est pas du texte narratif. Stratégie : chaque feuille → un "doc", chaque ligne → un chunk avec colonnes comme contexte.
- **PPTX** — via `python-pptx`. Chaque slide → un chunk typiquement.
- **TXT / MD** — direct, juste un `.read()`. Markdown bénéficie du structure-aware splitting.
- **HTML** — via `beautifulsoup4`. Important : virer scripts, styles, et navigation. Garder le contenu narratif.
- **EPUB** — pour les livres. `ebooklib`.

Chaque format mérite son propre loader avec ses spécificités.

## Métadonnées à conserver

À chaque chunk, on attache un payload pour le retrouver et l'afficher plus tard :
- `filename` — d'où il vient
- `page` — pour ouvrir le PDF à la bonne page
- `chunk_index` — sa position dans le doc
- `text` — le contenu textuel brut (oui, on le duplique dans le payload en plus de l'avoir embeddé — c'est ce qu'on passera au LLM lors du RAG)

Selon le contexte on peut ajouter :
- `section` — titre du chapitre/section pour aider à situer
- `created_at` / `modified_at` — pour des filtres temporels
- `author` — pour filtrer par source
- `language` — pour le multi-langue
- `doc_type` — `"pdf"`, `"docx"`, `"web"`...
- `user_id` — pour le multi-tenancy

Ce payload, c'est ce qui transforme un retrieval brut ("voici 5 vecteurs proches") en quelque chose d'exploitable ("voici 5 chunks, leur texte, et leur source").

## L'erreur classique : chunker après avoir nettoyé

Tentation : on enlève les sauts de ligne, on normalise les espaces, on uppercase tout. **Mauvaise idée**. Le splitter récursif a besoin de la structure originale (paragraphes, phrases) pour bien couper. Nettoyer avant le chunking détruit l'information de structure.

Règle : **chunke d'abord, nettoie ensuite chunk par chunk si nécessaire** (par exemple `text.strip()`, retirer les en-têtes répétés). Et garde le texte original dans le payload — le LLM préfère de la prose normale.

## Cas particuliers

### Documents très longs (livre, thèse)

100+ pages, 10 000+ chunks possibles. Considérations :
- Ingestion par batch (sinon RAM explose).
- Indexation hiérarchique : embed des résumés de chapitres en plus des chunks.
- Augmenter k au retrieval pour ne pas rater le bon passage.

### Documents très courts (tweet, email)

Inutile de chunker un email de 200 caractères. Stocke-le tel quel, comme un seul chunk.

### Mix de langues

Si ton corpus mêle français et anglais, vérifie que ton embedder est multilingue. Sinon, fais un détecteur de langue au ingestion et route vers différentes collections par langue. Mauvaise idée d'avoir des vecteurs FR et EN dans le même espace si l'embedder n'est pas multilingue.

### Documents avec citations / footnotes

Les notes de bas de page polluent le texte courant. Stratégie : extraire les footnotes en amont, les chunker séparément avec une métadata `is_footnote=True`, et joindre à la demande lors du retrieval.
