# 1.7 — Assemblage d'un pipeline RAG

## Pourquoi un fichier d'orchestration séparé

Tu as 5 modules qui font chacun une chose :
- `embed_text.py` — vectorise du texte
- `load_pdf.py` — découpe un PDF en chunks
- `store_chunks.py` — pousse les chunks dans Qdrant
- `retrieve.py` — récupère les chunks pertinents pour une question
- `ask_llm.py` — appelle le LLM avec le contexte

Chacun fait UNE chose, bien. C'est le **single responsibility principle**. Mais aucun ne fait le travail complet de l'utilisateur final.

`mini_rag.py` est l'**orchestrateur** : il connecte ces briques dans le bon ordre pour répondre à un cas d'usage concret ("ingère ce PDF puis réponds à cette question").

Cette séparation entre **briques** et **orchestrateurs** est un pattern fondamental qu'on retrouvera amplifié en phase 4 avec le sous-package `pipelines/`. C'est ce qui rend le code testable, réutilisable, et facile à faire évoluer.

## Le pattern pipeline en RAG

Un pipeline RAG a deux phases distinctes :

### Phase offline : Ingestion

```
documents → load → chunk → embed → store
```

Lente, ponctuelle (on l'exécute à chaque ajout de docs), batch.

### Phase online : Génération

```
question → embed → retrieve → prompt → LLM → answer
```

Rapide, à chaque requête utilisateur, à faible latence.

Les deux phases partagent **les mêmes** modules d'embed et de connection Qdrant — d'où l'importance de la modularité. Tu ne dupliques jamais la logique.

## Idempotence et reprise

Un bon pipeline d'ingestion est **idempotent** : tu peux le relancer 10 fois sur le même input, le résultat est le même qu'une seule exécution.

On a déjà cette propriété grâce à `chunk_id()` déterministe en 1.4 — réingérer le même PDF écrase les points au lieu de dupliquer. C'est la base du **Smart Resume** qu'on implémentera en phase 5 : sauter complètement les fichiers déjà ingérés au lieu de juste écraser.

## Logging et observabilité

En production, on veut **savoir ce qu'il s'est passé** :
- Combien de chunks ingérés ?
- Quel temps a pris chaque étape ?
- Quels documents ont été retournés ?
- Quel modèle a répondu, avec combien de tokens ?

En MVP, des `print()` suffisent. En phase 4 on passera à `logging` standard, avec des niveaux INFO/DEBUG/ERROR. Pour Streamlit phase 3, on affichera ces infos dans l'UI.

Règle générale : **chaque étape de pipeline doit produire au moins une ligne de log** (combien d'éléments en entrée, combien en sortie, durée). Ça t'évite des heures de debug plus tard.

## Gestion d'erreurs : où trapper quoi

Trois niveaux d'erreurs typiques en RAG :

**Erreurs d'infrastructure** — Qdrant injoignable, modèle non téléchargé, Ollama down. Crash propre avec message explicite, **pas de retry silencieux**. L'utilisateur doit savoir.

**Erreurs de données** — PDF illisible, chunks vides, fichier manquant. Logger et skip, ne pas crasher le pipeline entier. Important quand on bulk-ingère 100 PDF et qu'un seul est corrompu.

**Erreurs de génération** — LLM qui timeout, qui renvoie une réponse vide, qui hallucine. Retry avec backoff (1s, 2s, 4s, 8s), puis fallback (réponse "je n'ai pas pu générer", ou bascule vers un autre provider).

En MVP on garde simple : on let crash sauf pour les erreurs de données où on skip avec un warning.

## Test unitaire d'un pipeline

Un pipeline orchestrateur est par nature un **integration test** quand on le lance — il touche tout. Pour le tester proprement en unit test, on injecte des mocks à la place des briques :

```python
def test_pipeline(mocker):
    mocker.patch("mini_rag.load_and_chunk", return_value=[fake_chunk])
    mocker.patch("mini_rag.ask", return_value="42")
    assert mini_rag.run("test.pdf", "What is 6x7?") == "42"
```

On valide la **glue logic** sans dépendre de Qdrant ou du LLM. C'est ce qu'on fera en phase 4 (tests).

## Anatomie d'un script CLI propre

Un script CLI bien construit a une structure prévisible :

```python
def main():
    args = parse_args()
    configure_logging()
    try:
        result = run_pipeline(args)
        print(result)
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

- `parse_args()` valide les inputs avant tout traitement.
- `run_pipeline()` contient la logique métier, séparée du parsing.
- Le `try/except` au top niveau évite les stack traces brutes en cas d'erreur.
- `sys.exit(code)` permet aux scripts shell qui appellent le tien de savoir si ça a réussi (0) ou échoué (1+).

Pour le MVP on simplifie, mais ce pattern est ce qui transforme un script jetable en outil réutilisable.

## L'argparse module

Le module standard de Python pour parser les arguments CLI. Permet :

- Arguments positionnels (`mini_rag.py question.pdf "ma question"`)
- Arguments nommés (`--top-k 10`, `--no-ingest`)
- Validation de types (int, float, choices)
- Help auto-généré (`mini_rag.py --help`)
- Subcommands (`mini_rag.py ingest`, `mini_rag.py query`)

Alternatives : `click` (plus ergonomique, dépendance externe), `typer` (basé sur les type hints, très moderne). On reste sur `argparse` pour zéro dépendance.

## Penser la CLI comme une API

Une CLI propre est une **API en ligne de commande**. Les principes UX d'API web s'appliquent :

- **Verbes clairs** : `ingest`, `query`, `list`, `delete`. Pas `do`, `run`, `process`.
- **Aide intégrée** : `--help` doit suffire à comprendre l'outil.
- **Erreurs explicites** : "fichier non trouvé : X.pdf" pas "FileNotFoundError".
- **Sortie machine-readable** en option : un flag `--json` pour piper la sortie dans un autre script.
- **Modes verbeux** : `--verbose` pour debug, silence par défaut pour le scriptage.

On vise un MVP simple, mais on garde ces principes en tête pour la phase 4.