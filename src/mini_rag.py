"""
Mini RAG end-to-end CLI.

Two subcommands:
    ingest  <pdf_path>          → load, chunk, embed, store in Qdrant
    query   "your question"     → retrieve + LLM, print answer with sources
    ask     <pdf_path> "..."    → ingest then query in one call

Le retrieval se décline en trois modes (option --mode) :
    dense    cosinus pur (top-k Qdrant)
    hybrid   dense enrichi par la couche graphe spectrale (Singular/Hinge/Theta)
    compare  exécute les deux et affiche la différence — pour visualiser
             l'impact d'eigenmind sur les chunks récupérés et sur la réponse.
"""
import argparse
import sys
import time
from pathlib import Path

from load_pdf import load_and_chunk
from store_chunks import upsert_chunks, COLLECTION_NAME, get_client
from retrieve import retrieve, TOP_K
from ask_llm import ask, PROVIDER, PROVIDER_CONFIG


def cmd_ingest(pdf_path: str, ocr: str = "auto") -> int:
    """Load + chunk + embed + store. Return number of points written."""
    if not Path(pdf_path).exists():
        print(f"Erreur : fichier non trouvé : {pdf_path}")
        return 1

    print(f"→ Ingestion de {pdf_path} (ocr={ocr})")
    t0 = time.time()

    chunks = load_and_chunk(pdf_path, ocr=ocr)
    print(f"  {len(chunks)} chunks extraits")

    if not chunks:
        print("  Aucun texte extrait — PDF vide ou scanné ?")
        return 1

    n = upsert_chunks(chunks)
    print(f"  {n} points upserted dans '{COLLECTION_NAME}'")

    info = get_client().get_collection(COLLECTION_NAME)
    elapsed = time.time() - t0
    print(f"  Total collection : {info.points_count} points")
    print(f"  Temps : {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Retrieval — normalisation des deux backends vers un même format de dict
# ---------------------------------------------------------------------------

def _dense_to_dicts(hits: list[dict]) -> list[dict]:
    """Uniformise la sortie de retrieve() (déjà des dicts) avec un tag de type."""
    return [
        {
            "score": h["score"],
            "base_cosine": h["score"],
            "filename": h["filename"],
            "page": h["page"],
            "chunk_index": h["chunk_index"],
            "text": h["text"],
            "types": ["dense"],
        }
        for h in hits
    ]


def _hybrid_to_dicts(results) -> list[dict]:
    """Uniformise la sortie de hybrid_retrieve() (HybridResult) au même format."""
    return [
        {
            "score": r.score,
            "base_cosine": r.base_cosine,
            "filename": r.payload["filename"],
            "page": r.payload["page"],
            "chunk_index": r.payload.get("chunk_index", "?"),
            "text": r.payload["text"],
            "types": r.types,
        }
        for r in results
    ]


def _retrieve_dense(question: str, k: int) -> tuple[list[dict], float]:
    t0 = time.time()
    chunks = _dense_to_dicts(retrieve(question, k=k))
    return chunks, time.time() - t0


def _retrieve_hybrid(question: str, k: int) -> tuple[list[dict], float]:
    # Import paresseux : ne charge networkx/scipy que si l'on fait de l'hybride.
    from hybrid_retrieve import GraphAwareCache, hybrid_retrieve

    print("  → Construction du graphe sémantique + décomposition spectrale (one-shot)…")
    cache = GraphAwareCache()
    cache.build()
    t0 = time.time()
    chunks = _hybrid_to_dicts(hybrid_retrieve(question, cache, k_final=k))
    return chunks, time.time() - t0


def _print_sources(chunks: list[dict], elapsed_ms: float) -> None:
    print(f"  {len(chunks)} chunks récupérés en {elapsed_ms:.0f}ms :")
    for i, c in enumerate(chunks, start=1):
        preview = c["text"][:80].replace("\n", " ")
        types_str = ",".join(c.get("types") or ["dense"])
        print(
            f"    #{i}  score={c['score']:.3f} (cos={c['base_cosine']:.3f}) "
            f"[{types_str}]  {c['filename']} p{c['page']} — {preview}..."
        )
    print()


# ---------------------------------------------------------------------------
# Commandes query / compare
# ---------------------------------------------------------------------------

def cmd_query(question: str, k: int = TOP_K, mode: str = "hybrid", show_sources: bool = True) -> int:
    """Retrieve + LLM. Print the answer with citations."""
    print(f"→ Question : {question}")
    print(f"  Provider : {PROVIDER} ({PROVIDER_CONFIG[PROVIDER]['model']})")
    print(f"  Mode retrieval : {mode}\n")

    if mode == "compare":
        return _cmd_compare(question, k, show_sources)

    if mode == "hybrid":
        chunks, t_retrieve = _retrieve_hybrid(question, k)
    else:
        chunks, t_retrieve = _retrieve_dense(question, k)

    if not chunks:
        print("Aucun chunk trouvé. La collection est-elle vide ?")
        return 1

    if show_sources:
        _print_sources(chunks, t_retrieve * 1000)

    t0 = time.time()
    answer = ask(question, chunks)
    t_llm = time.time() - t0

    print("─" * 60)
    print("Réponse :\n")
    print(answer)
    print()
    print("─" * 60)
    print(f"Temps : retrieve {t_retrieve*1000:.0f}ms · LLM {t_llm:.1f}s")
    return 0


def _cmd_compare(question: str, k: int, show_sources: bool) -> int:
    """Exécute dense ET hybride, met en évidence la différence, répond deux fois."""
    dense_chunks, t_dense = _retrieve_dense(question, k)
    hybrid_chunks, t_hybrid = _retrieve_hybrid(question, k)

    if not dense_chunks and not hybrid_chunks:
        print("Aucun chunk trouvé. La collection est-elle vide ?")
        return 1

    # Diff des deux ensembles, clé = (fichier, page, chunk)
    key = lambda c: (c["filename"], c["page"], c["chunk_index"])
    dense_keys = {key(c) for c in dense_chunks}
    hybrid_keys = {key(c) for c in hybrid_chunks}
    only_hybrid = hybrid_keys - dense_keys
    only_dense = dense_keys - hybrid_keys

    if show_sources:
        print("═" * 60)
        print("DENSE (cosinus pur)")
        _print_sources(dense_chunks, t_dense * 1000)
        print("═" * 60)
        print("HYBRIDE (dense + graphe spectral)")
        _print_sources(hybrid_chunks, t_hybrid * 1000)
        print("═" * 60)
        print(
            f"  Δ retrieval : +{len(only_hybrid)} chunk(s) apportés par le graphe, "
            f"−{len(only_dense)} chunk(s) du dense écartés."
        )
        for c in (c for c in hybrid_chunks if key(c) in only_hybrid):
            graph_types = ",".join(t for t in c["types"] if t != "dense") or "boost"
            print(f"    + {c['filename']} p{c['page']} chunk#{c['chunk_index']} [{graph_types}]")
        print()

    print("─" * 60)
    print("Réponse DENSE :\n")
    t0 = time.time()
    answer_dense = ask(question, dense_chunks)
    t_llm_dense = time.time() - t0
    print(answer_dense)
    print()

    print("─" * 60)
    print("Réponse HYBRIDE :\n")
    t0 = time.time()
    answer_hybrid = ask(question, hybrid_chunks)
    t_llm_hybrid = time.time() - t0
    print(answer_hybrid)
    print()

    print("─" * 60)
    print(
        f"Temps : dense retrieve {t_dense*1000:.0f}ms + LLM {t_llm_dense:.1f}s · "
        f"hybride retrieve {t_hybrid*1000:.0f}ms + LLM {t_llm_hybrid:.1f}s"
    )
    return 0


def cmd_ask(pdf_path: str, question: str, k: int = TOP_K, mode: str = "hybrid", ocr: str = "auto") -> int:
    """Ingest then query in one shot."""
    rc = cmd_ingest(pdf_path, ocr=ocr)
    if rc != 0:
        return rc
    print()
    return cmd_query(question, k=k, mode=mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini RAG CLI : ingère des PDF et pose des questions dessus."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingérer un PDF dans la collection.")
    p_ingest.add_argument("pdf_path", help="Chemin vers le fichier PDF.")
    p_ingest.add_argument(
        "--ocr", choices=["auto", "always", "never"], default="auto",
        help="auto = OCR les pages sans texte (défaut) · always = OCR toutes les pages "
             "(PDF mixte texte+images) · never = couche texte seule.",
    )

    p_query = sub.add_parser("query", help="Poser une question sur la collection.")
    p_query.add_argument("question", help="Question en langage naturel.")
    p_query.add_argument("-k", "--top-k", type=int, default=TOP_K, help=f"Nombre de chunks (défaut {TOP_K}).")
    p_query.add_argument(
        "-m", "--mode", choices=["dense", "hybrid", "compare"], default="hybrid",
        help="dense = cosinus pur · hybrid = + graphe spectral (défaut) · compare = les deux côte à côte.",
    )
    p_query.add_argument("--no-sources", action="store_true", help="Masquer la liste des sources.")

    p_ask = sub.add_parser("ask", help="Ingérer puis poser une question.")
    p_ask.add_argument("pdf_path", help="Chemin vers le fichier PDF.")
    p_ask.add_argument("question", help="Question en langage naturel.")
    p_ask.add_argument("-k", "--top-k", type=int, default=TOP_K)
    p_ask.add_argument(
        "-m", "--mode", choices=["dense", "hybrid", "compare"], default="hybrid",
        help="dense · hybrid (défaut) · compare.",
    )
    p_ask.add_argument(
        "--ocr", choices=["auto", "always", "never"], default="auto",
        help="auto (défaut) · always · never.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "ingest":
        return cmd_ingest(args.pdf_path, ocr=args.ocr)
    if args.cmd == "query":
        return cmd_query(args.question, k=args.top_k, mode=args.mode, show_sources=not args.no_sources)
    if args.cmd == "ask":
        return cmd_ask(args.pdf_path, args.question, k=args.top_k, mode=args.mode, ocr=args.ocr)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
