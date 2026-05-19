"""
Mini RAG end-to-end CLI.

Two subcommands:
    ingest  <pdf_path>          → load, chunk, embed, store in Qdrant
    query   "your question"     → retrieve + LLM, print answer with sources
    ask     <pdf_path> "..."    → ingest then query in one call
"""
import argparse
import sys
import time
from pathlib import Path

from load_pdf import load_and_chunk
from store_chunks import upsert_chunks, COLLECTION_NAME, get_client
from retrieve import retrieve, TOP_K
from ask_llm import ask, PROVIDER, PROVIDER_CONFIG


def cmd_ingest(pdf_path: str) -> int:
    """Load + chunk + embed + store. Return number of points written."""
    if not Path(pdf_path).exists():
        print(f"Erreur : fichier non trouvé : {pdf_path}")
        return 1

    print(f"→ Ingestion de {pdf_path}")
    t0 = time.time()

    chunks = load_and_chunk(pdf_path)
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


def cmd_query(question: str, k: int = TOP_K, show_sources: bool = True) -> int:
    """Retrieve + LLM. Print the answer with citations."""
    print(f"→ Question : {question}")
    print(f"  Provider : {PROVIDER} ({PROVIDER_CONFIG[PROVIDER]['model']})\n")

    t0 = time.time()
    chunks = retrieve(question, k=k)
    t_retrieve = time.time() - t0

    if not chunks:
        print("Aucun chunk trouvé. La collection est-elle vide ?")
        return 1

    if show_sources:
        print(f"  {len(chunks)} chunks récupérés en {t_retrieve*1000:.0f}ms :")
        for i, c in enumerate(chunks, start=1):
            preview = c["text"][:80].replace("\n", " ")
            print(f"    #{i}  score={c['score']:.3f}  {c['filename']} p{c['page']} — {preview}...")
        print()

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


def cmd_ask(pdf_path: str, question: str, k: int = TOP_K) -> int:
    """Ingest then query in one shot."""
    rc = cmd_ingest(pdf_path)
    if rc != 0:
        return rc
    print()
    return cmd_query(question, k=k)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini RAG CLI : ingère des PDF et pose des questions dessus."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingérer un PDF dans la collection.")
    p_ingest.add_argument("pdf_path", help="Chemin vers le fichier PDF.")

    p_query = sub.add_parser("query", help="Poser une question sur la collection.")
    p_query.add_argument("question", help="Question en langage naturel.")
    p_query.add_argument("-k", "--top-k", type=int, default=TOP_K, help=f"Nombre de chunks (défaut {TOP_K}).")
    p_query.add_argument("--no-sources", action="store_true", help="Masquer la liste des sources.")

    p_ask = sub.add_parser("ask", help="Ingérer puis poser une question.")
    p_ask.add_argument("pdf_path", help="Chemin vers le fichier PDF.")
    p_ask.add_argument("question", help="Question en langage naturel.")
    p_ask.add_argument("-k", "--top-k", type=int, default=TOP_K)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "ingest":
        return cmd_ingest(args.pdf_path)
    if args.cmd == "query":
        return cmd_query(args.question, k=args.top_k, show_sources=not args.no_sources)
    if args.cmd == "ask":
        return cmd_ask(args.pdf_path, args.question, k=args.top_k)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())