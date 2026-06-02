"""
PDF loading and chunking.
Reads a PDF page by page, splits each page into overlapping chunks,
returns a list of dicts ready to be embedded and stored in Qdrant.
"""
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Read a PDF and return one dict per page:
    {"filename": str, "page": int, "text": str}
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)

    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        pages.append(
            {"filename": pdf_path.name, 
            "page": page_num, 
            "text": text
            })
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split each page into overlapping chunks.
    Return one dict per chunk:
        {"filename": str, "page": int, "chunk_index": int, "text": str}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    chunks = []
    global_index = 0
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "filename": page["filename"],
                "page": page["page"],
                "chunk_index": global_index,
                "text": chunk_text,
            })
            global_index += 1
    return chunks


def load_and_chunk(pdf_path: str | Path) -> list[dict]:
    """Convenience wrapper: load + chunk in one call."""
    pages = load_pdf_pages(pdf_path)
    return chunk_pages(pages)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_pdf.py <path/to/file.pdf>")
        sys.exit(1)

    chunks = load_and_chunk(sys.argv[1])
    print(f"\nFichier      : {chunks[0]['filename'] if chunks else 'aucun'}")
    print(f"Chunks total : {len(chunks)}")
    if chunks:
        avg_len = sum(len(c['text']) for c in chunks) / len(chunks)
        print(f"Taille moy.  : {avg_len:.0f} caractères")
        print(f"\n--- Premier chunk (page {chunks[0]['page']}) ---")
        print(chunks[0]["text"][:300] + "...")
        print(f"\n--- Dernier chunk (page {chunks[-1]['page']}) ---")
        print(chunks[-1]["text"][:300] + "...")
