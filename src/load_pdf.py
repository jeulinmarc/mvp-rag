"""
PDF loading and chunking.
Reads a PDF page by page, splits each page into overlapping chunks,
returns a list of dicts ready to be embedded and stored in Qdrant.

Fallback OCR (Phase 5.1) : une page dont la couche texte est trop pauvre
(typiquement une page scannée ou une image contenant du texte) est rendue en
image puis passée à Tesseract. Idée du mémo officiel (§2.1) : l'extraction est
un canal bruité ; quand le chemin "pas cher" (couche texte du PDF) échoue, on
bascule sur l'OCR. On applique le test caractères/page **par page** (et non au
niveau du document) pour gérer les PDF mixtes texte + images.

Dépendances OCR : `tesseract` + `poppler` (système, via brew) et
`pytesseract` + `pdf2image` (pip). Si elles manquent, l'OCR est simplement
ignoré (warning) et le reste fonctionne.
"""
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# OCR
MIN_CHARS_FOR_OCR = 20      # seuil caractères/page sous lequel on tente l'OCR (cf. mémo §2.1)
OCR_DPI = 200              # résolution de rendu des pages pour l'OCR
DEFAULT_OCR_LANG = "eng+fra"  # langues Tesseract (fra via `brew install tesseract-lang`)


def _ocr_page(pdf_path: Path, page_num: int, dpi: int = OCR_DPI, lang: str = DEFAULT_OCR_LANG) -> str:
    """
    Rend une seule page du PDF en image et la passe à Tesseract.
    Renvoie le texte OCR (vide en cas d'échec : deps manquantes, langue absente…).
    Imports paresseux pour que le module reste utilisable sans les deps OCR.
    """
    import tempfile
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        print("  ⚠️  OCR ignoré : installe `pip install pytesseract pdf2image`.")
        return ""

    # On rend la page vers un fichier image temporaire et on passe le CHEMIN à
    # Tesseract : passer un objet PIL casse avec Pillow ≥ 12 (décodage stderr).
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            paths = convert_from_path(
                pdf_path, dpi=dpi, first_page=page_num, last_page=page_num,
                fmt="png", output_folder=tmpdir, paths_only=True,
            )
        except Exception as e:  # poppler absent, PDF illisible…
            print(f"  ⚠️  OCR ignoré (rendu page {page_num} impossible : {e}). "
                  "Vérifie `brew install poppler`.")
            return ""

        if not paths:
            return ""

        # Essaie la langue demandée, repli sur l'anglais si le pack manque.
        for try_lang in (lang, "eng"):
            try:
                return pytesseract.image_to_string(paths[0], lang=try_lang).strip()
            except pytesseract.TesseractError:
                continue
            except Exception as e:
                print(f"  ⚠️  OCR échoué page {page_num} : {e}. Vérifie `brew install tesseract`.")
                return ""
    return ""


def load_pdf_pages(pdf_path: str | Path, ocr: str = "auto") -> list[dict]:
    """
    Read a PDF and return one dict per page:
    {"filename": str, "page": int, "text": str}

    `ocr` :
      - "auto"   : OCR une page seulement si sa couche texte fait < MIN_CHARS_FOR_OCR (défaut)
      - "always" : OCR chaque page (utile pour un PDF mixte dont chaque page porte des images)
      - "never"  : couche texte uniquement (comportement historique)
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)

    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        needs_ocr = ocr == "always" or (ocr == "auto" and len(text) < MIN_CHARS_FOR_OCR)
        if needs_ocr:
            ocr_text = _ocr_page(pdf_path, page_num)
            # Le rendu OCR de la page capture aussi le texte des images : on garde
            # la version la plus riche.
            if len(ocr_text) > len(text):
                text = ocr_text

        if not text:
            continue
        pages.append({"filename": pdf_path.name, "page": page_num, "text": text})
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


def load_and_chunk(pdf_path: str | Path, ocr: str = "auto") -> list[dict]:
    """Convenience wrapper: load + chunk in one call."""
    pages = load_pdf_pages(pdf_path, ocr=ocr)
    return chunk_pages(pages)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_pdf.py <path/to/file.pdf> [auto|always|never]")
        sys.exit(1)

    ocr_mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
    chunks = load_and_chunk(sys.argv[1], ocr=ocr_mode)
    print(f"\nFichier      : {chunks[0]['filename'] if chunks else 'aucun'}")
    print(f"Mode OCR     : {ocr_mode}")
    print(f"Chunks total : {len(chunks)}")
    if chunks:
        avg_len = sum(len(c['text']) for c in chunks) / len(chunks)
        print(f"Taille moy.  : {avg_len:.0f} caractères")
        print(f"\n--- Premier chunk (page {chunks[0]['page']}) ---")
        print(chunks[0]["text"][:300] + "...")
        print(f"\n--- Dernier chunk (page {chunks[-1]['page']}) ---")
        print(chunks[-1]["text"][:300] + "...")
